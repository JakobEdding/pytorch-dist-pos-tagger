import time
import random
import sys
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from torchtext import data
from torchtext import datasets
from tqdm import tqdm

from common.model.gru_pos_tagger_model import GRUPOSTaggerModel


# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


class DistributedTrainer(object):
    def __init__(self):
        self.TEXT = data.Field(lower = True)  # can have unknown tokens
        self.UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

        # don't load PTB tags
        self.fields = (("text", self.TEXT), ("udtags", self.UD_TAGS), (None, None))

        train_data, valid_data, test_data = datasets.UDPOS.splits(self.fields, root='/home/pi/.data')
        train_data_tuple = self.split_data_arbitrary_splits(train_data, PARALLELISM_LEVEL)

        self.TEXT.build_vocab(train_data, min_freq = MIN_FREQ)

        self.UD_TAGS.build_vocab(train_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iterators = data.BucketIterator.splits(
            train_data_tuple,
            batch_size=BATCH_SIZE,
            device=device)

        self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=device)

        self.INPUT_DIM = len(self.TEXT.vocab)

        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 128

        self.OUTPUT_DIM = len(self.UD_TAGS.vocab)
        self.N_LAYERS = 2
        self.BIDIRECTIONAL = False
        self.DROPOUT = 0.25
        self.PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.TAG_PAD_IDX = self.UD_TAGS.vocab.stoi[self.UD_TAGS.pad_token]

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    # inspired by torchtext internals because their splits method is limited to 3 workers... https://github.com/pytorch/text/blob/e70955309ead681f924fecd36d759c37e3fdb1ee/torchtext/data/dataset.py#L325
    def split_data_arbitrary_splits(self, examples, number_of_parts):
        N = len(examples)
        randperm = random.sample(range(N), len(range(N)))
        indices = [randperm[int(N * (part / number_of_parts)):int(N * (part+1) / number_of_parts)] for part in range(number_of_parts-1)]
        indices.append(randperm[int(N * (number_of_parts-1) / number_of_parts):])
        examples_tuple = tuple([examples[i] for i in index] for index in indices)
        splits = tuple(data.dataset.Dataset(elem, examples.fields) for elem in examples_tuple if elem)
        # In case the parent sort key isn't none
        if examples.sort_key:
            for subset in splits:
                subset.sort_key = examples.sort_key
        return splits

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def categorical_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.TAG_PAD_IDX).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def diff_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, model, iterator, optimizer, criterion, tag_pad_idx, rank, epoch):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        model.train()

        for batch_idx, batch in tqdm(enumerate(iterator), desc=f'Rank {rank} processing epoch {epoch+1} ...'):
            text = batch.text
            tags = batch.udtags
            optimizer.zero_grad()
            #text = [sent len, batch size]
            predictions = model(text)
            #predictions = [sent len, batch size, output dim]
            #tags = [sent len, batch size]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            #predictions = [sent len * batch size, output dim]
            #tags = [sent len * batch size]
            loss = criterion(predictions, tags)
            acc = self.categorical_accuracy(predictions, tags)
            print(f'Epoch: {epoch+1}, batch {batch_idx}, rank: {rank} | {str(datetime.now())} | Done with batch')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        end_time = time.time()
        mins, secs = self.diff_time(start_time, end_time)
        print(f'Epoch {epoch+1:02} train time: {mins}m {secs}s')

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion, tag_pad_idx, method, epoch):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                text = batch.text
                tags = batch.udtags
                predictions = model(text)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                loss = criterion(predictions, tags)
                acc = self.categorical_accuracy(predictions, tags, tag_pad_idx)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        end_time = time.time()
        mins, secs = self.diff_time(start_time, end_time)
        print(f'Epoch {epoch+1:02} {method} time: {mins}m {secs}s')

        return epoch_loss / len(iterator), epoch_acc / len(iterator)