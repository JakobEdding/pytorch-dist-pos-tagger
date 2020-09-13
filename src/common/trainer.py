from datetime import datetime
import os
import time
import random

import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from tqdm import tqdm

from common.model.gru_pos_tagger_model import GRUPOSTaggerModel


MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self):
        self.TEXT = data.Field(lower=True)  # can have unknown tokens
        self.UD_TAGS = data.Field(unk_token=None)  # can't have unknown tags

        self.fields = (("text", self.TEXT), ("udtags", self.UD_TAGS), (None, None))

        train_data, valid_data, test_data = datasets.UDPOS.splits(self.fields, root='/home/pi/.data')
        train_data_tuple = self.split_data_arbitrary_splits(train_data, PARALLELISM_LEVEL)

        self.TEXT.build_vocab(train_data, min_freq=MIN_FREQ)

        self.UD_TAGS.build_vocab(train_data)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iterators = data.BucketIterator.splits(
            train_data_tuple,
            batch_size=BATCH_SIZE,
            device=self.device)

        self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=self.device)

        self.INPUT_DIM = len(self.TEXT.vocab)

        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 128

        self.OUTPUT_DIM = len(self.UD_TAGS.vocab)
        self.N_LAYERS = 2
        self.BIDIRECTIONAL = False
        self.DROPOUT = 0.25
        self.PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

        self.TAG_PAD_IDX = self.UD_TAGS.vocab.stoi[self.UD_TAGS.pad_token]

        self.model = GRUPOSTaggerModel(self.INPUT_DIM,
                                       self.EMBEDDING_DIM,
                                       self.HIDDEN_DIM,
                                       self.OUTPUT_DIM,
                                       self.N_LAYERS,
                                       self.BIDIRECTIONAL,
                                       self.DROPOUT,
                                       self.PAD_IDX)
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.TAG_PAD_IDX)

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

    def iterate_epochs(self, rank):
        best_valid_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            print(f'Starting epoch {epoch + 1:02}')
            epoch_start_time = time.time()

            train_loss, train_acc = self.train(rank, epoch)
            valid_loss, valid_acc = self.evaluate(self.valid_iterator, 'valid', epoch)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.diff_time(epoch_start_time, epoch_end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if not os.path.exists('../tmp_model'):
                    os.makedirs('../tmp_model')
                # save model outside directory that could be sshfs-mounted
                torch.save(self.model.state_dict(), '../tmp_model/model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    def train(self, rank, epoch):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        self.model.train()

        for batch_idx, batch in tqdm(enumerate(self.train_iterators[rank]), desc=f'Rank {rank} processing epoch {epoch+1} ...'):
            text = batch.text
            tags = batch.udtags
            self.optimizer.zero_grad()
            predictions = self.model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = self.criterion(predictions, tags)
            acc = self.categorical_accuracy(predictions, tags)
            print(f'Epoch: {epoch+1}, batch {batch_idx}, rank: {rank} | {str(datetime.now())} | Done with batch')
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        end_time = time.time()
        mins, secs = self.diff_time(start_time, end_time)
        print(f'Epoch {epoch+1:02} train time: {mins}m {secs}s')

        return epoch_loss / len(self.train_iterators[rank]), epoch_acc / len(self.train_iterators[rank])

    def evaluate(self, iterator, method, epoch, silent=False):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        self.model.eval()

        with torch.no_grad():
            for batch in iterator:
                text = batch.text
                tags = batch.udtags
                predictions = self.model(text)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                loss = self.criterion(predictions, tags)
                acc = self.categorical_accuracy(predictions, tags)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        end_time = time.time()
        mins, secs = self.diff_time(start_time, end_time)
        if not silent:
            print(f'Epoch {epoch+1:02} {method} time: {mins}m {secs}s')

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def test(self):
        self.model.load_state_dict(torch.load('../tmp_model/model.pt'))
        test_loss, test_acc = self.evaluate(self.test_iterator, 'test', -1)
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')
