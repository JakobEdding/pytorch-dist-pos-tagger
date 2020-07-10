#!/usr/bin/env python

import ray
from datetime import datetime

class Model():
    def __init__(self):
        self.weights = [0.0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.model = Model()

    def combine_weights(self, *weights):
        # *weights are several arguments, weights is then a list
        # print('new weights:', weights)
        avg = float(sum([weight[-1] for weight in weights]) / len(weights))

        weights = self.model.get_weights().copy()
        weights = weights + ([avg] * 1000)
        self.model.set_weights(weights)

        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote(num_cpus=3)
class DataWorker(object):
    def __init__(self, rank):
        self.rank = rank
        print(f'i have rank {self.rank}')
        self.model = Model()

    def compute_new_weight(self, weights):
        self.model.set_weights(weights)

        weights = self.model.get_weights().copy()
        weights = weights + ([float(weights[-1] + self.rank)] * 1000)
        # print(len(weights))
        self.model.set_weights(weights)

        return self.model.get_weights()

def run():
    ray.init(address='auto', ignore_reinit_error=True, webui_host='0.0.0.0', redis_password='5241590000000000')
    ps = ParameterServer.remote()

    num_workers = 2
    workers = [DataWorker.remote(i) for i in range(num_workers)]

    current_weights = ps.get_weights.remote()
    before = datetime.now()
    # this still works:
    # for i in range(11):
    # this doesn't work anymore:
    for i in range(12):
        weights = [
            worker.compute_new_weight.remote(current_weights) for worker in workers
        ]
        current_weights = ps.combine_weights.remote(*weights)

    final_weights = ray.get(ps.get_weights.remote())
    print(f'final result {final_weights[-1]}, len: {len(final_weights)}, took {datetime.now() - before}')
    # print(final_weights)

    # ray.shutdown()

if __name__ == "__main__":
    run()
