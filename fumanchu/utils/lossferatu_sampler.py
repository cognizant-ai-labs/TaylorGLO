
import torch
import torch.nn as nn
import numpy as np

class LossferatuSampler(torch.utils.data.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        total_samples: the total number of samples in this dataset
        start: offset where we should start selecting from
        preshuffle_with_seed: an optional one-time shuffle with a given seed that takes place before any sampling
    """
    def __init__(self, num_samples, total_samples, start, shuffle, preshuffle_with_seed=None):
        self.num_samples = num_samples
        self.start = start
        self.shuffle = shuffle

        self.indices = np.array(range(0, total_samples))
        if preshuffle_with_seed is not None:
            reset_seed = np.random.randint(np.iinfo(np.uint32).max) # get a seed from the current prng
            np.random.seed(preshuffle_with_seed) # set a static seed
            np.random.shuffle(self.indices)
            np.random.seed(reset_seed) # reset to seed
        print("Indices preview:")
        print(self.indices)



    def __iter__(self):
        if self.shuffle:
            return iter(self.indices[(torch.randperm(self.num_samples) + self.start).tolist()])
        return iter(self.indices[range(self.start, self.start + self.num_samples)])

    def __len__(self):
        return self.num_samples