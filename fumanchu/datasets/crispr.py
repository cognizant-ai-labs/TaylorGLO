# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Pytorch dataset for crispr genomic data
#

import pickle
import os
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

dna_to_integer_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}

def dna_to_integer(dna):
    integer_dna = np.zeros(len(dna), dtype=int)
    for i, nucleotide in enumerate(dna):
        integer_dna[i] = dna_to_integer_map[nucleotide]
    return integer_dna

class CrisprDataset(Dataset):

    def __init__(self, read_data, dna_data, split_sequences,
                 batch_size, context_size, steps=None):

        self.read_data = read_data
        self.dna_data = dna_data
        self.split_sequences = split_sequences
        self.batch_size = batch_size
        self.context_size = context_size
        self.steps = steps

        self.samples = self.enumerate_samples()
        print("Num samples",len(self.samples))
        random.shuffle(self.samples)

    def enumerate_samples(self):

        samples = []
        for record, position in self.split_sequences:
            position_data = self.read_data[record][position]
            position_length = position_data['length']
            for element in range(position_length):
                sample = (record, position, element)
                samples.append(sample)
        return samples

    def on_epoch_end(self):
        random.shuffle(self.samples)

    def get_dna(self, record_idx, position_data):
        record_dna_data = self.dna_data[record_idx]
        index_position = position_data['index_position']
        start_offset = position_data['start_offset']
        end_offset = position_data['end_offset']
        return record_dna_data[index_position][start_offset:end_offset]

    def slice_dna(self, position_dna, element):
        start_idx = element
        end_idx = start_idx + 2 * self.context_size + 1
        element_dna = position_dna[start_idx:end_idx]
        return element_dna

    def __len__(self):
        if self.steps is None:
            return len(self.samples)
        else:
            return self.steps

    def __getitem__(self, idx):

        record_idx, position, element = self.samples[idx]
        position_data = self.read_data[record_idx][position]
        position_dna = self.get_dna(record_idx, position_data)
        element_dna = self.slice_dna(position_dna, element)
        x = dna_to_integer(element_dna)
        y = position_data['aba']
        x = torch.from_numpy(x)
        y = torch.Tensor([y])

        return x, y

def load_data(data_directory):
    print("Loading data.")
    with open(data_directory + '/read_data.pkl', 'rb') as f:
        read_data = pickle.load(f, encoding='latin1')
    with open(data_directory + '/dna_data.pkl', 'rb') as f:
        dna_data = pickle.load(f, encoding='latin1')
    with open(data_directory + '/splits.pkl', 'rb') as f:
        splits = pickle.load(f, encoding='latin1')
    return read_data, dna_data, splits

def _init_fn(worker_id):
    np.random.seed(301 + worker_id)

def load_crispr_genomic(dataset_folder, batch_size=512, context_size=100,
                        steps_per_epoch=1000000, validation_steps=100000, dataset_percentage=1.0, num_workers=4):

    assert(dataset_percentage == 1.0)

    print("Creating generators.")
    read_data, dna_data, splits = load_data(dataset_folder)

    training_sequences = splits['training']
    trainset = CrisprDataset(read_data, dna_data, training_sequences,
                            batch_size, context_size, steps=steps_per_epoch)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    validation_sequences = splits['validation']
    valset = CrisprDataset(read_data, dna_data, validation_sequences,
                            batch_size, context_size, steps=validation_steps)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           worker_init_fn=_init_fn, pin_memory=True)

    print(splits.keys())
    testing_sequences = splits['test']
    testset = CrisprDataset(read_data, dna_data, testing_sequences,
                            batch_size, context_size, steps=validation_steps)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=_init_fn, pin_memory=True)

    classes = None

    return trainloader, valloader, testloader, None

if __name__ == '__main__':
    import sys
    trainloader, valloader, testloader, classes = load_crispr_genomic(dataset_folder='./data', batch_size=2)
    i = 0
    for input, target in trainloader:
        print(input, target)
        i += 1
        if i == 10:
            sys.exit(0)