#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:58:43 2021

@author: jx283
"""
import torch
from torch import nn
import numpy as np
import pandas as pd

PATH_TO_D_WEIGHTS = 'discriminator.pt'
PATH_TO_G_WEIGHTS = 'generator.pt'

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # false_indices = []
        # for i,entry in enumerate(x):
        #     if (entry.detach().numpy() > np.ones(entry.detach().shape)).any() == True:
        #         false_indices.append(i)
        #     if (entry.detach().numpy() < np.zeros(entry.detach().shape)).any() == True:
        #         false_indices.append(i)
        output = self.model(x)
        # helper = torch.ones(output.shape)
        # for i in false_indices:
        #     helper[i] = 0
        # output = output* helper
        
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
def inference(data, d, g):
    ''' Given 13 rows of data, output prediction on nacelle weight and blade weight '''
    # data is a numpy array
    data = data[:-2]
    latent_sample_space = torch.randn((1000,2))
    
    sample_inputs = torch.ones((1000,13))
    for i in range(len(sample_inputs)):
        sample_inputs[i,:-2] = torch.tensor(data)
    sample_inputs[:,-2:] = latent_sample_space
    non_disturbed = sample_inputs[:,:]
    
    sample_outputs = g(sample_inputs)
    
    #sample_outputs = torch.rand((1000,2))
    non_disturbed[:,-2:] = sample_outputs
    
    output_probabilities = d(non_disturbed)
    output_probabilities = output_probabilities.detach().numpy()
    print(max(output_probabilities)[0])
    index = output_probabilities.argmax()
    prediction = non_disturbed.detach().numpy()[index][-2:]
    
    return prediction, max(output_probabilities)[0]
    
def preprocess(data, min_, range_):
    data = np.divide((data - min_),range_)
    return data

def post_process(data, min_, range_):
    sub_min = min_[-2:]
    sub_range = range_[-2:]
    data = np.multiply(data, sub_range) + sub_min
    return data



discriminator = Discriminator()
generator = Generator()

discriminator.load_state_dict(torch.load(PATH_TO_D_WEIGHTS))
generator.load_state_dict(torch.load(PATH_TO_G_WEIGHTS))

df = pd.read_csv('data_imp.csv')
df = df[(df['nac_weight'].isna() == False) & (df['blade_weight'].isna() == False)]
df = df.drop(['Operator','Region_UK','Region_Asia','Region_US','Region_Europe','Region_Middle East'], axis=1)
max_values = df.max().values
min_values = df.min().values
range_values = max_values - min_values
# print(df.head())

# evaluating
abs_mean = np.zeros((1,2))
guesses  = 0
for i in range(len(df)):
    data = preprocess(df.iloc[i].values, min_values, range_values)
    predicted, confidence = inference(data, discriminator, generator)

    predicted = post_process(predicted, min_values, range_values)
    print(predicted)
    
    if confidence > 0.7:
        actual = data[-2:]
        difference = actual - predicted
        abs_diff = abs(difference)
        abs_mean += abs_diff
        guesses += 1
    else:
        pass

print(abs_mean/guesses)
print(guesses)
    