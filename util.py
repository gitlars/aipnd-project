"""Utilities module for image classifier project"""

import datetime
from collections import OrderedDict
from torch import nn

def datestr():
    "Returns date and time string in a particular format"
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

def build_sequential(checkpoint):
    "Builds and returns nn.Sequential() classifier from checkpoint"
    
    ## Create classifier for our flower classification task
    # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
    args = checkpoint['args']
    args_dict = vars(args)
    #print(args_dict)
    
    n_input = (checkpoint['args']).n_input
    n_hidden = (checkpoint['args']).hidden_units
    n_output = 102
    n_sequence = n_hidden.copy()
    # https://stackoverflow.com/questions/8537916/whats-the-idiomatic-syntax-for-prepending-to-a-short-python-list
    n_sequence.insert(0, checkpoint['args'].n_input)
    n_sequence.append(n_output)

    ## Build nn.Sequential()
    # https://pymotw.com/3/collections/ordereddict.html
    new_classifier = OrderedDict()

    for i in range(len(n_sequence) - 1):
        ordstr = str(i + 1)
        new_classifier['fc' + ordstr] = nn.Linear(n_sequence[i], n_sequence[i + 1])    
        # Don't do ReLU or dropout on output layer
        if i < len(n_sequence) - 2:
            new_classifier['relu' + ordstr] = nn.ReLU()
            new_classifier['drop' + ordstr] = nn.Dropout(args.prob_dropout)

    new_classifier['output'] = nn.LogSoftmax(dim=1)

    return nn.Sequential(new_classifier)