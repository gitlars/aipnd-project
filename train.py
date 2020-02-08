# Train a new network on a data set with train.py
#
#    Basic usage: python train.py data_directory
#    Prints out training loss, validation loss, and validation accuracy as the network trains
#    Options:
#        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#        Choose architecture: python train.py data_dir --arch "vgg13"
#        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#        Use GPU for training: python train.py data_dir --gpu
# 

# Imports here
import argparse
import os
import numpy as np
import torch
import random
from util import datestr
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Parse input arguments
parser = argparse.ArgumentParser(
    description='Train a new network on a data set of images.'
)

parser.add_argument('data_directory', type=str, help='path to directory of image training data')
parser.add_argument('--seed', type=int, default=16, help='manual random seed (default=%(default)s)')
parser.add_argument('--gpu', action='store_const', const='use_gpu', default=False, help='use gpu (default=%(default)s)')
parser.add_argument('--arch', type=str, choices=['vgg13', 'vgg16', 'densenet161'], default='vgg16', help='deep neural network architecture for image features (default=%(default)s)') # , 'resnet50', 'inception_v3'
parser.add_argument('--learning_rate', type=float, default='0.0001', help='learning rate (default=%(default)s)')
parser.add_argument('--hidden_units', type=int, nargs='?', default=[], help='hidden layer sizes (default=%(default)s)')
parser.add_argument('--prob_dropout', type=float, default=0.25, help='dropout probability (fraction) (default=%(default)s)')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs (default=%(default)s)')
parser.add_argument('--warm_start', action='store_const', const=True, default=False, help='warm start with checkpoint.pth')
parser.add_argument('--print_every', type=int, default=5, help='decimation factor for printing per-batch performance (default=%(default)s)')
parser.add_argument('--save_dir', type=str, nargs='?', default='.', help='directory for saving checkpoints (default=\'%(default)s\')')
args = parser.parse_args()
print(args)

# Note: I evolved my idea of what the classifier structure should be over time, failing to reach the 
# 70% threshold (possibly partially owing to other bugs) with hidden_units of [4096, 1024, 1024] and
# [1024, 512, 256, 256, 256, 256]. While i did not copy any code, I do attribute the /idea/ of 
# running with hidden_units = [] to https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/,
# found via a Google search for "flowers vgg 102 hidden classifier".

# Set randomizer behavior
manual_seed = args.seed
attempt_determinism = True # Ran into some problems that /seemed/ influenced by this
# https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/4
# https://pytorch.org/docs/stable/notes/randomness.html
if attempt_determinism:
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # if you are using GPU
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    #torch.backends.cudnn.enabled = False 
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

# Load the data
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## DONE: Define your transforms for the training, validation, and testing sets
## https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
data_transforms = {
    'train': transforms.Compose([transforms.Resize(255),
                                 transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])
                                ]),
    'valid': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])
                                ]),
    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
}

## DONE: Load the datasets with ImageFolder
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'valid', 'test']
}

## DONE: Using the image datasets and the transforms, define the dataloaders
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in ['train', 'valid', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
n_batches = {x: len(dataloaders[x]) for x in ['train', 'valid', 'test']}

## Use GPU if it's available and specified
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")
elif args.gpu:
    print("Warning: --gpu specified but cuda is not available.")
    device = torch.device("cpu")
else:
    device = torch.device("cpu")
    
print("Training with {}...".format(device))

# DONE: Build and train your network

## https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    n_input = model.classifier[0].in_features
elif args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    n_input = model.classifier[0].in_features
elif args.arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    n_input = model.classifier.in_features
#elif args.arch == 'inception_v3':
    # RuntimeError: Expected tensor for argument #1 'input' to have the same dimension as tensor for 'result'; but 4 does not equal 2 (while checking arguments for cudnn_convolution)
#    model = models.inception_v3(pretrained=True)
#    n_input = model.fc.in_features
#elif args.arch == 'resnet50':
     # https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/2
#    model = models.resnet50(pretrained=True)
#    n_input = model.fc.in_features
else:
    # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
    raise ValueError('Encountered unexpected --arch argument')

## Remember the number of input features (size of input to the classifier)
    
## Freeze parameters so that we don't backpropagate through them
model.to(device)
for param in model.parameters():
    param.requires_grad = False

## Create classifier for our flower classification task
n_hidden = args.hidden_units
#print(args.hidden_units)
#print(n_hidden)
#print(type(n_hidden))
#print(len(n_hidden))
n_output = 102
n_sequence = n_hidden.copy()
# https://stackoverflow.com/questions/8537916/whats-the-idiomatic-syntax-for-prepending-to-a-short-python-list
n_sequence.insert(0, n_input)
n_sequence.append(n_output)

## Build nn.Sequential()
# https://pymotw.com/3/collections/ordereddict.html
new_classifier = OrderedDict()

print("n_hidden = {}".format(n_hidden))
print("n_sequence = {}".format(n_sequence))
for i in range(len(n_sequence) - 1):
    ordstr = str(i + 1)
    new_classifier['fc' + ordstr] = nn.Linear(n_sequence[i], n_sequence[i + 1])    
    # Don't do ReLU or dropout on output layer
    if i < len(n_sequence) - 2:
        new_classifier['relu' + ordstr] = nn.ReLU()
        new_classifier['drop' + ordstr] = nn.Dropout(args.prob_dropout)

new_classifier['output'] = nn.LogSoftmax(dim=1)

## Replace classifier in deep neural network with my own
model.classifier = nn.Sequential(new_classifier)
print(model)

## Warm start with existing checkpoint
if args.warm_start:
    filepath = os.path.join(args.save_dir, 'checkpoint.pth')
    checkpoint = torch.load(filepath)
    model.classifier.load_state_dict(checkpoint['model_classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print('Loaded checkpoint for warm start with args: {}'.format(checkpoint['args']))
else:
    checkpoint = dict()
    checkpoint['cum_epochs'] = 0
    
# Initialize dictionary keys if null (appended to at end for checkpoint)
for key in ['epoch_loss_train', 'epoch_loss_valid', 'epoch_acc_train', 'epoch_acc_valid']:
    if checkpoint.get(key) is None:
        checkpoint[key] = []

print('Validation accuracy history after {} cumulative epochs is {}.'.format(
    checkpoint['cum_epochs'], checkpoint['epoch_acc_valid']))

## Initialize the weights
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        # https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

manually_init_weights = False       
if manually_init_weights:
    model.classifier.apply(init_weights)

## Use negative log likelihood loss (corresponding to log-softmax)
criterion = nn.NLLLoss()

## Only train the classifier parameters (feature parameters are frozen)
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train the network
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
num_epochs = args.epochs
print_every = args.print_every
epoch_losses = {x: [] for x in ['train', 'test', 'valid']}
epoch_accuracies = {x: [] for x in ['train', 'test', 'valid']}

model.to(device) # Use GPU if available

for epoch in range(num_epochs):
    
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train() # Training mode
            print(datestr() + "(epoch {}/{}) Model is set to training mode for '{}' phase.".format(
                epoch + 1, num_epochs, phase))
        else:
            model.eval() # Evaluation mode
            print(datestr() + "(epoch {}/{}) Model is set to evaluation mode for '{}' phase.".format(
                epoch + 1, num_epochs, phase))
        
        batch_counter = 0
        running_loss = 0.0
        running_n_correct = 0
        
        for inputs, labels in dataloaders[phase]:
            batch_counter += 1
            
            # Move inputs and label tensors to the target device (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                # Forward pass
                outputs = model.forward(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)

                # Backward pass and step optimizer only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Calculate and record statistics
                batch_loss = loss.item()
                #print(outputs.shape)
                #print(preds)
                #print(labels.data)
                batch_n_correct = torch.sum(preds == labels.data)
                batch_acc = float(batch_n_correct) / len(labels)
                running_loss += batch_loss
                running_n_correct += batch_n_correct

                # Subsample training batch output but print results for every validation batch
                if (((batch_counter - 1) % print_every) == 0) or (phase == 'valid'):
                    print(datestr() 
                        + "    (epoch {}/{} batch {}/{}) {} loss = {:.4f}, acc = {:.2f}% ({}/{})".format(
                            epoch + 1, num_epochs, batch_counter, n_batches[phase], phase, 
                            batch_loss, 100.0*batch_acc, batch_n_correct, len(labels)))
                    
        # Batches complete
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_n_correct.float() / dataset_sizes[phase]
        
        epoch_losses[phase].append(epoch_loss)
        epoch_accuracies[phase].append(epoch_acc)
        
        print(datestr() + "(epoch {}/{}) {} loss = {:.4f}, acc = {:.2f}% ({}/{})".format(
            epoch + 1, num_epochs, phase, epoch_losses[phase][-1], 100.0*epoch_accuracies[phase][-1],
            running_n_correct, dataset_sizes[phase]))

# DONE: Save the checkpoint
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
model.class_to_idx = image_datasets['train'].class_to_idx

# Update checkpoint data (these are set to null above when not warm start)
cum_epochs = checkpoint['cum_epochs'] + num_epochs
checkpoint['epoch_loss_train'].append(epoch_losses['train'])
checkpoint['epoch_loss_valid'].append(epoch_losses['valid'])
checkpoint['epoch_acc_train'].append(epoch_accuracies['train'])
checkpoint['epoch_acc_valid'].append(epoch_accuracies['valid'])

checkpoint['epoch'] = epoch
checkpoint['model_classifier_state_dict'] = model.classifier.state_dict()
checkpoint['optimizer_state_dict'] = optimizer.state_dict()
checkpoint['class_to_idx'] = model.class_to_idx
checkpoint['args'] = args
checkpoint['n_input'] = n_input
checkpoint['cum_epochs'] = cum_epochs

torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
print("Saved checkpoint. For this training session:")

# Print out summary performance
print("    epoch_losses['train'] = {}".format(epoch_losses['train']))
print("    epoch_losses['valid'] = {}".format(epoch_losses['valid']))
print("    epoch_accuracies['train'] = {}".format(epoch_accuracies['train']))
print("    epoch_accuracies['valid'] = {}".format(epoch_accuracies['valid']))
print("    cum_epochs = {}".format(cum_epochs))
print("    args = {}".format(args))
