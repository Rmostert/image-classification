import argparse
import numpy as np
import json
from PIL import Image
import utility
import model_helper
import torch
from torch import optim
from torch import nn




parser = argparse.ArgumentParser()

parser.add_argument('data_dir',
                    type=str,
                    help='Directory where training data reside')
parser.add_argument('--save_dir', action='store',
                    type=str,
                    default=False,
                    required=False,
                    dest='checkpoint_dest',
                    help='Directory where checkpoint should be saved')

parser.add_argument('--arch', action='store',
                    type=str,
                    dest='architecture',
                    default='densenet121',
                    choices=('vgg16', 'densenet121','resnet50'),
                    help='Model Architecture')

parser.add_argument('--learning_rate', action='store',
                    dest='lr',
                    type=float,
                    default=0.001,
                    help='Model learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    type=int,
                    default=12595,
                    help='Number of hidden units')


parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    type=int,
                    default=6,
                    help='Number of epochs')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

device = torch.device("cuda:0" if results.gpu else "cpu")
data_dir = results.data_dir
checkpoint_dest = results.checkpoint_dest
architecture = results.architecture
lr = results.lr
hidden_units = results.hidden_units
epochs = results.epochs

print('Loading and transforming data \n----------------------------')
trainloader,validloader,testloader, class_to_idx = utility.load_data(data_dir)
print('Done \n')

for i, (img, species) in enumerate(trainloader):
    if i == 0:
        outcomes = species
    else:
        outcomes = torch.cat((outcomes,species),0)

output_size = len(outcomes.unique())

print('Initialising model \n-----------------')
model, criterion, optimizer = model_helper.initialise_model(architecture,hidden_units,output_size,lr)
print('Done \n')

print('Training model \n-------------')
model_helper.train_model(model, trainloader, validloader,optimizer,criterion, epochs, device,print_every=40)
print('Done \n')

if checkpoint_dest:
    
    print('Saving checkpoint \n-------------')

        
    state = {
    'epoch': epochs,
    'architecture': architecture,
    'hidden_units': hidden_units,
    'output_size': output_size,
    'state_dict': model.state_dict(),
    'class_to_idx': class_to_idx
    }

    torch.save(state,checkpoint_dest + '/checkpoint.pth')
    print('Done \n')
    
print('Done!')
