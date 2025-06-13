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

parser.add_argument('image_path',
                    type=str,
                    help='Directory where image is saved')
parser.add_argument('checkpoint', action='store',
                    type=str,
                    help='Directory where checkpoint was saved')

parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    type=int,
                    required=False,
                    default=1,
                    help='Return the K most likely classes')

parser.add_argument('--category_names', action='store',
                    type=str,
                    dest='category_names',
                    required=False,
                    help='Use a mapping of categories to real names')


parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

device = torch.device("cuda:0" if results.gpu else "cpu")
top_k = results.top_k

print('Loading checkpoint \n------------------')
model = utility.load_checkpoint(results.checkpoint)
print('Done \n')
print('Processing Image \n------------------')
image_to_process = Image.open(results.image_path)
print('Done \n')

processed_image = utility.process_image(image_to_process)

print('Running prediction \n------------------')
probs, classes = model_helper.predict(processed_image,model,device,top_k)
print('Done \n')

print('Predicted categories: \n---------------------')
if results.category_names:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
        names_of_categories = [cat_to_name[i] for i in classes]
        
        for cat, prob in zip(names_of_categories,probs):
            print(cat + " : {:2.2f}".format(prob*100))
        
else:
        for classes, prob in zip(names_of_categories,probs):
            print(cat + " : {:2.2f}".format(prob*100))
