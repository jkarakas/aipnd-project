#! /usr/bin/env python3
# coding=utf-8

'''
Loads a saved pytorch model checkpoint and an image and prints the most likely
image class and it's associated probability. If provided, uses a category to
name json file to map categories to names and print the names as well.

SPECS:
- Allows users to print out the top K classes along with associated
 probabilities.
- Allows users to use the GPU to calculate the predictions.
- Allows users to load a JSON file that maps the class values to other category
 names.

 TODO:
    - args validation,
    - complete docstrings,
    - write unit tests
'''

import os
import argparse
import json
from PIL import Image

import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models


def main():
    ''''''
    args = get_input_args()

    # Load model from checkpoint
    model = load_checkpoint(args)

    # Predict and print top K classes along with their probabilities
    predict(model, args)


def get_input_args():
    ''''''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('checkpoint_path', metavar='CHKPT_PATH',
                        help='path to chekpoint')
    parser.add_argument('image_path', metavar='IMG_PATH',
                        help='path to image')
    parser.add_argument('--gpu', dest='gpu', default=False,
                        action='store_true', help='use gpu for the prediction')
    parser.add_argument('-k', '--topk', dest='topk', default=1,
                        type=int,
                        help='number of top K classes to print (default: 1)')
    parser.add_argument('-ctn', '--cat_to_name', dest='cat_to_name',
                        default=None,
                        type=str,
                        help="""
                        The path to an alternative JSON file that maps the class
                         values to category names (default:None)
                         """)
    return parser.parse_args()


def load_checkpoint(args):
    ''''''
    checkpoint_path = os.path.relpath(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model = models.__dict__[checkpoint['architecture']](pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    image_size = image.size

    # Resize the image where the shortest side is 256 pixels,
    # keeping the aspect ratio
    shorter_side_idx = image_size.index(min(image_size))
    bigger_side_idx = image_size.index(max(image_size))
    aspect_ratio = image_size[bigger_side_idx] / image_size[shorter_side_idx]

    new_size = [None, None]
    new_size[shorter_side_idx] = 256
    new_size[bigger_side_idx] = int(256 * aspect_ratio)

    image = image.resize(new_size)

    # Crop out the center 224x224 portion of the image
    width, height = new_size
    new_width, new_height = (224, 224)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    image = image.crop((left, top, right, bottom))

    # Convert image color channels from 0-255 to floats 0-1.
    np_image = np.array(image)
    np_image = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension but it's the
    # third dimension in the PIL image and Numpy array. Traspose the numpy array
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(model, args):
    ''' Predict the class (or classes) of an image using a trained deep learning
     model. If available, uses a category to name json file to map categories to
      names and print the names as well'''
    print("=> Predicting probabilities..\n")
    model.eval()

    # Create class to name dictionary
    idx_to_class = {i: k for k, i in model.class_to_idx.items()}

    # Load and process image
    image_path = os.path.relpath(args.image_path)
    image = process_image(Image.open(image_path))
    image = torch.FloatTensor([image])

    # Configure use of gpu
    if args.gpu:
        print('   Using GPU..\n')
        model = model.cuda()
        image = image.cuda()

    # map model indexes to image classes
    idx_to_class = {i: k for k, i in model.class_to_idx.items()}

    # get top K predictions and indexes
    output = model.forward(Variable(image))
    ps = torch.exp(output).data[0]
    cl_index = ps.topk(args.topk)

    # Map to classes and names
    classes = [idx_to_class[idx]
               for idx in cl_index[1].cpu().numpy()]
    probs = cl_index[0].cpu().numpy()

    print('   Probabilities: ', probs)

    if args.cat_to_name:
        ctn_path = os.path.relpath(args.cat_to_name)
        with open(ctn_path, 'r') as f:
            cat_to_name = json.load(f)
            names = [cat_to_name[cl] for cl in classes]
            print('    Classes:       ', [(cl, nm) for cl, nm in
                                          zip(classes, names)])
    else:
        print('   Classes:       ', classes)


if __name__ == '__main__':
    main()
