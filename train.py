#! /usr/bin/env python3
# coding=utf-8

"""
Trains a classifier on a dataset of images using a pretrained
network and saves the model to a checkpoint.

SPECS:
- The training loss, validation loss, and validation accuracy are printed out as
 a network trains.
- Allows users to choose from **four** different architectures available from
 torchvision.models.
- Allows users to set hyperparameters for learning rate, number of hidden units,
 and training epochs.
- Allows users to choose training the model on a GPU.

TODO:
    - args validation,
    - complete docstrings,
    - allow to resume training on loaded model
    - catch cuda exceptions
    - write unit tests

"""

import os
import argparse
from collections import OrderedDict
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

MODEL_CHOICES = ['vgg16', 'vgg19', 'densenet121', 'densenet161']


def main():
    ''''''
    args = get_input_args()

    # Load and transform images
    dataloaders = load_transform(args)

    # Create model
    model = models.__dict__[args.arch](pretrained=True)
    model = build_classifier(model, args)

    # Define loss function (criterion) and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=args.lr)

    # Print model and training parameters
    print_params(model, args)

    # Train the model , test it and save checkpoint
    model = train(dataloaders, model, criterion, optimizer, args)
    model = test(dataloaders, model, criterion, args)
    checkpoint(model, dataloaders, optimizer, args)


def get_input_args():
    ''''''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', dest='arch',
                        default='densenet121',
                        choices=MODEL_CHOICES,
                        help='choose model architecture: ' +
                        ' | '.join(MODEL_CHOICES) +
                        ' (default: densenet121)')
    parser.add_argument('--gpu', dest='gpu', default=False,
                        action='store_true', help='train model on gpu')
    parser.add_argument('-e', '--epochs', dest='epochs', default=2,
                        type=int,
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('-b', '--batch-size', dest='batch',
                        default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.001,
                        type=float, help='learning rate (default: 0.001)')
    parser.add_argument('-hl', '--hidden-layers', dest='hidden_layers',
                        default=None,
                        type=str,
                        help='define custom\
                                hidden layers (use comma separated values)' +
                        """default layers:  {
                                    'densenet121': '500',
                                    'densenet161': '1000, 500',
                                    'vgg16': '4096, 4096, 1000',
                                    'vgg19': '4096, 4096, 1000'
                                    }""")

    return parser.parse_args()


def load_transform(args):
    ''''''
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')

    # Define image normalization parameters
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    # Define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test']),
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                             batch_size=args.batch,
                                             shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                             batch_size=args.batch,
                                             shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'],
                                            batch_size=args.batch,
                                            shuffle=True),
    }

    batch_shape = next(iter(dataloaders['train']))[0].size()
    print()
    print('=> Succesfully transformed and loaded images.')
    print('   Batch size after transformations: ', batch_shape)
    print('   No. of training images          : ', len(image_datasets['train']))
    print('   No. of validation images        : ', len(image_datasets['valid']))
    print('   No. of testing images           : ', len(image_datasets['test']))
    print('   No. of classes                  : ',
          len(image_datasets['train'].classes), '\n')

    return dataloaders


def build_classifier(model, args):
    '''Builds a classifier based on the input model and the hidden
    layers argument passed in the command line'''

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    train_dir = os.path.join(args.data_dir, 'train')

    in_features = {
        'densenet121': 1024,
        'densenet161': 2208,
        'vgg16': 25088,
        'vgg19': 25088,
    }

    default_layers = {
        'densenet121': '500',
        'densenet161': '1000, 500',
        'vgg16': '4096, 4096, 1000',
        'vgg19': '4096, 4096, 1000',
    }

    out_features = len([i for i in os.listdir(train_dir)])

    output = nn.LogSoftmax(dim=1)
    relu = nn.ReLU()
    dropout = nn.Dropout()

    if args.hidden_layers:
        # Get args hidden layers
        hidden_layers = args.hidden_layers.split(',')
    else:
        hidden_layers = default_layers[args.arch].split(',')

    no_hidden_layers = len(hidden_layers)

    # Define the first and the last layer
    first = [nn.Linear(in_features[args.arch], int(hidden_layers[0]))]
    first.append(relu)
    if args.arch[:3] == 'vgg':
        first.append(dropout)

    last = nn.Linear(int(hidden_layers[-1]), out_features)

    # Compose the middle ones
    middle = []
    if len(hidden_layers) > 1:
        for i in range(len(hidden_layers) - 1):
            middle.append(
                nn.Linear(int(hidden_layers[i]), int(hidden_layers[i + 1])))
            middle.append(relu)
            if args.arch[:3] == 'vgg':
                middle.append(dropout)

    model.classifier = nn.Sequential(*first, *middle, last, output)

    return model


def print_params(model, args):
    ''''''
    print()
    print("=> Using pre-trained model: '{}'".format(args.arch))
    print('   Classifier             : ', model.classifier)
    print('   Epochs                 : ', args.epochs)
    print('   Learning Rate          : ', args.lr, '\n')


def train(dataloaders, model, criterion, optimizer, args):
    ''''''
    print("=> Training Classifier..\n")

    # Configure use of gpu
    if args.gpu:
        model = model.cuda()
        print('   Using GPU..')

    start_time = datetime.now()
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 15
    for e in range(epochs):
        for images, labels in iter(dataloaders['train']):

            # Configure use of gpu
            if args.gpu:
                criterion = criterion.cuda()
                images = images.cuda()
                labels = labels.cuda()

            steps += 1

            # Wrap images and labels in Variables so we can calculate gradients
            inputs = Variable(images)
            targets = Variable(labels)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                val_loss = 0
                for ii, (images, labels) in enumerate(dataloaders['valid']):

                    # Configure use of gpu
                    if args.gpu:
                        images = images.cuda()
                        labels = labels.cuda()

                    # Set volatile to True so we don't save the history
                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)

                    output = model.forward(inputs)
                    val_loss += criterion(output, labels).item()

                    # Calculating the accuracy
                    # Model's output is log-softmax,
                    # take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class,
                    # compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct prsedictions
                    # divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(
                          val_loss / len(dataloaders['valid'])),
                      "Validation Accuracy: {:.3f}".format(
                    accuracy / len(dataloaders['valid'])))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()

    print("=> Classifier trained!")
    time_elapsed = datetime.now() - start_time
    print('   Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

    return model


def test(dataloaders, model, criterion, args):
    ''''''
    print("=> Calculating Test Accuracy..\n")

    model.eval()
    accuracy = 0
    test_loss = 0
    for ii, (images, labels) in enumerate(dataloaders['test']):

        # Configure use of GPU
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    model.test_accuracy = accuracy.numpy() / len(dataloaders['test'])

    print("\nBatch: {} ".format(ii + 1),
          "Test Loss: {:.3f}.. ".format(test_loss / len(dataloaders['test'])),
          "Test Accuracy: {:.3f}\n".format(model.test_accuracy))

    return model


def checkpoint(model, dataloaders, optimizer, args):
    ''''''
    # Switch to CPU for loading compatability
    model = model.cpu()

    # Map classes to model indices
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx

    checkpoint = {
        'classifier': model.classifier,
        'architecture': args.arch,
        'epochs': args.epochs,
        'lr': args.lr,
        'train_batch_size': dataloaders['train'].batch_size,
        'val_batch_size': dataloaders['valid'].batch_size,
        'accuracy': model.test_accuracy,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    _datetime = datetime.now().strftime("%Y%m%d")
    checkpoint_path = \
        os.path.relpath('{}_{}_{}acc_chkpt.pth'.format(_datetime,
                                                       args.arch,
                                                       np.round(
                                                           model.test_accuracy,
                                                           2)))
    torch.save(checkpoint, checkpoint_path)
    print("=> Saved checkpoint at: ", checkpoint_path)


if __name__ == '__main__':
    main()
