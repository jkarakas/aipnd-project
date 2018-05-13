

# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Training

Trains a classifier on a dataset of images using a pretrained network and saves the model to a checkpoint.

SPECS:
- The training loss, validation loss, and validation accuracy are printed out as
 a network trains.
- Allows users to choose from **four** different architectures available from
 `torchvision.models`.
- Allows users to set hyperparameters for learning rate, number of hidden units,
 and training epochs.
- Allows users to choose training the model on a GPU.

### Usage

To train a model, run `train.py` with the desired model architecture and the path to the image dataset. For example:

`python train.py --arch densenet161 --gpu -e 5 -lr 0.01  -b 128  -hl '800,400,200' ./flowers`



```
usage: train.py [-h] [--arch {vgg16,vgg19,densenet121,densenet161}] [--gpu]
                [-e EPOCHS] [-b BATCH] [-lr LR] [-hl HIDDEN_LAYERS]
                DIR


positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch {vgg16,vgg19,densenet121,densenet161}
                        choose model architecture: vgg16 | vgg19 | densenet121
                        | densenet161 (default: densenet121)
  --gpu                 train model on gpu
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs to run (default: 2)
  -b BATCH, --batch-size BATCH
                        mini-batch size (default: 64)
  -lr LR, --learning-rate LR
                        learning rate (default: 0.001)
  -hl HIDDEN_LAYERS, --hidden-layers HIDDEN_LAYERS
                        define custom hidden-layers(use comma separated
                        values) 
  ```

## Predicting

Loads a saved pytorch model checkpoint and an image and prints the most likely
image class and it's associated probability. If provided, uses a category to
name json file to map categories to names and print the names as well.

SPECS:
- Allows users to print out the top K classes along with associated
 probabilities.
- Allows users to use the GPU to calculate the predictions.
- Allows users to load a JSON file that maps the class values to other category
 names.


### Usage

To calculate a prediction, run `predict.py` with the paths to the checkpoint, the image and category to name json file if available. For example:

`python predict.py --gpu -k 3 -ctn cat_to_name.json 20180513_densenet121_0.93acc_chkpt.pth ./flowers/test/85/image_04805.jpg `



```
usage: predict.py [-h] [--gpu] [-k TOPK] [-ctn CAT_TO_NAME]
                  CHKPT_PATH IMG_PATH


positional arguments:
  CHKPT_PATH            path to chekpoint
  IMG_PATH              path to image

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 use gpu for the prediction
  -k TOPK, --topk TOPK  number of top K classes to print (default: 1)
  -ctn CAT_TO_NAME, --cat_to_name CAT_TO_NAME
                        The path to an alternative JSON file that maps the
                        class values to category names (default:None)
  ```
