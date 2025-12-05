# Image Classifier Project for Flower Species Using PyTorch

In this project, we'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  dataset of 102 flower categories.

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

This project uses PyTorch and the torchvision package; the Jupyter Notebook walks through the implementation of the image classifier and shows an example of the classifier's prediction on a test image. The classifier was also converted into a Python application, which could be run from the command line using "train.py" and "predict.py".


## Installation
You only need to install Conda and run the following commands:

```
conda env create -f im-classification.yml
source activate im-classification
```