## This is an implementation of an emotion recognition system using transfer learning.
    The idea behind this project is to give you the availability of choosing the desired emotions to train the model
        on, at least two of these: {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}.
    Thus to be able to create different models triggering different features (tags).

## dataset:
using fer2013 dataset at kaggle [link](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz)

## requirements:
    + python
    + pip install -r requirements
    + download dataset, and place it at "./data/" then tar (unzip) it there.

## how to run:
    1- To create a model:
        python main.py
        and pick the appropriate arguments.
        
    2- To use a model:
        python predict.py
        and pick the model with its timestamp
