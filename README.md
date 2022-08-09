# CS523 FinalProject

Boston University 2022 Summer2 CS523 Deep Learning Final Project Team 2

Team Members: Kaiyang Zhao (kyzhao@bu.edu) / Bingquan Cai (bqcai@bu.edu) / Bin Xu (xu842251@bu.edu)

# Facial Expression Recognition(FER) with Feature Visualization & Explainability

## Motivation

Being able to recognize facial expressions is key to nonverbal communication between humans. Facial expressions and other gestures convey nonverbal communication cues that play an important role in interpersonal relations. We can use algorithms to instantaneously detect faces, code facial expressions, and recognize emotional states.

## Goal

- Look for FER related papers to reproduce and fine-tune the model and get results.
- Perform feature visualization of different expressions through models.
- Find explainability of predictions made by models.

## Dataset

**FER2013 dataset**
- Consists of 35,887 faces which are 48x48 pixel grayscale images
- Training 28,709 / Validation 3,589 / Testing 3,589
- 7 catagories of facial expressions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

## Methodology

### State-of-Art
- Networks: CNN
  - Batch normalization
  - Fully connected (fc) layer
  - Drop-out layer 
  - Max-pooling & stochastic pooling
- Dataset Preprocessing: illumination Correction
  - Normalizing the images
- CNN architecture: 
  - VGG
  - Inception
  - ResNet 
- CNN training and inference: 
  - 300 epochs
  - Learning rate = 0.1
  - Batch size = 128
  - Data augmentation

### Our model
- Transfer learning
- Additional layers
- Categorial Crossentropy
- Adam optimizer
- 300 Epochs

## Tools
- Tensorflow
  - Tensorflow.keras
- Pandas
- Numpy
- Google. Colab
- Pip 
- Zipfile
- Matplotlib
- Seaborn
- skimage
