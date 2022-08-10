# CS523 FinalProject

Boston University 2022 Summer2 CS523 Deep Learning Final Project Team 2

Team Members: Kaiyang Zhao (kyzhao@bu.edu) / Bingquan Cai (bqcai@bu.edu) / Bin Xu (xu842251@bu.edu)

# Facial Expression Recognition(FER) with Feature Visualization & Explainability

## Guide of running code

After installing libs, running Four .ipynb in our project. There are specific instruction in every file.

FER2013_explainability.ipynb

feature_visulization.ipynb

Copy_of_VGG16_FACIAL_training.ipynb

training.ipynb

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

## Model Training

### Origin model

- Training Results of VGG16:
  - Early stoppage at epoch 203
  - Loss: 1.2933
  - Learning rate: 0.0005
  - Validation Accuracy: 0.8852

### Other models

- 5-layer-CNN: Accuracy: 0.6339
- CPCPCPFFF: Accuracy: 0.5598
- CCPCCPCCPFF: Accuracy: 0.5929
- Ensemble Model: Accuracy: 0.6863

## Experiment

### Setup

- 1st experiment: CNN training and inference - State-of-Art: 
  - 600 epochs
  - learning rate = 0.1
  - batch size = 128

- 2nd experiment: Networks: Classification & dataset preprocessing
  - Illumination Correction


### Result
- 1st Experiment
  - Early stop at epoch 25 
  - Model overfitting

- 2nd Experiment
  - Early stopping at Epoch 164
  - Loss: 1.1461
  - Accuracy: 0.8912
  - No Overfitting nor Underfitting


## Feature visualization

### Visualizing Filters
Visualize the learned filters, used by CNN to convolve the feature maps.

### Visualizing feature maps
Called Activation Map, is obtained with the convolution operation, applied to the input data using the filter/kernel.


## Explainability

### Correct predictions

### Wrong predictions

## Conclusion

- Reproduce and fine-tune the model for FER2013 dataset
- Perform feature visualization on the model
- Show the explainability of the model
- Future steps:
  - Further improve accuracy 
  - Study the bias that affect datasets related to FER problem
  - Evaluate the model on additional datasets (real-world questions)


## References

- Goodfellow I J, Erhan D, Carrier P L, et al. Challenges in representation learning: A report on three machine learning contests[C]//International conference on neural information processing. Springer, Berlin, Heidelberg, 2013: 117-124.

- Pramerdorfer C, Kampel M. Facial expression recognition using convolutional neural networks: state of the art[J]. arXiv preprint arXiv:1612.02903, 2016.

- Debnath, T., Reza, M., Rahman, A., Beheshti, A., Band, S. S., & Alinejad-Rokny, H. (2022). Four-layer ConvNet to facial emotion recognition with minimal epochs and the significance of data diversity. Scientific Reports, 12(1), 1-18.

- Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE international conference on computer vision. 2017: 618-626.
