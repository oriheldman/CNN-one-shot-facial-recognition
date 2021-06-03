# Siamese Neural Networks for One-shot Image Recognition

## Overview'
A convolutional neural networks model for facial recognition based on the paper "Siamese Neural Networks for One-shot Image Recognition". The models preforms a one-shot learning task for previously unseen objects. Given two facial images of previously unseen persons, the model determines whether they are the same person.

## Data
The "Labeled Faces in the Wild-a" (LFW-a) image collection is a database of labeled, face
images intended for studying Face Recognition in unconstrained images. We augmented the training set with an affine distortion.

### Train images
images from 4038 people paired for creating 1100 positive pairs (images from
same person) and 1100 negative pairs (images from different persons)
### Test images
images from 1711 people paired for creating 500 positive pairs (images from
same person) and 500 negative pairs (images from different persons)

## CNN Architecture
Overall, the model consists of 4 convolutional layers and 2 fully connected layers.

Inspired by the paper “Siamese Neural Networks for One-shot Image Recognition”, the
model is a siamese convolutional neural network. The model takes as an input two input
images (x1 and x2) which are passed through the ConvNet to generate a fixed length feature
vector for each image (h(x1) and h(x2)). Later the L1 distance is computed between the two
feature vectors (h(x1) and h(x2)) to compute the similarity of the two images. 

## Model Results
| Set / Measure        | Loss           | Accuracy  | AUC
|:-------------:|:-------------:|:-------------:|:-------------:|
| Train      | 1169 | 1    | 1    |
| Validation | 1170 | 0.69 | 0.76 |
| Test       | 1170 | 0.65 | 0.71 |
