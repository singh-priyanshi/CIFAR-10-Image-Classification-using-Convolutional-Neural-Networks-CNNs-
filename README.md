# CIFAR-10 Image Classification using Convolutional Neural Networks (CNNs)

## Overview

This project focuses on building an image classification model to classify the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). CIFAR-10 is a well-known dataset consisting of 60,000 32x32 color images across 10 classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. This repository contains all the necessary code, model training processes, and analysis required to classify the CIFAR-10 images efficiently.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Machine Learning Techniques](#machine-learning-techniques)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Cifar-10-Image-Classification-using-CNNs.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Cifar-10-Image-Classification-using-CNNs
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset can be directly downloaded using the `torchvision` library:

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

## Model Architecture

The model used for this project is a Convolutional Neural Network (CNN). CNNs are well-suited for image classification due to their ability to capture spatial hierarchies in images. The architecture consists of the following layers:

1. **Convolutional Layers**: Extract feature maps from the images. We used multiple convolutional layers with ReLU activation to learn complex image features.
2. **Pooling Layers**: Down-sample the feature maps to reduce dimensionality and computational complexity.
3. **Fully Connected Layers**: Flatten the feature maps and classify them into one of the 10 classes.
4. **Dropout Layers**: Used to prevent overfitting by randomly setting some neurons to zero during training.

A sample architecture used in this project:

- Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPooling2D
- Conv2D -> ReLU -> Conv2D -> ReLU -> MaxPooling2D
- Flatten -> Fully Connected -> Dropout -> Fully Connected -> Output

## Machine Learning Techniques

### 1. **Data Augmentation**
   To improve the generalizability of the model, data augmentation techniques such as random cropping, horizontal flipping, and random rotation were used to artificially increase the diversity of the dataset.

### 2. **Normalization**
   The images were normalized to have zero mean and unit variance, which helps in speeding up the convergence of the model during training.

### 3. **Regularization**
   Regularization techniques, such as **Dropout**, were used to prevent the model from overfitting. Dropout randomly disables certain neurons during training, which helps in making the model more robust.

### 4. **Learning Rate Scheduling**
   To optimize the training process, a learning rate scheduler was used. It reduces the learning rate when the model's performance plateaus, allowing for more precise convergence.

### 5. **Optimizer**
   The **Adam** optimizer was used due to its adaptive learning rate properties, making it ideal for fast and efficient training.

## Training and Evaluation

The model was trained for 50 epochs using the **Cross-Entropy Loss** function, which is commonly used for multi-class classification problems. The model was evaluated on the test dataset to compute its accuracy.

```python
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    # Training code here
    pass
```

The evaluation metrics include **accuracy**, **precision**, **recall**, and **F1 score**.

## Results

The model achieved an accuracy of approximately **88%** on the CIFAR-10 test dataset. The use of data augmentation and regularization techniques helped improve the model's performance and prevented overfitting.

## Usage

To train the model, run the following command:

```bash
python src/train.py
```

To evaluate the model on the test set:

```bash
python src/evaluate.py
```

## Contributing

Contributions are welcome! Please create a pull request if you would like to contribute to this project.
