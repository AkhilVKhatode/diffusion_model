# Simple Diffusion Model Project

This repository contains the implementation of a simple diffusion model, using TensorFlow, for image generation. The model is trained on the CIFAR-10 dataset to demonstrate the process of generating images through forward and reverse diffusion steps.

## Project Overview

The code provided in this repository implements a basic diffusion model for generating images. The model is designed to start with a noisy image and gradually reduce the noise through a series of steps, resulting in a clearer image. The diffusion process is guided by a neural network trained on the CIFAR-10 dataset.

## Implementation Details

### Dataset

- **Dataset Used**: CIFAR-10
- **Training Data**: Only images belonging to the "automobile" class (label 1) from CIFAR-10 are used for training.

### Model Architecture

The model architecture consists of the following key components:

- **Noise Schedule**: A noise schedule that gradually reduces the noise added to the images over time.
- **Forward Diffusion Process**: A process that adds noise to the images at each timestep.
- **Neural Network Model**: A convolutional neural network that learns to predict the denoised image from a noisy image at each timestep.
- **Upsampling and Downsampling Blocks**: Blocks to process the image at different scales within the neural network.

### Training

The model is trained using a Mean Absolute Error (MAE) loss function and the Adam optimizer. The learning rate is gradually reduced during training to ensure convergence.

### Key Functions

- `forward_noise`: Adds noise to the image at a specific timestep.
- `make_model`: Constructs the neural network model.
- `train_one`: Performs a single training step on a batch of images.
- `train`: Trains the model over multiple iterations.
- `predict` and `predict_step`: Functions to generate images and visualize the intermediate steps during the diffusion process.

### Model Saving and Loading

The trained model weights are saved to Google Drive, allowing for easy resumption of training. Functions are provided to load the latest saved weights and continue training from where it was left off.

## Usage

To run the project:

1. **Environment Setup**: Ensure you have the required dependencies installed, particularly TensorFlow.
2. **Google Colab**: The code is designed to run on Google Colab. The `weights_dir` points to a directory on Google Drive for saving model weights.
3. **Training**: Execute the training loop, which will continuously train the model, reduce the learning rate, and periodically save the model weights.
4. **Prediction**: After every few epochs, the model will generate and display images to visualize the diffusion process.

## Resources

- [Diffusion Model from Scratch](https://tree.rocks/make-diffusion-model-from-scratch-easy-way-to-implement-quick-diffusion-model-e60d18fd0f2e): This resource provides a detailed explanation of diffusion models and how they work.

## Acknowledgments

This project is inspired by the concept of diffusion models in generative modeling and uses the CIFAR-10 dataset to demonstrate the effectiveness of this approach in image generation.
