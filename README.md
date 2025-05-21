# Semantic Segmentation on CamVid Dataset

![image](https://github.com/user-attachments/assets/00f61ad0-93d0-4a0c-a53a-9762e5ff9237)

This repository contains the code and experiments for a semantic segmentation task on the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). We implemented and compared several approaches, including:

* A custom **UNet**-inspired architecture from scratch
* A **MobileNetV2**-based model with transfer learning
* **DeepLab** using ResNet and ASPP (Atrous Spatial Pyramid Pooling)

## Dataset
* **Total images**: 701
* **Split**:
  * Train: 401
  * Validation: 150
  * Test: 150
* **Labels**: 32 semantic classes (including a "Void" class for unlabeled pixels)

### Preprocessing Steps
* **Shuffling** of images and labels
* **Resizing** to 224×224:
  * Bicubic interpolation for images
  * Nearest neighbor for labels
* **Label Encoding**: RGB to class index mapping + one-hot encoding

## Models

### 1. UNet (Custom)

* Fully convolutional with encoding and expansive paths
* Skip connections
* Categorical cross-entropy loss
* Optimizer: Adam
* Activation: ReLU
* Performance metric: Mean IoU

**Experiments:**

* Various data augmentations (rotation, cropping, Gaussian noise)
* Hyperparameter tuning:

  * Batch size (4, 8, 16)
  * Regularization (L1, L2, Dropout)
  * Activation functions (ReLU, Sigmoid, Tanh — trainable activations for future work)
  * Optimizers (Adam, SGD, RMSprop)

### 2. MobileNetV2

* Pretrained on ImageNet
* Used as the encoder in a segmentation head
* Efficient but struggled due to domain mismatch (natural vs. street scenes)

### 3. DeepLab

* Based on ResNet encoder with ASPP
* Pretrained on ImageNet
* Best performance across all metrics

## Results

| Model       | Mean IoU |
| ----------- | -------- |
| **DeepLab**     | **0.5242**   |
| UNet        | 0.4507   |
| MobileNetV2 | 0.3836   |

* DeepLab performed best due to its powerful feature extractor.
* All models struggled with shadowed regions and fine-grained class distinctions.

## Future Work

* Trainable activation functions
* Advanced augmentation strategies (e.g., ClassMix)
* Hyperparameter optimization (grid/Bayesian search)
* Better handling of the Void class
* Exploration of additional architectures

## Project Contributors

* Cristian Longoni
* Robin Smith 
* Sergio Verga
