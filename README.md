# Binary Image Segmentation using U-Net
This repository focuses on binary image segmentation utilizing the U-Net architecture. Its primary aim is to accurately segment images into two classes.

## Purpose
The repository provides a framework for binary image segmentation tasks. The U-Net architecture is chosen for its effectiveness in capturing complicated features, resulting in precise segmentation masks.

## Training
### Data Augmentation
The training process involves data augmentation techniques using the library albumentations to enhance model generalization and robustness. Techniques such as flipping, rotation, normalization are applied to augment the dataset.
### Loss Function and Optimizer
The model employs the `BCEWithLogitsLoss` (Binary Cross Entropy with Logits Loss) function, suitable for binary segmentation tasks. The Adam optimizer is used to efficiently update model weights during training.
## Image Processing
The whole input image is used for segmentation.

## Repository Structure
- `data/`: Contains images and masks used for training and validation.
- `model.py`: Includes the U-Net model architecture definition and related utilities.
- `train.py`: Python script for training the U-Net model.
- `test.py`: Script for evaluating the trained model on test data.
- `infer.py`: Script to perform inference

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/binary-segmentation.git
   cd binary-segmentation
2. Prepare the data (train_images, train_mask, val_images, val_mask)
3. Adjust configurations in train.py (e.g., file paths, hyperparameters)
4. Start training by running:
   ```bash
   python train.py
5. Evaluate the model's performance by running
   ```bash
   python test.py


## Acknowledgments
The U-Net architecture was proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in the paper U-Net: Convolutional Networks for Biomedical Image Segmentation.
The code it's been adapted from various open-source repositories and research papers, especially from Aladdin Persson [Repository](https://github.com/username/example-repo](https://github.com/aladdinpersson/Machine-Learning-Collection)https://github.com/aladdinpersson/Machine-Learning-Collection)
