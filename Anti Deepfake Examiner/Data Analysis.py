# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_path = 'D://FYP Work//dataset//training'

# Function to load and preprocess images
def load_images(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        for filename in tqdm(os.listdir(label_path), desc=f"Loading {label} images"):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images(dataset_path)


# Basic information about the dataset
print("Number of images:", len(images))
print("Number of classes:", len(np.unique(labels)))
print("Class distribution:", np.unique(labels, return_counts=True))

# Display random samples from each class
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Random Samples from the Dataset')

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(len(images))
    ax.imshow(images[idx])
    ax.set_title(labels[idx])
    ax.axis('off')

plt.show()


# Calculate basic statistics
mean_pixel = np.mean(images, axis=(0, 1, 2))
std_pixel = np.std(images, axis=(0, 1, 2))
median_pixel = np.median(images, axis=(0, 1, 2))
variance_pixel = np.var(images, axis=(0, 1, 2))
print("Average Mean Pixel Values:", np.average(mean_pixel))
print("Average Standard Deviation of Pixel Values:", np.average(std_pixel))
print("Average Median Values:", np.average(median_pixel))
print("Average Variance of Pixel Values:", np.average(variance_pixel))

# Visualize pixel intensity distribution for a sample image
sample_image = images[np.random.randint(len(images))]
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image)
plt.title('Sample Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.hist(sample_image[:, :, 0].ravel(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('Red Channel Histogram')

plt.subplot(1, 3, 3)
plt.hist(sample_image[:, :, 1].ravel(), bins=256, color='green', alpha=0.7, rwidth=0.8)
plt.title('Green Channel Histogram')

plt.tight_layout()
plt.show()

# Calculate and visualize tendencies
mean_image = np.mean(images, axis=0)
median_image = np.median(images, axis=0)

# Display mean and median images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(mean_image.astype(np.uint8))
axes[0].set_title('Mean Image')
axes[0].axis('off')

axes[1].imshow(median_image.astype(np.uint8))
axes[1].set_title('Median Image')
axes[1].axis('off')

plt.show()

print(mean_image.sum())
print(median_image.sum())

# Calculate and visualize dispersion
variance_image = np.var(images, axis=0)

# Display the variance image
plt.figure(figsize=(6, 6))
plt.imshow(variance_image.astype(np.uint8))
plt.title('Variance Image')
plt.axis('off')
plt.show()

# Calculate the correlation matrix for the images
# Resize images before calculating the correlation matrix
resized_images = [cv2.resize(img, (50, 50)) for img in images]

# Calculate the correlation matrix for the resized images
correlation_matrix = np.corrcoef(np.array(resized_images).reshape(len(images), -1), rowvar=False)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()
print(correlation_matrix)
correlationavg = np.average(correlation_matrix)
print(correlationavg)


# Visualize pixel intensity distribution for each channel using histograms
plt.figure(figsize=(12, 6))

for i in range(3):  # Assuming RGB channels
    plt.subplot(2, 3, i + 1)
    plt.hist(images[:, :, :, i].ravel(), bins=256, color=['red', 'green', 'blue'][i], alpha=0.7, rwidth=0.8)
    plt.title(f'Channel {i + 1} Histogram')

plt.subplot(2, 3, 4)
plt.hist(images[:, :, :].ravel(), bins=256, color='gray', alpha=0.7, rwidth=0.8)
plt.title('Overall Image Histogram')

plt.tight_layout()
plt.show()





from skimage.feature import greycomatrix

import matplotlib.pyplot as plt

# Convert images to grayscale
gray_images = [color.rgb2gray(img) for img in images]

# Compute Haralick features for a sample image
haralick_features = greycomatrix(gray_images[0], [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

# Visualize Haralick features
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray_images[0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(haralick_features[:, :, 0, 0], cmap='hot', interpolation='nearest')
plt.title('Haralick Feature 1')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(haralick_features[:, :, 0, 1], cmap='hot', interpolation='nearest')
plt.title('Haralick Feature 2')
plt.axis('off')

plt.show()

from skimage import measure
from skimage import io, color
# Convert images to grayscale
gray_images = [color.rgb2gray(img) for img in images]

# Find contours in a sample image
contours = measure.find_contours(gray_images[0], 0.8)

# Visualize contours
plt.figure(figsize=(8, 8))
plt.imshow(gray_images[0], cmap='gray')
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.title('Contour Detection')
plt.axis('off')
plt.show()

# Convert images to LAB color space
lab_images = [color.rgb2lab(img) for img in images]
# Visualize LAB color space for a sample image
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(images[0])
plt.title('Original Image')
plt.axis('off')


for i, channel in enumerate(['L', 'A', 'B']):
    plt.subplot(1, 4, i + 2)
    plt.imshow(lab_images[0][:, :, i], cmap='gray')
    plt.title(f'LAB {channel}')
    plt.axis('off')

plt.show()

