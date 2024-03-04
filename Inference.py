import os
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from NeuralNetworkarchutecture1 import model
from NeuralNetworkarchutecture1 import Delameta_klasifikasi
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from Training import test_data
from Training import predict_image
from Training import test_loader
# Define the function to display image predictions with a grid
def display_image_predictions(test_data, model, transform, num_images=16, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Iterate over a subset of images from the test dataset
    for i in range(num_images):
        image_path, true_label = test_data.samples[i]
        image = Image.open(image_path)
        image = transform(image)  # Apply the transformation

        # Perform inference
        predicted_label = predict_image(image_path, model, transform)

        # Plot the image and predicted label
        ax = axes[i // cols, i % cols]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f'Predicted: {predicted_label}\nTrue: {true_label}')
        ax.axis('off')

    plt.show()

# Display image predictions
display_image_predictions(test_data, model, transforms)

correct = 0
total = 0
net = Delameta_klasifikasi()
net.load_state_dict(torch.load('Klasifikasi.pt'))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %') #60 persen