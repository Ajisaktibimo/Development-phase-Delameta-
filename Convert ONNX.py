import torch.onnx
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
# Instantiate your model
model = Delameta_klasifikasi()

# Provide a sample input tensor
dummy_input = torch.randn(1, 3, 64, 64)  # Adjust the shape according to your input requirements

# Define the path to save the ONNX file
onnx_file_path = "Delameta_klasifikasi.onnx"

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True)

print(f"Model successfully exported to {onnx_file_path}")
import os
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Load your ONNX model
onnx_model_path = '/home/arjuna/Documents/Delameta/Delameta_klasifikasi.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match model's input size
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define the function to predict image label
def predict_image(image_path, ort_session, transform):
    # Open the image file using PIL
    image = Image.open(image_path)

    # Apply preprocessing transformations
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Convert the preprocessed image to float32
    image_float32 = image.numpy().astype(np.float32)

    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: image_float32}
    ort_outs = ort_session.run(None, ort_inputs)

    # Assuming you have a list of class labels
    class_labels = ['hyundai', 'lexus', 'mazda', 'opel', 'toyota', 'mercedes', 'volkswagen']  # Replace with your class labels

    predicted_label_index = np.argmax(ort_outs[0])
    if predicted_label_index < len(class_labels):
        predicted_label = class_labels[predicted_label_index]
    else:
        predicted_label = 'Unknown'

    return predicted_label

# Define the function to display image predictions with a grid
def display_image_predictions(test_data_folder, ort_session, transform, num_images=16, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Get a list of image files from the test data folder
    image_files = [os.path.join(test_data_folder, file) for file in os.listdir(test_data_folder)][:num_images]

    # Iterate over the image files
    for i, image_path in enumerate(image_files):
        # Predict the label for the image
        predicted_label = predict_image(image_path, ort_session, transform)

        # Plot the image and predicted label
        ax = axes[i // cols, i % cols]
        ax.imshow(Image.open(image_path))
        ax.set_title(f'Predicted: {predicted_label}')
        ax.axis('off')

    plt.show()

# Example usage:
test_data_folder = '/home/arjuna/Documents/Delameta/Test/lexus'
display_image_predictions(test_data_folder, ort_session, preprocess)

