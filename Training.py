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
def organize_files(source_folder):
    files = os.listdir(source_folder)
    car_brands = ['hyundai', 'volkswagen', 'lexus', 'mazda', 'opel', 'toyota', 'mercedes']
    for brand in car_brands:
        os.makedirs(os.path.join(source_folder, brand), exist_ok=True)

    for file_name in files:
        brand_name = file_name.split('_')[0]
        if brand_name == 'mercy' or brand_name == 'mercedes':
            brand_name = 'mercedes' 
        shutil.move(
            os.path.join(source_folder, file_name),
            os.path.join(source_folder, brand_name, file_name)
        )

    print("Files organized successfully!")

if __name__ == "__main__":
    source_folder = input("Masukkan path: ")
    organize_files(source_folder)

import os
from pathlib import Path
path_traindata = '/home/arjuna/Documents/Delameta/Train'
batch_size = 64
def membuka(dir_path):
 
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

membuka(path_traindata)
train_loader = torch.utils.data.DataLoader(path_traindata, batch_size=batch_size, shuffle=True, num_workers=2)
print("Jumlah Epoch Training: ", len(train_loader)*batch_size)

path_testdata = '/home/arjuna/Documents/Delameta/Test'
test_loader = torch.utils.data.DataLoader(path_testdata, batch_size=batch_size, shuffle=False, num_workers=2)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('hyundai', 'lexus', 'mazda', 'mercedes', 'opel', 'toyota', 'fvolkswagen')
           
def membuka(dir_path):
 
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

membuka(path_testdata)


import torchinfo
    
from torchinfo import summary
summary(model, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size 
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
train_data = datasets.ImageFolder('/home/arjuna/Documents/Delameta/Train', transform=transforms)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.ImageFolder('/home/arjuna/Documents/Delameta/Test', transform=transforms)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),         
])


model = Delameta_klasifikasi().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
writer = SummaryWriter()

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()  

        # Move tensors to GPU if available
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    # Print the average loss per epoch
    epoch_loss = running_loss / len(train_data)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Write the loss to TensorBoard
    writer.add_scalar('Training Loss', epoch_loss, epoch)

# Close the TensorBoard writer
torch.save(model.state_dict(), 'Klasifikasi.pt')
writer.close()

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


model = Delameta_klasifikasi()
model.load_state_dict(torch.load('/home/arjuna/Documents/Delameta/32/Klasifikasi.pt'))
model.eval()

def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Example usage:
image_path = '/home/arjuna/Documents/Delameta/Test/mazda/15.jpg'
predicted_label = predict_image(image_path, model, transform)
print(f"Predicted label: {predicted_label}")

