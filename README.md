# intel-2024-

#To create a data set
import cv2
import os
import numpy as np
from google.colab import files

# Upload images (assuming all images are uploaded in one go)
print("Upload your images:")
uploaded = files.upload()

# Create output directory
output_dir = '/content/pixelated_images/'
os.makedirs(output_dir, exist_ok=True)

# Function to pixelate image
def pixelate_image(image, block_size):
    (h, w) = image.shape[:2]
    temp = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

# Process each uploaded image
for filename in uploaded.keys():
    image = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)

    for m in range(3, 43, 2):
        pixelated_image = pixelate_image(image, m)
        m_adjusted = (m - 1) // 2

        output_filename = f'{os.path.splitext(filename)[0]}_p{m_adjusted}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, pixelated_image)

        print(f"Processed block size {m}x{m} for image {filename}")

print("Processing completed.")

import os
# List files in the output directory
output_files = os.listdir(output_dir)
for file in output_files:
    print(file)

from google.colab import files
# Download all files in the output directory
for file in output_files:
    files.download(os.path.join(output_dir, file))

import shutil
# Create a zip file of the output directory
shutil.make_archive('/content/pixelated_images', 'zip', output_dir)

# Download the zip file
files.download('/content/pixelated_images.zip')


#main model

from google.colab import drive
drive.mount('/content/drive')
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageDataset('/content/drive/MyDrive/train_images', transform=transform)
test_dataset = ImageDataset('/content/drive/MyDrive/test_images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 

for epoch in range(num_epochs):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        inputs = data.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)
            outputs = model(inputs)



torch.save(model.state_dict(), '/content/drive/MyDrive/trained_model.pth')

from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(true_images, generated_images):
    # Convert tensors to numpy arrays
    true_images = true_images.cpu().numpy().reshape(-1)
    generated_images = generated_images.cpu().numpy().reshape(-1)

    # Convert to binary classification problem for F1 score
    true_images_binary = (true_images > 0.5).astype(int)
    generated_images_binary = (generated_images > 0.5).astype(int)

    f1 = f1_score(true_images_binary, generated_images_binary)
    accuracy = accuracy_score(true_images_binary, generated_images_binary)

    return f1, accuracy

model.eval()
f1_scores = []
accuracies = []

with torch.no_grad():
    for data in test_loader:
        inputs = data.to(device)
        outputs = model(inputs)
        f1, acc = calculate_metrics(inputs, outputs)
        f1_scores.append(f1)
        accuracies.append(acc)

mean_f1 = sum(f1_scores) / len(f1_scores)
mean_accuracy = sum(accuracies) / len(accuracies)

print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Mean Accuracy: {mean_accuracy:.4f}")

model.eval()
with torch.no_grad():
    total_mse = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(test_loader):
        inputs = data[0].to(device)  # Assuming your data loader returns (input, target) pairs
        targets = data[1].to(device)

        # Print statements to debug
        print(f"Batch [{batch_idx+1}/{len(test_loader)}], Input Shape: {inputs.shape}, Target Shape: {targets.shape}")

        # Ensure there are actually samples in the test set
        if inputs.size(0) == 0:
            continue

        # Forward pass
        outputs = model(inputs)

        # Calculate MSE
        batch_mse = criterion(outputs, inputs).item()
        total_mse += batch_mse
        num_batches += 1





