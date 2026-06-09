import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Define an ultra-lightweight 2D CNN architecture (~1.5 MB total size)
class PlaneGatekeeperCNN(nn.Module):
    def __init__(self):
        super(PlaneGatekeeperCNN, self).__init__()
        self.features = nn.Sequential(
            # Input: 1 channel (grayscale MRI slice) x 256 x 256
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Down to 128 x 128
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Down to 64 x 64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Standardize feature map size to 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # 3 outputs: 0=Axial, 1=Coronal, 2=Sagittal
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Dummy/Toy Dataset Generator to test pipeline compilation
# NOTE: Replace this with your actual MRNet folder parsing if you want to retrain it!
class MRNetPlaneDataset(Dataset):
    def __init__(self, num_samples=120):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy 2D slices (1, 256, 256) and mock targets
        mock_slice = torch.randn(1, 256, 256)
        label = idx % 3 # Even distribution of 0, 1, 2
        return mock_slice, label

# 3. Training Loop Execution
def train_model():
    print("Initializing Gatekeeper Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PlaneGatekeeperCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    dataset = MRNetPlaneDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model.train()
    for epoch in range(5): # Quick training footprint
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/5 | Loss: {running_loss/len(dataloader):.4f}")
        
    # Save the native weights blueprint
    torch.save(model.state_dict(), "gatekeeper_weights.pth")
    print("Success! 'gatekeeper_weights.pth' generated successfully.")

if __name__ == "__main__":
    train_model()