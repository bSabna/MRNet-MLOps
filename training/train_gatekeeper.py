import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
 

# 1. MODEL ARCHITECTURE

 
class PlaneGatekeeperCNN(nn.Module):
    """
    Ultra-lightweight 2D CNN (~1.5 MB).
    Input : (B, 1, 256, 256) — single grayscale MRI slice
    Output: (B, 3)           — logits for Axial / Coronal / Sagittal
    """
    def __init__(self):
        super(PlaneGatekeeperCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                        
 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))               
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
 
 

# 2. DATASET

 
class MRNetPlaneDataset(Dataset):
    """
    Loads real MRNet .npy volumes and extracts the middle slice
    from each volume as a 2D training sample.
 
    Label mapping:
        0 = Axial
        1 = Coronal
        2 = Sagittal
    """
    PLANE_LABELS = {"axial": 0, "coronal": 1, "sagittal": 2}
 
    def __init__(self, data_root, target_size=(256, 256)):
        """
        Args:
            data_root : path to MRNet-v1.0/ folder
            target_size: resize each slice to this (H, W)
        """
        self.samples = []        # list of (file_path, label)
        self.target_size = target_size
 
        train_dir = os.path.join(data_root, "train")
 
        for plane, label in self.PLANE_LABELS.items():
            plane_dir = os.path.join(train_dir, plane)
            if not os.path.isdir(plane_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {plane_dir}\n"
                    f"Make sure data_root points to MRNet-v1.0/"
                )
            files = [f for f in os.listdir(plane_dir) if f.endswith(".npy")]
            for fname in files:
                self.samples.append((os.path.join(plane_dir, fname), label))
 
        print(f"\n Dataset loaded: {len(self.samples)} volumes total")
        for plane, label in self.PLANE_LABELS.items():
            count = sum(1 for _, l in self.samples if l == label)
            print(f"   {plane.capitalize():10s}: {count} volumes")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
 
        # Load volume: shape (num_slices, H, W)
        volume = np.load(path)
 
        # Extract middle slice
        mid = volume.shape[0] // 2
        slice_2d = volume[mid].astype(np.float32)   # (H, W)
 
        # Normalize to [0, 1]
        vmin, vmax = slice_2d.min(), slice_2d.max()
        if vmax > vmin:
            slice_2d = (slice_2d - vmin) / (vmax - vmin)
 
        # Resize to target_size using simple interpolation
        slice_2d = self._resize(slice_2d, self.target_size)
 
        # Add channel dim → (1, H, W)
        tensor = torch.tensor(slice_2d).unsqueeze(0)
        return tensor, label
 
    @staticmethod
    def _resize(arr, target_size):
        """Resize a 2D numpy array using nearest-neighbor (no cv2 needed)."""
        from PIL import Image
        img = Image.fromarray(arr)
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        return np.array(img)
 
 

# 3. TRAINING

 
def train(data_root, epochs=10, batch_size=32, lr=0.001, val_split=0.15):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
 
    # Dataset
    full_dataset = MRNetPlaneDataset(data_root)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
 
    print(f"\n Train: {train_size} | Val: {val_size}")
 
    # Model
    model = PlaneGatekeeperCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
 
    best_val_acc = 0.0
 
    print("\n Starting training...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>8}")
    print("-" * 45)
 
    for epoch in range(1, epochs + 1):
 
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
 
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
 
        scheduler.step()
 
        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
 
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        avg_loss  = train_loss    / len(train_loader)
 
        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc:>8.1%} | {val_acc:>7.1%}")
 
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "gatekeeper_weights.pth")
            print(f"         Best model saved (val_acc={val_acc:.1%})")
 
    # ── Final Report ──
    print("\n" + "=" * 45)
    print(f" Training complete. Best Val Accuracy: {best_val_acc:.1%}")
    print("\n Classification Report (best model on validation set):")
    print(classification_report(all_labels, all_preds,
                                 target_names=["Axial", "Coronal", "Sagittal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n Weights saved to: gatekeeper_weights.pth")
 

# 4. ENTRY POINT

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI Plane Gatekeeper CNN")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help='Path to MRNet-v1.0 folder. Example: "C:/Users/bsabn/MRNet-v1.0"'
    )
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()
 
    train(
        data_root  = args.data_root,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )