import torch
import numpy as np
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)

        return torch.relu(x + identity)


class Improved3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.block2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.block3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.block4 = ResidualBlock(128, 256)
        self.pool4 = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, 3)
        self.plane_weights = nn.Parameter(torch.ones(3))

    def forward_plane(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        return x.view(x.size(0), -1)

    def forward(self, sag, cor, axi):
        f = torch.stack([
            self.forward_plane(sag),
            self.forward_plane(cor),
            self.forward_plane(axi)
        ], dim=1)

        weights = torch.softmax(self.plane_weights * 2, dim=0)
        fused = (f * weights[None,:,None]).sum(dim=1)

        return self.classifier(self.dropout(fused))

def load_production_model(weights_path: str):
    model = Improved3DCNN()
    
    # Load the master checkpoint dictionary onto CPU
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # FIXED: Extract the actual weights out of the "model_state_dict" key
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def preprocess_volume(vol_array: np.ndarray, max_slices=32) -> torch.Tensor:
    """Standardizes input shape to match training parameters."""
    vol = vol_array.astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-5)
    
    if vol.shape[0] >= max_slices:
        vol = vol[:max_slices]
    else:
        pad = max_slices - vol.shape[0]
        vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)))
        
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0) # [B=1, C=1, S, H, W]        