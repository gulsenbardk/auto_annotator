from header import *
from utils import *
class SimplePointNet(nn.Module):
    def __init__(self, input_dim=6, num_classes=10):
        super(SimplePointNet, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Global feature

        self.classifier = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        # x: [B, N, input_dim]
        x = x.permute(0, 2, 1)  # [B, input_dim, N]

        x1 = self.mlp1(x)  # [B, 64, N]
        x2 = self.mlp2(x1) # [B, 128, N]
        x3 = self.mlp3(x2) # [B, 256, N]

        global_feat = self.global_pool(x3)         # [B, 256, 1]
        global_feat = global_feat.expand(-1, -1, x.size(2))  # [B, 256, N]

        concat = torch.cat([x3, global_feat], dim=1)  # [B, 512, N]

        out = self.classifier(concat)  # [B, num_classes, N]
        out = out.permute(0, 2, 1)     # [B, N, num_classes]

        return out

class SimplePointNetConv(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=10, use_bn=True, dropout=0.0):
        super().__init__()
        
        def conv_block(in_c, out_c):
            layers = [nn.Conv1d(in_c, out_c, 1)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        # Input feature transform
        self.mlp1 = nn.Sequential(
            conv_block(input_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim)
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            conv_block(hidden_dim * 2, hidden_dim),
            nn.Conv1d(hidden_dim, num_classes, 1)
        )
    def forward(self, x):  # x: (B, N, 3)
        #if x.dim() == 2:
        #    x = x.unsqueeze(0)  # (1, N, 3)
        print("x shape", x.shape)
        #import ipdb 
        #ipdb.set_trace()
        if x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (B, N, 3), got {x.shape}")
        x = x.permute(0, 2, 1)  # → (B, 3, N)
        local_feat = self.mlp1(x)  # → (B, hidden_dim, N)
        # Global feature by max pooling
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]  # (B, hidden_dim, 1)
        global_feat = global_feat.expand(-1, -1, x.size(2))  # → (B, hidden_dim, N)
        feat = torch.cat([local_feat, global_feat], dim=1)  # → (B, hidden_dim*2, N)
        out = self.classifier(feat)  # → (B, num_classes, N)
        return out.permute(0, 2, 1)  # → (B, N, num_classes)

"""
class GeoPointNet(nn.Module):
    def __init__(self, input_dim=6, num_classes=10):
        super(GeoPointNet, self).__init__()

        # Per-point feature extraction with Conv1d (kernel_size=1)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Classification layers (fully connected)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, N, 6]
        x = x.permute(0, 2, 1)        # [B, 6, N]

        x = F.relu(self.conv1(x))     # [B, 64, N]
        x = F.relu(self.conv2(x))     # [B, 128, N]
        x = F.relu(self.conv3(x))     # [B, 1024, N]

        x = self.global_pool(x)       # [B, 1024, 1]
        x = x.squeeze(-1)             # [B, 1024]

        x = F.relu(self.fc1(x))       # [B, 512]
        x = F.relu(self.fc2(x))       # [B, 256]
        x = self.dropout(x)
        x = self.fc3(x)               # [B, num_classes]

        return x
"""

class GeoPointNet(nn.Module):
    def __init__(self, input_dim=6, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, num_classes, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, N, 6] → convert to [B, 6, N] for conv1d
        x = x.permute(0, 2, 1)  # [B, 6, N]

        x = F.relu(self.conv1(x))  # [B, 64, N]
        x = F.relu(self.conv2(x))  # [B, 128, N]
        x = F.relu(self.conv3(x))  # [B, 256, N]
        x = F.relu(self.conv4(x))  # [B, 512, N]
        x = F.relu(self.conv5(x))  # [B, 256, N]
        x = self.dropout(x)
        x = self.conv6(x)          # [B, num_classes, N]

        x = x.permute(0, 2, 1)    # [B, N, num_classes]
        return x