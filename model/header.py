import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch_optimizer as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from torchmetrics.functional import accuracy, precision, recall, f1_score, jaccard_index

# CUDA memory config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Initialize GradScaler for mixed precision training
scaler = GradScaler()