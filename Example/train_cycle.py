import os 
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class CIFARDataset(Dataset):
    MEAN = (0.485, 0.456, 0.406) 
    STD = (0.229, 0.224, 0.225)

    def __init__(self, img_path, labels=None, train=True):
        super().__init__()
        self.img_path = img_path
        self.labels = labels
        self.train = train

        if not train:
            self.transform = A.Compose([
                A.ToFloat(),
                A.Normalize(self.MEAN, self.STD, max_pixel_value=255),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.ToFloat(),
                A.Normalize(self.MEAN, self.STD, max_pixel_value=255),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        path_to_img = self.img_path[index]
        if not path_to_img.exists():
            print(f"The file {path_to_img} does not exists")

        image = cv2.imread(path_to_img)
        if image is None:
            raise FileNotFoundError(f"Can't read image: {path_to_img}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed_img = self.transform(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)
            return transformed_img, label
        else:
            return transformed_img
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return self.pool(x)


class BaselineModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extractor = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.feature_extractor.append(ConvBlock(in_channel, out_channel))

        self.dropout = nn.Dropout(0.2)
        self.fc_proj = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, 10)
        
    def forward(self, x):
        for layer in self.feature_extractor:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc_proj(x))
        x = self.out(x)
        return x


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        running_loss += batch_size * loss.item()
        preds = torch.argmax(outputs, dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_correct / running_total
    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            loss = criterion(outputs, labels)
            batch_size = outputs.size(0)
            running_loss += batch_size * loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += batch_size

    epoch_loss = running_loss / running_total # or / len(val_loader.dataset)
    epoch_acc = running_correct / running_total # or / len(val_loader.dataset)
    return epoch_loss, epoch_acc


def train_loop(model, train_loader, val_loader, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    best_val_acc = -np.inf

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in tqdm(range(epochs), position=0, leave=True):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "model_weights.pt")

        tqdm.write(f"Epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
    
    return model
        

def preprocess_labels(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels


def split_train_data(path_to_train_img_dir, annotations_path):
    image_names = sorted(
        [image_path for image_path in os.listdir(path_to_train_img_dir)],
        key=lambda name: int(name.split('.')[0])
    )

    all_image_paths = [path_to_train_img_dir / image_name for image_name in image_names]
    # print(all_image_paths[:10])
    
    annotaions_column = pd.read_csv(annotations_path, encoding='utf-8')['label']
    all_labels = preprocess_labels(annotaions_column).tolist()

    # exit(0)

    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        all_image_paths,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )

    return train_img_paths, val_img_paths, train_labels, val_labels


if __name__ == "__main__":
    current_path = Path.cwd()
    train_img_dir = Path("./train")
    test_img_dir = Path("./test")
    annotations_path = Path("./trainLabels.csv")
    
    path_to_train_img_dir = current_path / train_img_dir
    path_to_test_img_dir = current_path / test_img_dir    

    train_img_paths, val_img_paths, train_labels, val_labels = split_train_data(
        path_to_train_img_dir, 
        annotations_path
    )

    train_dataset = CIFARDataset(
        img_path=train_img_paths,
        labels=train_labels,
        train=True
    )

    val_dataset = CIFARDataset(
        img_path=val_img_paths,
        labels=val_labels,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    in_channels = [3, 32, 64]
    out_channels = [32, 64, 128]
    model = BaselineModel(in_channels, out_channels)
    model.to(device)

    model = train_loop(model, train_loader, val_loader, device)
