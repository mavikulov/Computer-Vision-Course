import os 
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BirdsDataset(Dataset):
    MEAN = (0.485, 0.456, 0.406) 
    STD = (0.229, 0.224, 0.225)

    def __init__(self, image_paths, labels=None, train=True):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.train = train

        self.test_transform = A.Compose([
            A.Resize(224, 224),
            A.ToFloat(),
            A.Normalize(self.MEAN, self.STD),
            ToTensorV2()
        ])

        self.train_transform = A.Compose([
            A.OneOf([
                A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5)
            ], p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift()
            ], p=0.5),

            A.Resize(224, 224),
            A.ToFloat(),
            A.Normalize(self.MEAN, self.STD),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_name = self.image_paths[index]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = self.train_transform if self.train else self.test_transform
        transformed_image = transform(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return transformed_image, label
        else:
            return transformed_image, img_name


def get_model(device, num_freeze_layers=4, fast_train=True):
    weights = None if fast_train else 'DEFAULT'
    mobilenet = torchvision.models.mobilenet_v2(weights=weights, progress=False).to(device)
    mobilenet.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(),
        nn.Dropout(0.2),
        nn.Linear(512, 50)
    ).to(device)

    for child in list(mobilenet.features.children())[:-num_freeze_layers]:
        for params in child.parameters():
            params.requires_grad = False 

    return mobilenet


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_save_path: str = 'birds_model.pt'


def train_model(model, train_loader, optimizer, criterion, config):
    train_loss_history = []
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        predictions = model(inputs)
        loss = criterion(predictions, targets).mean()

        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.cpu().data.numpy())
    return np.mean(train_loss_history)


def eval_model(model, test_loader, criterion, config):
    accuracy = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.device)
            logits = model(inputs)
            y_pred = logits.max(1)[1].data
            accuracy.append(np.mean((targets.cpu() == y_pred.cpu()).numpy()))
    return np.mean(accuracy)
    

def train_loop(model, config, train_loader, test_loader, fast_train=True):
    best_val_acc = -np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss().to(config.device)
    for epoch in range(config.epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, config)
        val_accuracy = eval_model(model, test_loader, criterion, config)

        if not fast_train and val_accuracy > best_val_acc:
            torch.save(model.state_dict(), config.model_save_path)
            best_val_acc = val_accuracy

    return model


def train_classifier(train_gt, train_img_dir, fast_train=True):
    all_image_paths = [
        os.path.join(train_img_dir, file) for file in sorted(os.listdir(train_img_dir))
    ]

    all_labels = [train_gt[os.path.basename(path)] for path in all_image_paths]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths,
        all_labels,
        test_size=0.1,
        random_state=42,
        stratify=all_labels
    )

    config = TrainingConfig()

    if fast_train:
        config.batch_size = 16
        config.epochs = 2
        config.device = 'cpu'

    train_dataset = BirdsDataset(train_paths, train_labels)
    val_dataset = BirdsDataset(val_paths, val_labels, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = get_model(config.device, fast_train=fast_train)
    
    return train_loop(model, config, train_dataloader, val_dataloader, fast_train)


def classify(model_path, test_img_dir):
    all_image_paths = [
        os.path.join(test_img_dir, file) for file in sorted(os.listdir(test_img_dir))
    ]

    test_dataset = BirdsDataset(all_image_paths, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = get_model(device='cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    pred_dct = {}

    with torch.no_grad():
        for image, img_path in test_loader:
            pred = model(image).argmax(dim=1).item()
            img_name = os.path.basename(img_path[0])
            pred_dct[img_name] = pred

    return pred_dct
