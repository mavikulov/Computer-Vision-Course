import os 
from dataclasses import dataclass
import PIL 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
        

class KeyFacepointsDataset(Dataset):
    def __init__(self, img_dir, transform, annotations=None, train=True):
        super().__init__()
        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.annotations_file = annotations if annotations else None

        self.image_names = sorted(
            [file for file in os.listdir(self.img_dir) if file.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.image_names)

    def resize_image_and_keypoints(self, image, keypoints):
        H, W = image.size[:2]
        new_w, new_h = 100, 100
        resized_image = T.functional.resize(image, (new_w, new_h))
        scale_x = new_w / float(W)
        scale_y = new_h / float(H)

        if keypoints is not None:
            resized_keypoints = keypoints.copy().astype(np.float32)
            resized_keypoints[0::2] = resized_keypoints[0::2] * scale_x
            resized_keypoints[1::2] = resized_keypoints[1::2] * scale_y
        else:
            resized_keypoints = None

        return resized_image, resized_keypoints

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')

        if self.train and self.annotations_file is not None:
            keypoints = np.array(self.annotations_file[self.image_names[index]]).astype(np.float32)
        else:
            keypoints = None

        resized_img, resized_keypoints = self.resize_image_and_keypoints(image, keypoints)

        if self.transform:
            transformed_img = self.transform(resized_img)

        if self.train:
            return transformed_img, torch.from_numpy(resized_keypoints)
        else:
            return transformed_img, self.image_names[index], image.size


# ====================== Models ======================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class KeyPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        conv_channels = [
            (3, 32),
            (32, 32),
            (32, 64),
            (64, 64),
            (64, 128),
            (128, 128)
        ]

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_channels, out_channels) for in_channels, out_channels in conv_channels
        ])

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_proj = nn.Linear(18432, 128)
        self.fc_out = nn.Linear(128, 28)

    def forward(self, x):
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            if i % 2 == 1:
                x = self.max_pool(x)

        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc_proj(x)))
        x = self.fc_out(x)
        return x 


# Read from https://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_Wing_Loss_for_CVPR_2018_paper.pdf
class WingLoss(nn.Module):
    def __init__(self, w=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        device = predictions.device
        w_tensor = torch.tensor(self.w, dtype=torch.float32, device=device)
        epsilon_tensor = torch.tensor(self.epsilon, dtype=torch.float32, device=device)
        C = w_tensor - w_tensor * torch.log(1 + w_tensor / epsilon_tensor)
        
        diff = torch.abs(predictions - targets)
        losses = torch.where(
            diff < w_tensor,
            w_tensor * torch.log(1 + diff / epsilon_tensor),
            diff - C
        )
        return losses.mean()
    

# ====================== Training And Validation ======================
@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-3
    model_save_path: str = "facepoints_model.pt"
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(train_loader, model, config, fast_train=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    if fast_train:
        config.epochs = 2

    train_loss_history = []
    best_loss = np.inf

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for _, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs).to(config.device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
        
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        if train_loss < best_loss and not fast_train:
            torch.save(model.state_dict(), config.model_save_path)
            best_loss = train_loss

        # print(f"Epoch {epoch} / {config.epochs} | Loss = {train_loss}") 

    return model


# ====================== Train Detector Function ======================
def train_detector(train_gt, train_img_dir, fast_train=False):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    train_dataset = KeyFacepointsDataset(
        img_dir=train_img_dir,
        transform=transform,
        annotations=train_gt,
        train=True
    )

    config = TrainingConfig()
    
    if fast_train:
        config.epochs = 2
        config.device = 'cpu'
        config.batch_size = 8

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    model = KeyPredictor().to(config.device)

    return train(train_loader, model, config, fast_train)
    

# ====================== Detect Function ======================
def detect(model_path, test_img_dir):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    test_dataset = KeyFacepointsDataset(
        img_dir=test_img_dir,
        transform=transform,
        annotations=None,
        train=False
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = KeyPredictor().to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    pred_dct = {}

    for resized_image, path_to_img, img_size in test_loader:
        keypoints_pred = model(resized_image).detach().numpy().reshape(28)
        keypoints_pred[::2] = keypoints_pred[::2] * img_size[0].item() / 100
        keypoints_pred[1::2] = keypoints_pred[1::2] * img_size[1].item() / 100
        pred_dct[path_to_img[0]] = keypoints_pred

    return pred_dct
