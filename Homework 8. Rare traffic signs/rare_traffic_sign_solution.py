# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import typing
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import lightning as L
import numpy as np
import scipy
import skimage
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# !Этих импортов достаточно для решения данного задания


CLASSES_CNT = 205
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.

    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root_folders: typing.List[str],
        path_to_classes_json: str,
    ) -> None:
        super().__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.samples = []

        for root_folder in root_folders:
            for img_dir_name in os.listdir(root_folder):
                path_to_images = os.path.join(root_folder, img_dir_name)
                for img_name in os.listdir(path_to_images):
                    path_to_img = os.path.join(path_to_images, img_name)
                    cls_index = self.class_to_idx[img_dir_name]
                    self.samples.append((path_to_img, cls_index))

        ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = {key: [] for key in range(len(self.classes))}

        for i, sample in enumerate(self.samples):
            self.classes_to_samples[sample[1]].append(i)

        ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(232, 232),
            A.Normalize(self.MEAN, self.STD),
            ToTensorV2()
        ])

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        path_to_image, cls_index = self.samples[index]
        image = np.array(Image.open(path_to_image).convert('RGB'))
        transformed_image = self.transform(image=image)['image']
        return transformed_image, path_to_image, cls_index

    @staticmethod
    def get_classes(
        path_to_classes_json,
    ) -> typing.Tuple[typing.List[str], typing.Mapping[str, int]]:
        """
        Считывает из classes.json информацию о классах.

        :param path_to_classes_json: путь до classes.json
        """

        ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        with open(path_to_classes_json, 'r') as file:
            classes_json = json.load(file)    

        ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        class_to_idx = {cl: value['id'] for cl, value in classes_json.items()}
        classes = list(classes_json.keys())
        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        ### YOUR CODE HERE
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.

    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """

    def __init__(
        self,
        root: str,
        path_to_classes_json: str,
        annotations_file: str = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        ### YOUR CODE HERE - список путей до картинок
        self.samples = [img for img in os.listdir(root) if img.endswith('.png')]

        ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(DatasetRTSD.MEAN, DatasetRTSD.STD),
            ToTensorV2()
        ])
        self.targets = None
        if annotations_file is not None:
            ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            self.targets = {}
            with open(annotations_file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                # Read from StackOverflow about skipping the first row with the header in csv file
                next(reader, None)
                for img_name, class_name in reader:
                    self.targets[img_name] = self.class_to_idx[class_name]

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str, int]:
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        path_to_image = self.samples[index]
        image = np.array(Image.open(os.path.join(self.root, path_to_image)).convert('RGB'))
        transformed_image = self.transform(image=image)['image']

        if self.targets is not None:
            return transformed_image, path_to_image, self.targets[path_to_image]
        else:
            return transformed_image, path_to_image, -1

    def __len__(self) -> int:
        """
        Возвращает размер датасета (количество сэмплов).
        """
        ### YOUR CODE HERE
        return len(self.samples)


class CustomNetwork(L.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.

    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """

    def __init__(
        self,
        features_criterion: (
            typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        internal_features: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters()
        ### YOUR CODE HERE
        self.features_criterion = features_criterion
        self.resnet50 = torchvision.models.resnet50(weights='IMAGENET1K_V2', progress=False)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(2048, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, 205)
        )

        for name, parameter in self.resnet50.named_parameters():
            if "fc" not in name:
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Функция для прогона данных через нейронную сеть.
        Возвращает два тензора: внутреннее представление и логиты после слоя-классификатора.
        """
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)

        features = torch.flatten(x, 1)
        logits = self.resnet50.fc(features)

        return features, logits
    
    def training_step(self, batch, batch_idx):
        images, _, labels = batch
        features, logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        outputs = torch.argmax(logits, dim=1)
        accuracy = (labels == outputs).float().mean()

        if self.features_criterion is not None:
            features_loss = self.features_criterion(features, labels)
            loss += features_loss
            self.log('train_features_loss', features_loss, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)

        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch 
    #     _, logits = self.resnet50(images)
    #     loss = F.cross_entropy(logits, labels)
    #     outputs = torch.argmax(logits, dim=1)
    #     accuracy = (labels == outputs).float().mean()

    #     self.log('val_loss', loss, prog_bar=True)
    #     self.log('val_acc', accuracy, prog_bar=True)

    #     return loss

    def configure_optimizers(self):
        learnable_parameters = [
            param for param in self.resnet50.parameters() if param.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            params=learnable_parameters,
            lr=1e-4,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.2
        )

        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param x: батч с картинками
        """
        self.eval()
        with torch.no_grad():
            _, logits = self.forward(x)
            outputs = torch.argmax(logits, dim=1)
        return outputs.cpu().numpy()

def train_simple_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на исходных данных.
    """
    ### YOUR CODE HERE
    model = CustomNetwork()
    train_dataset = DatasetRTSD(
        root_folders=['./cropped-train'],
        path_to_classes_json='./classes.json'
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=11
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='train_acc',           
        mode='max',                  
        save_top_k=1,                
        filename='best-model',       
        dirpath='./',               
        save_weights_only=True,      
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloader)
    
    if checkpoint_callback.best_model_path:
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        torch.save(best_checkpoint['state_dict'], 'simple_model.pt')
    
    return model

def apply_classifier(
    model: torch.nn.Module,
    test_folder: str,
    path_to_classes_json: str,
) -> typing.List[typing.Mapping[str, typing.Any]]:
    """
    Функция, которая применяет модель и получает её предсказания.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = []
    test_dataset = TestData(test_folder, path_to_classes_json)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()

    with torch.no_grad():
        for image, path_to_image, cls_index in test_dataloader:
            image = image.to(model.device)
            outputs = model.predict(image)
            for i, output in enumerate(outputs):
                results.append({
                    'filename': path_to_image[i],
                    'class': test_dataset.classes[output]
                })

    return results


def test_classifier(
    model: torch.nn.Module,
    test_folder: str,
    annotations_file: str,
) -> typing.Tuple[float, float, float]:
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.

    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    path_to_classes_json = "./classes.json"
    results = apply_classifier(model, test_folder, path_to_classes_json)
    results = {}

    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.

    :param background_path: путь до папки с изображениями фона
    """

    def __init__(self, background_path: str) -> None:
        super().__init__()
        ### YOUR CODE HERE

    ### Для каждого из необходимых преобразований над иконками/картинками,
    ### напишите вспомогательную функцию приблизительно следующего вида:
    ###
    ### @staticmethod
    ### def discombobulate_icon(icon: np.ndarray) -> np.ndarray:
    ###     ### YOUR CODE HERE
    ###     return ...
    ###
    ### Постарайтесь не использовать готовые библиотечные функции для
    ### аугментаций и преобразования картинок, а реализовать их
    ### "из первых принципов" на numpy

    def get_sample(self, icon: np.ndarray) -> np.ndarray:
        """
        Функция, встраивающая иконку на случайное изображение фона.

        :param icon: Массив с изображением иконки
        """
        ### YOUR CODE HERE
        icon = ...
        ### YOUR CODE HERE - случайное изображение фона
        bg = ...
        return  ### YOUR CODE HERE


def generate_one_icon(args: typing.Tuple[str, str, str, int]) -> None:
    """
    Функция, генерирующая синтетические данные для одного класса.

    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


def generate_all_data(
    output_folder: str,
    icons_path: str,
    background_path: str,
    samples_per_class: int = 1000,
) -> None:
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.

    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    with ProcessPoolExecutor(8) as executor:
        params = [
            [
                os.path.join(icons_path, icon_file),
                output_folder,
                background_path,
                samples_per_class,
            ]
            for icon_file in os.listdir(icons_path)
        ]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier() -> torch.nn.Module:
    """
    Функция для обучения простого классификатора на смеси исходных и ситетических данных.
    """
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """

    def __init__(self, margin: float) -> None:
        super().__init__()
        ### YOUR CODE HERE

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Функция, вычисляющая loss-функцию на признаки предпоследнего слоя нейросети.

        :param outputs: Признаки с предпоследнего слоя нейросети
        :param labels: Реальные метки объектов
        """
        ### YOUR CODE HERE


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.

    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """

    def __init__(
        self,
        data_source: DatasetRTSD,
        elems_per_class: int,
        classes_per_batch: int,
    ) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        ### YOUR CODE HERE

    def __len__(self) -> None:
        """
        Возвращает общее количество батчей.
        """
        ### YOUR CODE HERE


def train_better_model() -> torch.nn.Module:
    """
    Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки.
    """
    ### YOUR CODE HERE
    return model


class ModelWithHead(CustomNetwork):
    """
    Класс, реализующий модель с головой из kNN.

    :param n_neighbors: Количество соседей в методе ближайших соседей
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.eval()
        ### YOUR CODE HERE

    def load_nn(self, nn_weights_path: str) -> None:
        """
        Функция, загружающая веса обученной нейросети.

        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE

    def load_head(self, knn_path: str) -> None:
        """
        Функция, загружающая веса kNN (с помощью pickle).

        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE

    def save_head(self, knn_path: str) -> None:
        """
        Функция, сохраняющая веса kNN (с помощью pickle).

        :param knn_path: Путь, куда надо сохранить веса kNN
        """
        ### YOUR CODE HERE

    def train_head(self, indexloader: torch.utils.data.DataLoader) -> None:
        """
        Функция, обучающая голову kNN.

        :param indexloader: Загрузчик данных для обучения kNN
        """
        ### YOUR CODE HERE

    def predict(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.

        :param imgs: батч с картинками
        """
        ### YOUR CODE HERE - предсказание нейросетевой модели
        features, model_pred = ...
        features = features / np.linalg.norm(features, axis=1)[:, None]
        ### YOUR CODE HERE - предсказание kNN на features
        knn_pred = ...
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: DatasetRTSD, examples_per_class: int) -> None:
        ### YOUR CODE HERE
        pass

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        return  ### YOUR CODE HERE

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        ### YOUR CODE HERE


def train_head(nn_weights_path: str, examples_per_class: int = 20) -> torch.nn.Module:
    """
    Функция для обучения kNN-головы классификатора.

    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE


if __name__ == "__main__":
    # The following code won't be run in the test system, but you can run it
    # on your local computer with `python -m rare_traffic_sign_solution`.

    # Feel free to put here any code that you used while
    # debugging, training and testing your solution.


    model = train_simple_classifier()
    # model = CustomNetwork()
    # model.load_state_dict(torch.load('simple_model.pt', map_location='cpu', weights_only=True))
    # test_folder = 'smalltest'

    # results = apply_classifier(model=model, test_folder=test_folder, path_to_classes_json='classes.json')
    # print(results)