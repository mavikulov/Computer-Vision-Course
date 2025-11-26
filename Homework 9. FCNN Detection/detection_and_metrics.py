from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np


# ============================== 1 Classifier model ============================
def get_cls_model():
    """
    :return: nn model for classification
    """
    # your code here \/
    input_shape = (1, 40, 100) # (n_channels, n_rows, n_cols)
    classification_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(128),
        
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(128),
        
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
        nn.ReLU(),
        nn.BatchNorm2d(256),
        
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        
        nn.Flatten(),
        nn.Linear(256 * 5 * 13, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    
    return classification_model
    # your code here /\


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_save_path: str = "classifier_model.pt"


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    config = TrainingConfig()
    model = get_cls_model()
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_iterations = X.shape[0] // config.batch_size + 1

    model.train()

    for epoch in range(config.epochs):
        running_loss = 0
        num_batches = 0
        for i in range(num_iterations):
            num_batches += 1
            optimizer.zero_grad()
            X_batch = X[i * config.batch_size: (i + 1) * config.batch_size]
            y_batch = y[i * config.batch_size: (i + 1) * config.batch_size]
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            outputs = model(X_batch)
            outputs = outputs.to(config.device)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_per_epoch = running_loss / num_batches
        # print(f"Epoch: {epoch} | loss = {loss_per_epoch}")

    # torch.save(model.state_dict(), config.model_save_path)
    return model.to('cpu')
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    detection_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(128),
        
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(128),
        
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
        nn.ReLU(),
        nn.BatchNorm2d(256),
        
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        
        # Замена Flatten + Linear
        nn.Conv2d(256, 512, kernel_size=(5, 13)),
        nn.ReLU(),
        nn.Dropout(0.5),

        # Замена Linear
        nn.Conv2d(512, 128, kernel_size=(1, 1)),
        nn.ReLU(),
        nn.Dropout(0.3),

        # Замена последнего Linear
        nn.Conv2d(128, 2, kernel_size=(1, 1))
    )

    cls_model.eval()
    detection_model.eval()

    # Начиная с 24 слоя происходят замены flatten + линейных слоев на сверточные, делаю reshape весов для них
    common_layers_num = 24
    for i in range(common_layers_num):
        detection_model[i] = deepcopy(cls_model[i])
    
    with torch.no_grad():
        detection_model[24].weight.copy_(
            cls_model[25].weight.reshape(512, 256, 5, 13)
        )
        detection_model[24].bias.copy_(cls_model[25].bias)

        detection_model[27].weight.copy_(
            cls_model[28].weight.reshape(128, 512, 1, 1)
        )
        detection_model[27].bias.copy_(cls_model[28].bias)

        detection_model[30].weight.copy_(
            cls_model[31].weight.reshape(2, 128, 1, 1)
        )
        detection_model[30].bias.copy_(cls_model[31].bias)

    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    result = {}
    threshold = 0.9
    detection_model.eval()

    for filename in dictionary_of_images:
        detections = []
        image = dictionary_of_images[filename]
        img_shape = image.shape
        image = torch.FloatTensor(
            np.pad(image, ((0, 220 - img_shape[0]), (0, 370 - img_shape[1])))
        )

        image = image.reshape(1, 1, 220, 370)
        outputs = detection_model(image).detach()[0][1]
        pred_shape = (img_shape[0] // 8 - 5, img_shape[1] // 8 - 5)
        outputs = outputs[:pred_shape[0], :pred_shape[1]]

        for w in range(outputs.shape[0]):
            for h in range(outputs.shape[1]):
                if outputs[w, h].item() > threshold:
                    detections.append([w * 8, h * 8, 40, 100, outputs[w, h].item()])

        result[filename] = detections
    
    return result
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row1, col1, n_rows1, n_cols1 = first_bbox[:4]
    row2, col2, n_rows2, n_cols2 = second_bbox[:4]

    x1_min = col1
    y1_min = row1
    x1_max = col1 + n_cols1
    y1_max = row1 + n_rows1
    x2_min = col2
    y2_min = row2
    x2_max = col2 + n_cols2
    y2_max = row2 + n_rows2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    area1 = n_cols1 * n_rows1
    area2 = n_cols2 * n_rows2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    iou_thr = 0.5
    true_positives = []
    all_detections = []
    total_ground_truths = 0

    for image_name in pred_bboxes:
        image_ground_truths = gt_bboxes[image_name].copy()
        total_ground_truths += len(image_ground_truths)
        sorted_detections = sorted(pred_bboxes[image_name], key=lambda x: x[4], reverse=True)
        
        for detection in sorted_detections:
            detection_coords = detection[:4]
            confidence = detection[4]
            max_overlap = 0
            best_match = None
            
            for gt_box in image_ground_truths:
                iou_score = calc_iou(detection_coords, gt_box)
                if iou_score > max_overlap:
                    max_overlap = iou_score
                    best_match = gt_box
            
            if max_overlap >= iou_thr:
                image_ground_truths.remove(best_match)
                true_positives.append(confidence)
            
            all_detections.append(confidence)
    
    all_detections.sort(reverse=True)
    true_positives.sort(reverse=True)
    pr_points = []
    pr_points.append((0.0, 1.0, 1.0))
    tp_index = 0

    for i, conf_threshold in enumerate(all_detections):
        if i < len(all_detections) - 1 and all_detections[i] == all_detections[i + 1]:
            continue
            
        while tp_index < len(true_positives) and true_positives[tp_index] >= conf_threshold:
            tp_index += 1
            
        recall = tp_index / total_ground_truths if total_ground_truths > 0 else 0
        precision = tp_index / (i + 1) if (i + 1) > 0 else 0
        pr_points.append((recall, precision, conf_threshold))
    
    auc = 0.0
    final_recall = len(true_positives) / total_ground_truths if total_ground_truths > 0 else 0
    final_precision = len(true_positives) / len(all_detections) if len(all_detections) > 0 else 0
    pr_points.append((final_recall, final_precision, 0.0))
    
    for i in range(len(pr_points) - 1):
        recall_diff = pr_points[i + 1][0] - pr_points[i][0]
        precision_avg = (pr_points[i + 1][1] + pr_points[i][1]) / 2
        auc += recall_diff * precision_avg
    
    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.03):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    result = {}
    for image_name, detections in detections_dictionary.items():
        result[image_name] = []
        sorted_boxes = sorted(detections, key=lambda x: x[4], reverse=True)

        for candidate_box in sorted_boxes:
            for box in result[image_name]:
                if calc_iou(box, candidate_box) >= iou_thr:
                    break
            else:
                result[image_name].append(candidate_box) 

    return result
    # your code here /\
