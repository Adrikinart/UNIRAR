import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


def open_image(file, channels=3):
    if channels == 1:
        img = cv2.imread(file, 0)
    elif channels == 3:
        img = cv2.imread(file)

    shape_r = 288
    shape_c = 384
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    padding_info = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0 , 'w': shape_c, 'h': shape_r}

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        left_padding = (img_padded.shape[1] - new_cols) // 2
        img_padded[:, left_padding:left_padding + new_cols] = img
        padding_info['left'] = left_padding
        padding_info['right'] = img_padded.shape[1] - new_cols - left_padding
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        top_padding = (img_padded.shape[0] - new_rows) // 2
        img_padded[top_padding:top_padding + new_rows, :] = img
        padding_info['top'] = top_padding
        padding_info['bottom'] = img_padded.shape[0] - new_rows - top_padding

    img = np.array(img_padded) / 255.
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    img = torch.from_numpy(img).type(torch.FloatTensor)

    return img, padding_info


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img

