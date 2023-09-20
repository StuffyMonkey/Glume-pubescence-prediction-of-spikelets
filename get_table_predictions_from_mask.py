import os
import cv2
import torch
import warnings
import typing as tp
import numpy as np
import pandas as pd
from sys import argv

from preprocessing import get_bbox
from model import Model
from training_config import augmentations

warnings.simplefilter("ignore")


def get_all_png_files(path: str) -> tp.Iterator[str]:
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith(('png', 'PNG')):
                continue
            yield os.path.join(dirname, name)


def main():
    """
    This script returns a table with predictions based on images and their masks.
    Usage example: python3 get_table_predictions_from_mask.py model_path in_path out_path
    """
    # in_path is the path to directory with images and masks (.png -- masks, .jpg -- images)
    # model_path is the path to the model weights
    # out_path is the path where the table with the results will be saved
    _, model_path, in_path, out_path = argv
    model = Model('efficientnet_b1', os.getcwd())
    model.model.load_state_dict(torch.load(model_path))
    name_lst = list()
    img_lst = list()
    for path in get_all_png_files(in_path):
        mask = cv2.imread(path)
        img = cv2.imread(path.split('.')[0] + '.jpg')
        bbox = get_bbox(img, mask, dilation=False)
        preprocessed_img = augmentations['inference_transforms'](image=bbox)['image']
        preprocessed_img = np.moveaxis(preprocessed_img, -1, 0)
        name_lst.append(path)
        img_lst.append(preprocessed_img)
    imgs = np.array(img_lst)
    labels = model.predict(imgs)
    pd.DataFrame(list(zip(name_lst, labels)),
                 columns=['Path', 'Glume pubescence']).to_csv(os.path.join(out_path, 'glume_pubescence.csv'), index=False)

   
if __name__ == '__main__':
    main()
            
