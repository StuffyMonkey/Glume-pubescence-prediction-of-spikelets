import numpy as np
import cv2
import albumentations as A
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import os
import click
import re


def get_mask(input_path):
    init_dir = os.getcwd()
    inp = input_path.replace(' ', '\\ ')
    out = f'{os.getcwd()}'.replace(' ', '\\ ')
    os.chdir('/home/rostepifanov/import/bin/segmentation/')
    seg_script = f'./infer -bone efficientnet-b2 -mn model_efficientnet-b2.bin --cuda --verbose -bs 32 -ip {inp} -op {out}/'
    mask = []
    try:
        os.system(seg_script)
        img_patt = r'[-\w \d \s _{}]*.(jpg|png|PNG|JPG)'
        img_name = re.search(img_patt, inp).group().replace('jpg','png').replace('JPG', 'png')
        mask = cv2.imread(f'{out}/{img_name}')
        mask[:, :, 0] = mask[:, :, 1]
        mask[:, :, 2] = mask[:, :, 1]
        cv2.imwrite(f'{out}/{img_name}', mask)
    except FileNotFoundError:
        print("Image wasn't found or save dir doesn't exist. Check you script, please!")
    os.chdir(init_dir)


# Work fine
def get_crop(raw_img, mask):
    # get crop 512x512
    img = cv2.bitwise_and(raw_img, mask)  # image with black background and spikelet
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours of image. We need second size contour
    cntrs = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    areas = []
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        areas.append((x, y, w, h, w * h))
    areas = sorted(areas, key=lambda tup: tup[4])
    x, y, w, h, _ = areas[-2]
    img = img[y:y + h, x:x + w]
    return img


# Work fine
def preprocess(crop, size=480):
    to_tensor = transforms.ToTensor()
    crop_norm = A.Compose([
        A.CenterCrop(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
    # overlay dark square on crop
    y, x, z = crop.shape
    max_x, min_x = max(x, size), min(x, size)
    max_y, min_y = max(y, size), min(y, size)
    square = np.zeros((max_y, max_x, 3))

    # Getting the centering position
    ax, ay = round((max_x-x)/2), round((max_y-y)/2)

    # Putting the 'image' in a centering position
    square[ay:ay+y, ax:ax+x] = crop

    # Getting result as normalized tensor
    res = crop_norm(image=square)['image']
    res = to_tensor(res)
    return res


def predict(img):
    # device and grad computations
    torch.set_grad_enabled(False) 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # import model and load weights
    model = torchvision.models.efficientnet_b1(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1, inplace=True),
        nn.Linear(in_features=1280, out_features=1, bias=True),
        nn.Sigmoid())
    model = model.to(device)
    if device != 'cpu':
        model.load_state_dict(torch.load(f'{os.getcwd()}/efficientnet_b1.bin'))
    else:
        model.load_state_dict(torch.load(f'{os.getcwd()}/efficientnet_b1.bin', map_location='cpu'))

    # make prediction
    data = img.unsqueeze(0)
    data = data.to(device)
    prediction = model(data).squeeze(0).cpu().detach().numpy()[0]
    return prediction


@click.command()
@click.option('--inp', '-inp', help='Full path to image. Can process only png and jpg files!')
@click.option('--out', '-out', help='Path to output txt file with images and predictions')
def __main__(inp, out):
    img = []
    try:
        img = cv2.imread(inp)
    except FileNotFoundError:
        print("Check your input path and existence of image")
    get_mask(inp)
    
    img_patt = r'[-\w \d \s _{}]*.(jpg|png|PNG|JPG)'
    img_name = re.search(img_patt, inp).group().replace('jpg','png').replace('JPG', 'png')
    mask = cv2.imread(f'{os.getcwd()}/{img_name}')
    
    img_tp = preprocess(get_crop(img, mask))
    prediction = predict(img_tp)
    os.remove(f'{os.getcwd()}/{img_name}')
    with open(f'{out}labels.txt', 'a') as out:
        out.write(f'{inp} {prediction:.3f}\n')


if __name__ == '__main__':
    __main__()
