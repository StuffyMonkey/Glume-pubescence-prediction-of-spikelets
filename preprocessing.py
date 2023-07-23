from config import PROJECT_DIR, DATA_DIR, PATTERNS
import numpy as np
import cv2
import re
import os
import time

def make_bbox(ploid : str, species : str, img_type : str, img_name : str, overwrite=False, dilation=False):
    # if we don't want overwrite files
    if not overwrite and os.path.exists((f'''{save_dir}/{ploid}/{species}/{img_type}
                                        /{img_name}''').replace('.jpg','.png').replace('.JPG','.png')):
        return
    
    # Create directory with species
    if not os.path.exists(f'{save_dir}/{ploid}/{species}/{img_type}'):
        species = species.replace(' ', '\ ')
        os.system(f'mkdir {save_dir}/{ploid}')
        os.system(f'mkdir {save_dir}/{ploid}/{species}')
        os.system(f'mkdir {save_dir}/{ploid}/{species}/{img_type}')
            
            
    # Open files, mask and get product o mask and image
    raw_img = cv2.imread(dirpath + '/' + img_name)
    nm = (f'{mask_dir}/{ploid}/{img_type}/{img_name}').replace('.jpg','.png').replace('.JPG','.png')
    mask = cv2.imread(nm)
    
    if mask is None:
        logs.write(f'There is no mask for image: {dirpath}/{img_name}\n')
        return
    if mask.shape != raw_img.shape:
        logs.write(f'Dimension problem: {dirpath}/{img_name}\n')
        return
                
    # acquire 3-channel mask for spike only
    tmp = np.moveaxis(mask, 2, 0)
    spike_body_mask = tmp[1]
    # applying dilation to expand boundaries of mask
    if dilation:
        kernel = np.ones((11, 11), np.uint8)
        spike_body_mask = cv2.dilate(spike_body_mask, kernel, iterations=1)
    tmp[0] = spike_body_mask
    tmp[2] = spike_body_mask
    tmp[1] = spike_body_mask
    tmp = np.moveaxis(mask, 2, 2)
    
    # product of mask and image
    img = cv2.bitwise_and(raw_img, tmp)
            
    # Make crop and unsqueeze to square binary-power form
    original = img.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # List with dimensions of contours
    areas = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        areas.append((x, y, w, h, w*h))
                    
    areas = sorted(areas, key=lambda tup: tup[4], reverse=True)
    # we need second largest bounding box, first is the overall image
    x,y,w,h,_ = areas[0]
                    
    # get rectangle with corner x,y and the approproate w,h
    cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    img = original[y:y+h, x:x+w] 
                
    # save bounding box
    species = species.replace('\ ', ' ')
    cv2.imwrite((f'{save_dir}/{ploid}/{species}/{img_type}/{img_name}').replace('.jpg', '.png').replace('.JPG', '.png'), img)
    
def make_all_bboxes(overwrite=False, dilation=False):
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
    for fn in filenames:
        # Get data of ploidness, img_name, img_type
        ploid = re.search(PATTERNS['ploid'], dirpath)
        ploid = ploid.group() if ploid else None
        species = re.search(PATTERNS['species'], dirpath)
        species = species.group() if species else None
        img_type = 'pin'
        img_name = re.search(PATTERNS['pin'], fn)
        if not img_name:
            img_type = 'table'
            img_name = re.search(PATTERNS['table'], fn)
        if img_name:
            img_name = img_name.group()
            make_crop(ploid, species, img_type, img_name, overwrite, dilation)