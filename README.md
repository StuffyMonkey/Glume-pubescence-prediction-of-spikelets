## Abstract
This project was developed for solving problem of prediction the feature of glume pubescence of spikelets of wheat. The acquired classification model predicts pubescence feature of spikelets.
There is also segmentation model for segmenting spikelets of other objects. 

Each spikelets must be fed to CNN separately. If you have several spikelets on an image you may use, for instance, OpenCV method:

```python
countours, hierarchy = cv2.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
```
---

## Methods preview
Images for training and test have the same image format depicted below:

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}_pubesc.jpg" width="480" height="720">

You should follow throught these steps to obtain prediction of pubescence of spikelet:
1) Use segmentation model for obtaining 3-channel mask (color-checker, arista, spikelet)
This is an example of mask:

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}.png" width="480" height="720">

2) Extract spikelets from image mutiplied by mask as bounding boxes

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}.jpg" width="240" height="512">

3) Apply classification model for prediction of pubescence (model.py) <br/>
using following command:
```python -inp <full_path_to_iamge> -out <full_path_to_save_txt_file>```

- Results will be saved into txt file in format <full_path_to_image> <probability_prediction> <br/>
N.B. This requires the presence of binary file of segmentation model

### Best acuqired model properties
**accuracy:** 0.85  <br/>
**precision:** 0.8  <br/>
**roc_auc:** 0.9

Here is an example of class activation map of the model

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/heatmap_example.png" width="512" height="512">

---
<details open>
<summary>Files information</summary>
  
  - config.py - config file with constants, regular expression patterns and other essentials
  - dataset.py - python file with custom Dataset class. It's essential file and all methods are sharped for the usage of that class with specific annotation.
  - model.py - python file with single class named Model. It's used to train, evaluate and get predictions of model.
  - preprocessing.py - python file with class DataPreprocessor and other useful functions to work with raw annotation, images and so on.
  - training_config.py - config file with training information and augmentation. You may use script to make grid-search of params changing dictionaries of params in that file.
  - utils.py - python file with two essential ReadOnly classes and some functions, that idiologically not appropriate for files above.
 
</details>

<details>
<summary>Executable file guide</summary>
  
### Developer part 
  Follow steps bellow to create your own binary model.
  1) Install pyinstaller into your virtual environment and other requirements using 
  ```
  pip install -r requirements.txt
  ```
  2) In the folder https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/tree/main/bin_skeleton/
  there are file model.py with preprocessing and loading segmentation and classification models.
  That file we will convert into binary file with command.
  ```
  pyinstaller -F --hidden-import="sklearn.utils._typedefs" --hidden-import="sklearn.neighbors.typedefs" --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils"  model.py
  ```
  N.B. Here we use some hooks to collect manually all required modules, that weren't included by pyinstaller
  3) In the folder dist/ of your current directory will be executable binary file of model.
  
### User part
  1) Check options
  ```
  ./home/jupyter-n.artemenko/infer/spikelet_pubescence/random_model_exe --help
  ```
  2) Run model.
  ```
  ./home/jupyter-n.artemenko/infer/spikelet_pubescence/random_model_exe -inp full_path_to_image> -out <path_to_save_txt_file_with_predictions>
  ```
  by default prediction will be saved in your current dirrectory int txt file **predictions.txt** (if -out parameter wasn't passed)
  
---

</details>
*Co-authored by @rostepifanov (pretrained segmentation model)*

*N.B. The project was supported by Institute of Cytology and Genetics of SB RAS*
