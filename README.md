### Abstract
This project was developt for solving problem of glume pubescence of spikelets. The acquired model predicts pubescence feature of spikelets.
There is also segmentation model for segmenting spikelets of other objects. Each spikelets must be fed to CNN separately.
If you have several spikelets on an image you may use, for instance, OpenCV method

> **countours, hierarchy = cv2.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)**

### Instructions

---
Images for training and test looked like:

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}_pubesc.jpg" width="480" height="720">

You should follow throught these steps to obtain prediction of pubescence of spikelet:
1) Use segmentation model for obtaining 3-channel mask (color-checker, arista, spikelet)
This is an example of mask:

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}.png" width="480" height="720">

2) Extract spikelets from image mutiplied by mask as bounding boxes

<img src="https://github.com/StuffyMonkey/Glume-pubescence-prediction-of-spikelets/blob/main/Data/28n4_3_{V-21}.jpg" width="240" height="512">

3) Apply classification model for prediction of pubescence (model.py)

Using following command: python -inp <full_path_to_iamge> -out <full_path_to_save_txt_file>

Results will be saved into txt file in format <full_path_to_image> <probability_prediction>

N.B. This requires the presence of binary file of segmentation model

*Co-authored by @rostepifanov (pretrained segmentation model)*

*N.B. The project was supported by Institute of Cytology and Genetics of SB RAS*

*P.S. If you wish to get weights for models, you may write me on email*
