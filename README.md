### Abstract
This project was developt for solving problem of glume pubescence of spikelets. The acquired model predicts pubescence feature of spikelets.
There is also segmentation model for segmenting spikelets of other objects. Each spikelets must be fed to CNN separately.
If you have several spikelets on an image you may use, for instance OpenCV method

> **countours, hierarchy = cv2.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)**

### How to use this project
There are weights of two models:
  - efficientnet_b2 (for segmentation)
  - efficientnet_b1 (for classification)

---

You should follow throught this steps to obtain prediction of pubescence of spikelet

1) Use efficientnet_b2 for obtaining 3-channel mask (color-checker, arista, spikelet)
> Note, that I don't have script for efficientnet_b2, only trained weight
2) Extract spikelets from image mutiplied by mask as bounding boxes
3) Apply efficientnet_b1 for prediction of pubescence

*N.B. The project was supported by Institute of Cytology and Genetics of SB RAS*
