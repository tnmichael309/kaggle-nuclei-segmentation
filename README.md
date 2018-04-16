# kaggle-nuclei-segmentation

# Modified from: https://github.com/neptune-ml/open-solution-data-science-bowl-2018/tree/mask_rcnn_notebook 

# Software Environment:
  * Python:
  * Tensorflow: 1.7.0 
  * Pandas: 0.22.0
  * Numpy: 1.14.2
  * tqdm: 4.20.0
  * keras: 2.1.5
  * scipy: 1.0.1
  * skimage: 0.13.1
  * imgaug: 0.2.5

#	Hardware Environment:
  * Google Cloud Platform:
  * CPU: 8 core, 32GB RAM.
  * GPU: Nvidia K80, 11G.
  * 200G HDD.

# How to run & repeat submission:
## Basic Downloads: (Link: https://goo.gl/9Xjfuv)
  * stage 1 train\test data. (For model training repeat)
  * External data link (already in the provided stage 1 train data).
  * Link: https://www.kaggle.com/voglinio/bowl2018-external 
  * stage1_metadata.csv. (For model training repeat)
  * stage1_background_foreground_classes.csv. (For model training repeat)
  * Model trained weights. (For submissions repeat) 
  * stage2_metadata.csv. (For submissions repeat)
  * stage2_backgroud_foreground_classes.csv. (For submissions repeat)
  * Stage 2 test data downloads: https://www.kaggle.com/c/data-science-bowl-2018/data 
## Training:
  * Run the following jupyter notebooks for two models: (Using different pre-processing)
    * training.ipynb
  * Modifies:
    * 'paths' in the notebook to save the model weights trained.
## Generate Submissions:
  * Run "post process-stage2.ipynb" for two models’ predictions:
  * Modifies:
    * 'paths' in the notebook to the model weights to be used in inference.
    * TEST_PATH to stage 2 test data
    * META_DATA_PATH to stage2_metadata.csv
    * BG_DATA_PATH to stage2_backgroud_foreground_classes.csv
  * Run Ensemble-stage2.ipynb to combine two models’ predictions.

# Solution Description:
  * Mask R-CNN: Follow https://github.com/matterport/Mask_RCNN
  * Post-processing (Most time spent here):
    * Soft-nms
    * Flexible soft-nms
    * Parallel Processing to speed up
  * Augmentation.
    * np.fliplr, np.flpud, np.rot90
    * Random crop for external dataset.
      * Too many instances in one original image.
  * Testing-Time Augmentation.
    * Rotation, even np.rot90 hurt performance. I used np.fliplr\ud only.
  * Background-Foreground Processing:
    * Training different model based on different background-foreground type.
    * Also applied with different preprocessing.
  * IOU Calculation.
    * Pixel level instead of using box.
    * For situation when nuclei (say A) is long and inclined, s.t. the bounding box has very high 'box' iou with some other smaller nuclei around nuclei A.
  * Implementation: (Calculating pixel IOU between A and B):
    * Binarize masks of A and B.
    * Pixel >= .5 to be 1, else 0
    * Returns (np.logical_and(A, B).sum() / min(A.sum(), B.sum())
    * This improves a lot, since the following case performs much better:
    * Small nuclei totally overlapped with large nuclei
  * Misc.
    * Compile fixed labels\masks.
    * Hyper-parameter tuning: detection min confidence, max detected instances…
    * Filter invalid\small boxes out of training phase.
    * Train longer. (20 -> 90 epochs).
    * Schedule learning rate.
    * Compile external data set. (H&E stained dataset)
    * Tried deform conv layer.
      * But doesn’t work out well.
