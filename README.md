This is the winning solution to kaggle Ultramnist. This solution got 99.109 and Ranked 1st on leaderboard [https://www.kaggle.com/competitions/ultra-mnist]. 
Code is available here along with trained models. You can use these to regenerate the results. Instructions to train from scratch are also attached.  https://github.com/4g/umnist

## Idea
- Generate synthetic data that looks like the competition data. 
- Use a small object detector like yolov5s and reclassify all labels using a larger network (EffV2B1)

Input images are large. Any network consuming these directly will be slow. Yolo small gets good box-obj accuracy but low class accuracy. 

Objects in image are not complex. Even at a resolution of 128x128 these can be classified with > 99.9% accuracy using a sufficiently large network. So I train a classifier on digits cropped from synthetic training set and reclassify yolo results with this high accuracy classifier. Yolo alone gives > 95%, adding a classifier gives > 98%. Rest of accuracy comes from post processing magic.  

## Data generation
#### Detection
- 2560x2560 input images. Smallest digit size is 6x6. 
- Generate checkerboards with boxes, circles and triangles. Apply augmentations like perspective, rotation, zoom to make it look more like competition dataset. 
- Generate locations where digits should be put. Make sure digits don't overlap. 
- Ensure all 70000 instances from mnist are covered. 
- Uniformly distribute sizes of digits. 
- Generate empty images with no digits to reduce background confusion. 
- Train 34000 with 4000 empty images : val 2000 with 0 empty images.  

#### Classification
- 128x128 images. 
- Digit crops from synthetic data. 
- Add raw mnist digits 
- Train : 230,000 (160k synthetic + 70k raw mnist)

## Training
#### Detector
- Yolov5 small with batch size 2 (max that is supported at this resolution on my gpu)
- light augmentation, because harsh augmentations take digits too much out of the image : rotation, scale, translate, hsv, mosaic. 
- SGD 100 epochs. Stop when the object loss reaches .008. Cls loss doesn't matter. Recall should be high as detector needs to retrieve all instances. delete them later if the classifier has low confidence. 

#### Classifier
- Effnetv2b1 with 128x128x3 input. Use pretrained imagenet weights for faster convergence. 
- augmentations : Shift scale, perspective, invert img, blur, hsv, brightness/contrast. Keep augmentations light as competition data doesn't look augmented. 
- Reduce lr every 50 epochs (3e-4, 1e-4, 3e-5, 1e-5). Train for 200. 

## Inference 
#### Detection
- 2560x2560
- Try to increase recall of yolo detector. 
- iou_threshold = 0.1. Digits are not intersecting, so any intersecting outputs are wrong 
- conf_threshold = 0.1 . To increase recall. 
- class agnostic nms
- Maximum 5 digits allowed per image, max_det = 5
- To measure the accuracy, use competition training data. 
- > 95%

#### Classification
- Crop digits detected by yolo. Some of them have skewed aspect ratio, pick the larger side and crop a square. Classifier is trained with square images. Changing the aspect ratio reduces its accuracy. 
- TTA: Invertimg, average the predicted probabilties
- > 98%

#### Post processing
- Some of the large digits have artifacts, which get detected as other digits. IOU threshold cannot remove them because these are very small as compared to the larger digit. Remove any digit that lies completely inside another
- Some of the background is also classified as digits. But the classifier has low confidence for these. Remove any digits where classifier prob < 0.45
- > 99.1% 

## Debugging
Since the competition training data is not seen by models, it can be used for validation.  Looking at images that are getting incorrect sum, gives a good idea of augmentations and filters to add.  

## Tidbits
- Accuracy on competition train set closely matches the test accuracy
- Yolo took 3 days to train due to large image size
- This solution doesn't help the original problem which authors had proposed of processing large images via networks in any way
- In the beginning I was working towards innovative track using a pretrained unet + blazeblock regressor but then the dataset changed.

 Thanks Namtran. Your continuous uploads were a big encouragement. 
