##Vehicle Detection Project

The goal of this project is to detect cars in a clip of video recorded on the highway. The core idea behind it is to use a machine learning algorithm to train an image classifier which can tell whether an image contains a car or not. There are many details included, for example, a classifier can only consume values as input, so we have to extract features out of an image that can represent the characteristic of the image. In order to detect cars in different sizes and posistions in the image (video frame), we will need a way to scan the image with `sliding window`. Sliding window can affect the precision with the trade off of performance.

---

*The goals / steps of this project are the following:*

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run designed pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/color_space.png
[image2]: ./output_images/orient.jpg
[image3]: ./output_images/pix_per_cell.jpg
[image4]: ./output_images/cell_per_block.jpg
[image5]: ./output_images/heatmap.png
[video1]: ./project_video_output.mp4

##Data sources
Labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip).

Labeled data for [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Extract HOG features

HOG feature extraction is implemented in `skimage` library. It can simply be used with mainly the following parameters:
```
- orientations : The number of orientation bins.
- pixels_per_cell : Size of a cell.
- cells_per_block : Number of cells in each block.
```

####2. Choose HOG parameters

*Color space and channels*: According to the experiment YUV and YCrCb performs better than other color spaces in all channels 
![alt text][image1]

*Orient*: Then numbers smaller than 8 seem too simple and the numbers larger than 9 seem to be not much different.
![alt text][image2]

*Pixels per cell*: 8 is a great size for 64x64 training data.
![alt text][image3]

*Cells per block*: Don't see difference between the results. Used the value from the lecture.
![alt text][image4]


####3. Train the classifier

Linear SVM classifier is used because it is relatively fast and performs well enough for the project. 
Based on the experiment, using all features learned from the course has the best accuracy.
```
Binned color only: Test Accuracy of SVC =  0.9043
Color histogram only: Test Accuracy of SVC =  0.8964
HOG only: Test Accuracy of SVC =  0.9794
All features: Test Accuracy of SVC =  0.9893
```

###Sliding Window Search

####1. Size and position

In order to reduce the amount of windows searched, I restrict the searching area to be in the range between 400 and the bottom of the image. I can also search only on the lane area by using the result of project 4, but I decided not to do it due to the time limit. I have chosen three different sizes of windows, 64, 96, 128. The 64 size is only run between 400 to 600, while 96 and 128 are run between 400 to the bottom of the image. They work well but have very poor performance, so I cheat. I finally decide to use only size 96 because it works well on the project video for detecting the cars close to us.

![alt text][image5]

---

### Video Implementation

####1. Result
Here's a [link to my video result](./project_video.mp4)


####2. False positive elimination

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Using heat map alone to remove the false positive isn't good enough. The threshold either removes real positive or keeps some of the false positive. In this case, I used the smoothing technique from project 4. However, instead of averaging the values, I add up all the heat map in the lastest several frame (I chose 10 in the final result) and increase the threshold to 8. This way, I can remove almost all the false positive.

---

###Discussion

1. This project relies on the classifier heavily. However, Udacity prepared good training data which allows us to simply focus on the feature extraction. 
2. The time performance is a big issue in this project. No matter how I tweak the parameters, I couldn't reach the point that the detection is accurate and the run-time is low enough.
