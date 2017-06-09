**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/heat_map.png
[image71]: ./examples/heat_map1.png
[image72]: ./examples/heat_map3.png
[image8]: ./examples/all_together.png
[image9]: ./examples/window_sizes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

#### 1. Creating the dataset
The training dataset provided for this project by Udacity as set of png images ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


#### 2. Extracting the HOG features
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 3. Training classifier based on SVM, HOG and color features
After extracting HOG and color features from the `cars` and `notcars` I test the accuracy of the SVC by comparing the predictions on labeled X_train data. 
Test Accuracy of HOG based SVC and RGB color space is 96.79%. I played with different setting and color spaces and found the following params 
```python
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64   # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
and color space `YCrCb` show better test accuracy in my case: 99.07%

### 3. Sliding Window Search
The `slide_window` function takes in an image, start and stop positions, window size and overlap fraction and returns a list of bounding boxes for the search windows, which will then be passed to draw boxes. Below is an illustration of the slide_window function with adjusted y_start_stop values [400, 656].

![alt text][image3]

### 4. Selecting scale and y-axis range.
The sliding window is the cpu consuming operation therefore need to specify area and quantity of windows which really help to detect cars. So I started search using following y-axis range:
380 - 550, 380 - 600, 380 - 656, 380 - 700 and scale factors: 1.0, 1.5, 2.0, 2.5
So based on images below I decided keep first 3 ranges with corresponding scale factors.   
![alt text][image9]
### 4. Adding Heatmaps and Bounding Boxes
The `add_heat` function creates a map of positive "car" results found in an image by adding all the pixels found inside of search boxes. More boxes means more "hot" pixels. The `apply_threshold` function defines how many search boxes have to overlap for the pixels to be counted as "hot", as a result the "false-positve" search boxes can be discarded. The `draw_labeled_bboxes` function takes in the "hot" pixel values from the image and converts them into labels then draws bounding boxes around those labels. Below is an example of these functions at work.
![alt text][image7]
![alt text][image71]
![alt text][image72]

### 5. Pipeline
I developed function called `pipline` which includes all necessary actions for detecting cars on an image. This function will be used for processing a video file. 
![alt text][image4]

After that I imported from my previous project algorithm for processing lane lines and combine together with current project. This function `pipeline_with_lane_finding` could detect cars and lane lines for specified image.

![alt text][image8]

---

### 6. Video Implementation
Here's a [link to my video result](./video_out/project_video_output.mp4) [youtube](https://www.youtube.com/watch?v=8TDtqQk8oBY)


---

### Discussion
It was quite interesting project with various creative things. Also, the code that was used for studying help so much me. The pipeline not fast as required for real-time detection. Some false positives still remain in the video file even with applying heatmap filtering. However, there is ample room for improvement.
So the sliding window algorithm could be optimized a bit with splitting on several ROI. The evaluation of feature vectors is currently done sequentially, but could be parallelized. 
Actually today is more popular utilize CNN for object detecting and it show very nice results and performance as well. I checked several demos from [YOLO](https://pjreddie.com/darknet/yolo/) and [ADNET](https://sites.google.com/view/cvpr2017-adnet) and was very impressed!
