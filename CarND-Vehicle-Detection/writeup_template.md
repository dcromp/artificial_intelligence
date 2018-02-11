## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./resources/notcar_car.png
[image2]: ./resources/hog_features.png
[image3]: ./resources/windows.png
[image4]: ./resources/false_positives.png
[image5]: ./resources/pipeline.png
[image6]: ./resources/heatmap.png
[image7]: ./resources/labels.png
[video1]: ./resources/result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First I used the library glob to load all the training image file paths into two lists. One list for car images and another for not car images. An random example image from each list is shown below.

![alt text][image1]

Before the hog features are calculated I first change the convert the image colour from RGB to YCrCb. I choose YCrCb as it seemed to give a higher accuracy when it came to building the classifier.

Once colour converted, Hog features are then extracted from  the training images using the following function, which can be found in cell 5 in the notebook. 

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

The above function relies on the library skimage, which has a set of functions for extracting features from images, including HOG features. This functionality can be accessed by the following import statement.

```
from skimage.features import hog
```

The result of the hog features compared to the original input image can be seen below. It is interesting the note with car images the hog features seem to show almost a square like pattern, not seen in the non car images.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

To decide what HOG parameters to use I mostly relied on trial and error. I started with an orientation of 9 as this was suggested in some seminal papers as being the optimum amount, or at least increasing this number would make little difference. I also saw that 2 cells with 8x8 pixels blocks would roughly cover the cars back lights, in a 64x64 image. Based on the same papers this seemed like a good approach as these cell blocks would cover important features of what makes the car different from non cars.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First I used a function called extract_features which takes all the training data, converts the colour and extracts the HOG features. Additionally, I used spatial binning, which effectively downsizes the 64x64 image to 32x32 and colour histogram binning using 64 bins. Like the HOG features using 32x32 for spatial and 64 bins for colour was chosen mostly by trial and error. Test data was then separated out from the rest of the training data to measure the accuracy of the model after the model parameters had been tuned.

Initially I decided to use a few different models with different parameters to find the most optimum choice in approach. To do this I exploited scikit learn gridsearch functionality defined by the code below.

```
svr = SVC()
svr_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr_clf = GridSearchCV(svr, svr_parameters, n_jobs=-1, verbose=1)

rfc = RandomForestClassifier()
rfc_parameters = {'n_estimators':[120,300,500], 'max_depth':[5, 8, 30, None],
                  'min_samples_split':[2, 5, 10, 15], 'min_samples_leaf':[1,2,5,10]}
rfc_clf = GridSearchCV(rfc, rfc_parameters, n_jobs=-1, verbose=1)

gnb = GaussianNB()  # No parameters so no need for grid search
```

However, this would take almost 2 hours to search through all the parameters, so I decided to stick to SVC and ignored the other models, which seemed to be less accurate than SVC.

The actual training the model was done in cell 10 using only SVC with pre-set parameters of C = 10 and kernel = rbf for the sake of speed. These parameters where chosen after running the following command in a previous experiment.

```
svr_clf.best_params_
```

Once the model was trained it scored around 0.9961 on the test data. As a side note I would get around 0.94 accuracy when I tried to use only a linear SVR model. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows was first implemented in cell 11 as the function slide_window and search_window. Slide_window would take an image and save a list of the all the possible 64x64 windows that existed in that image under the defined conditions. Search_window would extract the HOG, spatial and colour features from these windows and return if the windows contained a car or not using the classifier detailed above. Examples positive detections, scaling the image by 2, shown below. Note how at this scale the detector struggled to find the white car. 

![alt text][image3]

For comparison here are the same images using a 94% accurate linear SVR model. See how the number of false positives is in far higher, especially around regions on contrast such as shadow and the tree sky boarder. 

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to optimize the pipeline the above functions were refactored so the HOG features are calculated once per image, rather than every time for each window. This is implemented by the nearly identical functions find_cars and find_bbox, the former returns an image with the positive windows drawn on the image and later returns only the bounding boxes of the positive windows. These functions also include a calculation of the spatial and colour histogram features.

The pipeline used in the final version can be found in code block 25 and in the code snippet below. I ended up using 3 different image scales as this seemed the most reliable way to find the white car in every frame. I also limited the window search size for each scale in order to speed up the feature creation and classification steps in the above functions. 

```
def pipeline(img):
    img = img.astype(np.float32)/255  # Test images are jpegs so rescale to be between 0-1
    
    total_bboxes = []
    total_bboxes.extend(find_bbox(img, clf=svr_clf, scaler=X_scaler, scale=0.8, y_start_stop=[450, 720]))
    total_bboxes.extend(find_bbox(img, clf=svr_clf, scaler=X_scaler, scale=1.5, y_start_stop=[450, 680]))
    total_bboxes.extend(find_bbox(img, clf=svr_clf, scaler=X_scaler, scale=2, y_start_stop=[400, 600]))

    #box.add_bboxes(total_bboxes)
    draw_img = box.draw_bboxes(img, total_bboxes)
    draw_img = draw_img.astype(np.float32)*255
    
    return draw_img
```

In code block 25 you can see the colours in test images have been scaled from 0-255 instead of 0-1 giving them the rainbow appearance. This is because the mp4 video requires 0-255 scaling, but throughout my experiments I used 0-1 as required by matplotlib imshow. Below are the same images through the pipeline with the scaling line commented out.

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./resources/result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The Class box_history is designed to filter out false window detections and smooth the bounding boxes over a period of history, see code cell 14. First this class accepts  an image and windows representing areas of positive car detections. The function add_heat is called to calculate the sum of overlapping window regions, with a single region having a value of 1. Then a threshold function is applied, which sets all pixels to 0 if they are not above a certain threshold.
The resulting heatmap is then used in the function label provided from scipy.ndimage.measurements. This function identifies distinct regions in the heatmap and labels them consecutively 1, 2,… etc. The top left and bottom right for each labelled area is then taken as our car bounding box which is drawn on the image.
This method insures several windows then must overlap the same car or it wont meet the threshold limit. Often false positives will be a single window far removed from any other window, so this detections are removed. 

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Perhaps most critically this pipeline will fail at the real time detection of cars. It can often take longer than 1 seconds to process an image, even as high as 60 seconds on my older laptop. You can imagine a self driving car with a 60 second delay in it visual recognition would not be a very effectiv e self driving car. Methods such as boosted cascades have made the “classical” machine learning approach achieved more real time speeds, but these methods still would not be satisfactory for a self driving car. 

The other area where this pipeline could fail is that the bounding boxes often jump sizes, become more narrow or wider between frames. This could confuse the car on if it has enough space to pass the vehicle or not, perhaps mistaking a bus for a bike in extreme circumstances. A good self driving car would be able to smoothly detect the entire shape car smoothly between each frame of the camera.
