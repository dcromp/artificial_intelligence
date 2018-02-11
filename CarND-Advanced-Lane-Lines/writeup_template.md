## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort.png "undistort"
[image2]: ./examples/find_corners.png "find_corners"
[image3]: ./examples/warp_area.png "warp_area"
[image4]: ./examples/warp_result.png "warp_result"
[image5]: ./examples/sobel_result.png "sobel_result"
[image6]: ./examples/color_result.png "color_result"
[image7]: ./examples/combined.png "combined"
[image8]: ./examples/histograms.png "histograms"
[image9]: ./examples/poly_fit.png "poly_fit"
[video1]: ./examples/test.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The first step in lane finding was to correct for distortions in the camera lens that could make objects appear at incorrect distances and curvatures. To do this I used the provided checkerboard images to generate the distortion coefficients, which are used to undistort all images used throughout the lane finding process. 

![alt text][image2]

The above image shows the result of applying OpenCV find chessboard corner function. From knowing that a chessboard squares are equal in size OpenCV can then calculated the distortion coefficients. To avoid errors several calibration images, such as the one above, are used.

![alt text][image1]

Finally, we can see how we can see how effective using these coefficients for undistorting images are.


### Pipeline (single images)

#### 1. Pipeline Overview

Briefly the steps in the pipeline for single images are as follows.
1. Use camera calibration to undistort image
2. Warp image so the road surface becomes flat from the viewers perspective
3. Apply thresholds to extract features from the images
4. Find the pixels that represent lane lines
5. Fit the pixels with a polynomial

#### 2. Warping the image

We want to warp the image so the road is flat facing the camera. This enables us to easily find the lanes in later steps. Because the camera is at a fixed position in the car, we only need to define the how to do warping once and then apply to all images. The warping itself is defined by four points on the image and where we want to project the four points to.

![alt text][image3]

Here I used an image of a straight road with clear lane lines to help me find which four points to use. These points are connected by the red lines shown in the image. I manually chose these points, and in my first iteration I had chosen the top two points much closer to the horizon. This had negative affects further down the pipeline as it made it much harder to identify lane lines, so I ended up moving the points towards to car from our perspective. The images is ultimelty warped by a transformation matrix which we can get by putting our four source points and four destination points into this openCv function cv2.getPerspectiveTransform()

![alt text][image4]

Above you can see the results of the warping on the test images, as applied by cv2.warpPerspective() with our transformation matrix. You can clearly see which lanes are curving and which are straight, a good sign we are onto the right track.


#### 3. Applying Thresholds 

I probably spent the most time in the project deciding how to use and combine thresholds with different limits.

![alt text][image5]

First I played with the solbol thresholds, ultimately settling for the example above.

![alt text][image6]

Next was the color thresholds. This included playing with stauration and hue of the images

![alt text][image7]

Finally I combined the image warping, sobol and color thresholds into the final result above. I ultimatly decided to use the white and yellow thresholds in the threhold pipeline, defined in cell 15 within the model notebook. You can see the result of this pipeline in cell 16.


#### 4. Find the pixels that represent lane lines

To find the centre of the lane lines I ended up using the histogram technique. The histogram technique takes a slice of image and calculates where along the x-axis do most pixels occur. The max point of pixel density is assumed to be lane, and we should see two such peaks for the left and right lane. All the positive pixel location around these peaks are recorded,  up to the height of the histogram slice. The histogram slides up the image to the next slice, where we again find the peaks and record the pixel locations around the peaks.

![alt text][image8]

In the above image you can see it finds all the lanes for each of the test images. The peak height representing where we should expect to see the lane.

#### 5. Fit the pixels with a polynomial

Finally a polynomial is fitted to the discovered pixels, one for the right lane and one for the left markers. 

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./examples/test.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have begun the implement a class for finding and storing information about the polynomial lines. The next step in this class would be to set it so that it only remembers the last 10 or so lines to avoid the memory filling up and slow down in the code. Also I manually use the curvature of the road as a sanity check, which is recorded in this class. Next I would implement this in such a way that if the sanity check fails then the polynomial is recalculated using the histogram method from scratch, dumping all previous calculated polynomial lines. 

There are certain conditions where this pipeline could fail. An extreme example would be if the road was covered by snow, hiding the lane lanes. This pipeline has no backup should this situation occur and would not be able to find the lane. Other weather conditions such as rain would also reduce the visibility of the lanes and could effect the pipeline.

A more common condition would be if other vehicles obscure the lane line. Large vehicles, such as trucks, could cover the lane line in the entire detection area for some time. In this situation, perhaps it would be possible to use lane line not covered by the truck to estimate where it the obscured lane line should be.

As mentioned above I also found it hard to find good thresholds, and there are almost certainly better combinations with different hyperparameters.
