# **Finding Lane Lines on the Road**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./solutions/canny_solidWhiteCurve.jpg "Canny"
[image2]: ./solutions/region_solidWhiteCurve.jpg "Region of interest"
[image3]: ./solutions/hough_solidWhiteCurve.jpg "Hough Lines"
[image4]: ./solutions/line_solidWhiteCurve.jpg "End result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I used guassian smoothing to insure there was no sharp changes in the gradient. Then I used canny edge detection, followed by mask the hide the region of the images I was not interested in. Then I used a hough transformation to find the lines and drew these lines back onto the original image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by including 3 new functions. The first function determines if the line belongs to the right of left lane from it's gradient. Lines with gradients between -0.1 and 0.1 are ignored as these are horizontal lines and we expect lanes to be vertical lines. The second function calculates the average line for both the left lane and the right line. Finally the last function extrapolates these lines to the bottom the screen and a mid point on the horizon.

If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when objects are directly in front of the car, such as another car. This would create many new lines which would bias the line averaging towards the detected object.

Another shortcoming could be if the car takes a sharp turn then the lanes could potentially disappear as the hough transformation only detects straight lines.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to detect and remove objects within the cars line of site. This would allow the hough transformation to calculate the lines which only correspond to the lanes.

Another potential improvement could be to modify the line calculation so we take into account curved lane lines on the road.
