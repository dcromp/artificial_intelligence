# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/signs.png "signs"
[image2]: ./resources/classes.png "classes"

[image3]: ./examples/random_noise.jpg "Random Noise"


[image4]: ./street_signs/12.jpeg "Traffic Sign 1"
[image5]: ./street_signs/17.jpeg "Traffic Sign 2"
[image6]: ./street_signs/13.jpeg "Traffic Sign 3"
[image7]: ./street_signs/14_33.jpeg "Traffic Sign 4"
[image8]: ./street_signs/9.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate the length of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
First is a plot of a random selection of images within the dataset.
![alt text][image1]
Second is a plot of the distribution of classes within the training set.
![alt text][image1]
Some classes like 0 (speed limit 20) and 19 (Dangerous curve to the right) are hardly represented at all. While class 2 (speed limit 50) has the highest representation out of all the classes. This class imbalance could make some of the classes hard to predict accurately.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing I used two methods. First I created new images by adding rotations to the original images. The rotations where between 0-10 degrees. Then I scaled the data to be between -1 and +1, which I found to be the most effective in producing high accuracy during the early epochs.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|
| Max pooling	      	| 2x2 stride,  outputs 4x4x32 				|
| Flatten	      	| outputs 512 				|
| Fully connected		| outputs 500        									|
| Fully connected		| outputs 100         									|
| Fully connected		| outputs 43         									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.513 - 0.522 in the last two runs
* validation set accuracy of approximately 0.935 - 0.937, around on epoch 21
* test set accuracy of 0.892 - 0.911 in the last two runs

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First I tried the basic LeNet model to make sure I could get a known model the run. Then I started by adding blocks of convolutions with a stride of 1, much like those used in VGG 16.
* What were some problems with the initial architecture?
At first the my new network wouldn't train, but the LeNet version produced validation accuracy of around 83%. After lowering the learning late, I finally got my new network to train.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
After 10 or so epochs the network would begin to overfit, as indicated by a lowering of validation accuracy. Adding dropout layers seemed to effectively solve this, and the network would increase in accuracy before completing all epochs.
* Which parameters were tuned? How were they adjusted and why?
I had to adjust the learning rate, otherwise the loss of the network was fixed at a constant at each epoch.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layers should be able to break down the spatial features of the images into a set of non-spatially correlated features that the deep layers can predict on. Also convolution layers are able to take into account the colours of the images, with red, blue and yellow being important colours in traffic signs, so likely to have a large predictive power.

If a well known architecture was chosen:
* What architecture was chosen?
VGG16
* Why did you believe it would be relevant to the traffic sign application?
VGG is a fairly simple network, so easy to code. But has be proven to outperform LeNet (Very deep convolutional networks for large scale image recognition: https://arxiv.org/pdf/1409.1556.pdf).
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The forth image may prove hard to classify has it has a smaller, second traffic sign underneath a larger traffic sign. Also both these signs don't appear as often in the dataset as the other images chosen.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction of the highest scoring run (20%). Often the model would produce 0% accuracy on this data set, as shown in the notebook cell 18:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Dangerous curve to the right   									|
| No entry     			| No entry										|
| Yield					| Priority road											|
| Stop	      		| No passing					 				|
| No passing			| Yield      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This significantly underperforms the 90% scored on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.
These results are from the last run of the notebook and actually scored 0% accuracy, but show some interesting patterns.

For the first the image, Priority Road, the algorithm seemed to think the image was either a speed limit or no passing sign. Interesting as these signs are fairly similar, red circle with white centre.  Priority road is a yellow diamond so clearly the algorithm does not understand colour or shape too well.

Vehicles over 3.5 metric tons prohibited - 0.95 %
No passing for vehicles over 3.5 metric tons - 0.51%
No passing
End of no passing by vehicles over 3.5 metric tons
Priority road

For the second image, No passing, the model was only 50% sure it was a Slippery road sign.

Slippery road - 0.507
Right-of-way at the next intersection
Traffic signals
Priority road
Beware of ice/snow

For the third image Yield, the algorithm was very sure the image was Priority road.

Priority road - 0.99
Road work
Yield
End of no passing by vehicles over 3.5 metric tons
Turn right ahead

Fourth image Stop also predicted with near certainty to be Priority road

Priority road - 0.999999
Yield
No entry
Stop
Keep right

No entry was predicted to be yield at almost 100%

Yield - 0.9999
Speed limit (50km/h)
Go straight or left
Keep right
Speed limit (30km/h)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
