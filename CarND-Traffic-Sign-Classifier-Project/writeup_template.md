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

[image3]: ./resources/transformations.png "Tilted"


[image4]: ./resources/german_signs.png "Traffic Sign 1"


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
![alt text][image2]
Some classes like 0 (speed limit 20) and 19 (Dangerous curve to the right) are hardly represented at all. While class 2 (speed limit 50) has the highest representation out of all the classes. This class imbalance could make some of the classes hard to predict accurately.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing I used three key methods. First I created a histogram equalization representation of the images. This should limit the effect of the image brightness to be consistent across all the images. This will be especially be useful in bringing out features in the darker images.
This was followed by a process of data augmentation, with new data created by adding rotations to the original images. The rotations where between 0-10 degrees. Then I scaled the data to be between -1 and +1, which I found to be the most effective in producing high accuracy during the early epochs.

![alt text][image3]
Example of transformations




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

To train the model I adjusted the following hyperparameters:

Batchsize of 62: The algorithm was run on the CPU with a fairly large memory. 62 seemed to run faster than a 32 batch

epochs 20: All the models I ran seemed to almost converge by 15 epochs, so 20 seemed like a safe choice between time vs. accuracy.

Learning Rate 0.0005: Learning rate was the most tricky to pick as the architecture greatly affected the optimum learning rate. For my model 0.0005 seemed to work well. At 0.0001 it would learn very slow and at 0.001 it failed to learn at all. The LeNet model on the other hand learnt very well at 0.001.

Optimizer: Adam and RMSprop were invistgated but I had more luck with adam, with better learning speed and accuracy. After a quick search I found several resources that came to the same conclusion.
 https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/

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

![alt text][image4]

The forth image may prove hard to classify has it has a smaller, second traffic sign underneath a larger traffic sign. Also both these signs don't appear as often in the dataset as the other images chosen.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction of the highest scoring run (40%) as shown in the notebook cell 18:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road   									|
| No passing     			| No entry										|
| Yield					| Yield											|
| Stop	      		| Vehicles over 3.5 metric tons prohibited					 				|
| No entry			| Stop      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This significantly underperforms the 90% scored on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The softmax probabilities of the results of the model are shown below. Interestingly the model scored extremely high, close to 0.99-1 for every prediction, so the model was very certain even when wrong.



The results for the class Priority road:
    Priority road with probability :- 1.0
    Ahead only with probability :- 4.718349044807724e-10
    No entry with probability :- 4.2280973167052593e-10
    Speed limit (120km/h) with probability :- 4.401532610609493e-11
    Roundabout mandatory with probability :- 2.3511449731561385e-11


    The results for the class No passing:
    No entry with probability :- 1.0
    Turn left ahead with probability :- 1.6427426086096375e-10
    Speed limit (80km/h) with probability :- 1.06284772649623e-10
    Stop with probability :- 1.0172858389001505e-10
    No vehicles with probability :- 2.4652009600334424e-11


    The results for the class Yield:
    Yield with probability :- 0.9999377727508545
    No passing for vehicles over 3.5 metric tons with probability :- 6.199027120601386e-05
    Keep right with probability :- 2.2937145160994987e-07
    Wild animals crossing with probability :- 2.4040096580080217e-09
    Speed limit (120km/h) with probability :- 6.102564287235879e-11


    The results for the class Stop:
    Vehicles over 3.5 metric tons prohibited with probability :- 0.9928694367408752
    No passing with probability :- 0.005848763510584831
    Speed limit (100km/h) with probability :- 0.0008854049956426024
    No passing for vehicles over 3.5 metric tons with probability :- 0.0003294068737886846
    Roundabout mandatory with probability :- 3.82636280846782e-05


    The results for the class No entry:
    Stop with probability :- 0.9996724128723145
    Turn right ahead with probability :- 0.00019469571998342872
    End of no passing by vehicles over 3.5 metric tons with probability :- 8.540235285181552e-05
    Speed limit (80km/h) with probability :- 2.573734673205763e-05
    No vehicles with probability :- 1.2314597370277625e-05

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
