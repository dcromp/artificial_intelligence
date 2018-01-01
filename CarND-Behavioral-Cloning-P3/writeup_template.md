# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/nvidia_cnn.png "Nvidia model"
[image2]: ./resources/my_network.png "my model"
[image3]: ./resources/recover_left.png "recover_left"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


The architecture of my model was almost identical to the model developed by Nvidia in their end to end self driving car paper. (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

Like the Nvidia architecture, my model consisted of one normalization layer, five convolution layers followed by three fully connected layers. The first three convolutions had a stride of two, with a filter size of five by five. This was followed by two convolutions also with a stride of two, but a filter size of three. ReLu layers where used throughout the network. 

The main difference between my model and the Nvidia model is I have an extra cropping layer at the start of the network. This was to avoid the model training of background details, such as mountains and trees, and focus on learning towards to road.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

Dropout layers of 20% where used for the fully connected layers to prevent overfitting. It is the fully connected layers that suffer most from  overfitting and this is where I saw the biggest difference in using dropout. Adding dropout to the convolutional layers did not seem to make much difference, so no dropout was added at these layers in the final model.

To determine if the model was over fitting, 20% of the data was held out as a validation set. The accuracy of the model with training vs. validation data could then be compared. If the validation error increased, while training accuracy increased, then this indicates the model is over fitting.

#### 3. Model parameter tuning

The model used the adam optimizer, but the learning rate was reduced from 0.001 to 0.0001. I also experimented with a learning rate of 0.0005, which also produced reasonable results and faster, but 0.0001 seemed more reliable, with less variation in the final loss of the model. 0.001 could train the model, but every so often it would seem to get stuck in a local minimum, producing high loss, that does not change from epoch to epoch. 

#### 4. Appropriate training data

Training data was a combination of my own generated data and the data provided by Udacity. For my own data I drove around the track twice, both counter and anti-clockwise. Also, I did two recovery laps, only recording events where the car corrects itself from the edge of the track.

Data augmentation and left and right camera images were also used from this training set, see section below.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first approach was to take three images, one where the car is steering strong left, strong right and no steering. I then developed the Nvidia model with cropping and trained it on these three images. I was looking to see if the model could overfit to these three images, which would be a good indicator that the model would work when the amount of training data increases. The model proved it could overfit this data and would perhaps work well with the full data set.

I then split the data into training and validation and made a generator which yields batches of size 32 to the model. This generator also included augmented data, and the use of the left and right camera with a correction factor (see next section). Dropout on the fully connected layers was used to prevent overfitting, and the validation data was used to monitor if the model was overfitting.

If the validation loss of the model does not lower for 5 epochs, the weights that produced the lowest validation loss are saved, typically the 3rd epoch produced the lowest loss. I then tested the model in the simulator to observe how well it drives. Typically some adjustments where necessary, such as modifying the left and right camera calibration. In these instances I reloaded to previous models weights, so to avoid having to train the model from scratch. This could be considered a type of transfer learning. 

After being satisfied with the results I then recorded the video using video.py and drive.py

#### 2. Final Model Architecture

The figure below shows the final architecture used for my model with the output shapes at each step. All convolutions used a stride of two and the dropout was 20%.

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

As mentioned in point four, I did two regular laps, two recovery laps, and used data provided by Udacity. The below figure shows an image taken from a recovery lap, here the car is being turned left back towards the center of the road.

![alt text][image3]

The left and right camera images are also used in the model. I experimented adjusting the steering correction factor between 0.05 and 0.4. The lower values produced a car that would drive straight, but struggle on the sharper corners. High values would cause the car to wobble along straight sections, but could turn all corners on the track.

Finally, data augmentation is implement by flipping images and reversing angle input. I found data augmentation did not always improve the model and was switched off in the final model. 