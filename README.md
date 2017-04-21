# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 containing the video of the autonomous driving
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```
Here, I applied the same pre processing steps to the images in the drive.py file to match the model ( Lines 67 and 68). 

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. You can find 2 functions that train 2 different model architectures. Here, I experimented with a [Lenet Architecture](http://yann.lecun.com/exdb/lenet/) and [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I also have differnt functions written to handle all the steps in the process I followed. I didnt have to use Python generators as I used a 64GB ram machine with a GPU. It was powerful enough to handle the data in memory. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Here I experimented with 2 model architectures. The first model I worked with is Lenet (model.py lines 106-123). Below is the model description. This architecture did not perform to the level I needed. Although It recognised the lanes, It didnt work very well with bends. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x320x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 76x316x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 38x158x6			|
|	Dropout					|	Keep_prob = 0.5											|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 34x154x16						|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 17x77x16 				|
| Flatten					|Output = 1309												|
|	Fully connected					|	Output = 120										|
|	Fully connected					|	Output = 84										|
|	Fully connected					|	Output = 1										|


The 2nd model architecture I used was the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Below is the representation of the Architecture (model.py lines 129 - 153). This is the model I used as the final successful model. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x320x3 RGB image  							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 76x316x24  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 38x158x24 				|
|	Dropout					|	Keep_prob = 0.5											|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 34x154x36								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 17x77x36 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 13x73x48								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x13x48 				|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 4x11x64								|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 2x9x64							|
| RELU					|												|
| Flatten					|Output = 1152												|
|	Fully connected					|	Output = 1164										|
|	Fully connected					|	Output = 100|
|	Fully connected					|	Output = 50										|
|	Fully connected					|	Output = 10									|
|	Fully connected					|	Output = 1										|


#### 2. Attempts to reduce overfitting in the model

I used 50% dropout in both models to avoid the overfitting (model.py lines 111 and 134). 
I also used 25% split of the training and Validation data. This avoids the overfitting as well. 
I also used shuffling t randomize the data.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 120 and 150).

#### 4. Appropriate training data

I used 2 laps each on track 1 in forward dirention and backward direction to get the regular center line driving data. I also collected data on recovering from crossing the lane lines in either side of the road. I also added data in the bends where the car takes larger angles. In regular centerline driving, data related to the bends are limited. 

Below is the histogram of the data distribution. 

![alt text][image1] 

I also used a flipped image of the original as an augmentation method. This provided me with additional training data and helped generalize the model. 

Below is an original image example and a flipped image of the same. 

![alt text][image2] ![alt text][image3]
 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with Lenet architecture and building up to other model architectures. 

My first step was to use a convolution neural network model similar to the Lenet architecture. I started with the same architecture I used for the traffic sign classification project. The details are explained in the previous section under "1. An appropriate model architecture has been employed" . I thought will be a great starting point for me. This is a quite small architecture that I can use to see quick results in the initial stages of the process. This helped me arrange my training data to cover most of the cases I needed to cover. Lenet didnt perform well in Bends and had hard time recovering from going off the road. part of this was due to unbalanced trainig data as well. I used various recovery data sets collected using the Simulator and also image augmentation to address this issue. 

Although Lenet performed better with these improvements, It wasnt able to complete the whole lap in the autonomous mode. 

I then implemented the model Architecture NVIDIA used for thier Self Driving car. [This link](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) explains the model. Model is explained under "1. An appropriate model architecture has been employed" above. This model performed really well with the same training data as Lenet. 

I used 25% Train/Validation split and used MSE as for the loss function. I used 10 Epochs for both model architectures. Lenet model displayed a really low MSE at the end of the 10th epoch but the validation error almost 3 times that. This meens that the model was overfitting. I observed a similar behavior for NVIDIA architecture as well. To combat this, I used below strategies.


* I used 50% dropout in both models to avoid the overfitting (model.py lines 111 and 134). 
* I also used 25% split of the training and Validation data. This avoids the overfitting as well. 
* I also used shuffling t randomize the data.

This reduced the validation error and brought it down closer to the training error. 

I choose the NVIDIA based model as my final model and used to the simulator to drive the car automously with that.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Below is the representation of the Architecture (model.py lines 129 - 153). This is the model I used as the final successful model. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x320x3 RGB image  							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 76x316x24  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 38x158x24 				|
|	Dropout					|	Keep_prob = 0.5											|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 34x154x36								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 17x77x36 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 13x73x48								|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x13x48 				|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 4x11x64								|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 2x9x64							|
| RELU					|												|
| Flatten					|Output = 1152												|
|	Fully connected					|	Output = 1164										|
|	Fully connected					|	Output = 100|
|	Fully connected					|	Output = 50										|
|	Fully connected					|	Output = 10									|
|	Fully connected					|	Output = 1										|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also used several recovery data sets. I Drove the car out the road to cross the lane marks and then started recording the recovery back to the center of the road. I used several of these recovery attemps to get enough data. Below images shows the process of recovery

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would balance the data to present both sides of the road. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also used left and right camera photos with +/- 0.2 angles from the original angle and added to the trainign set. 

After the collection process, I had 59850 number of data points. I then preprocessed this data by doing the following.
I converted the image to BGR and also croped the image to exclude the background from the image. This lets the model pick up feartures on the road. Below examples show before and after cropping. 

![alt text][image6]
![alt text][image7]


I finally randomly shuffled the data set and put 25% of the data into a validation set. 

Final Training and Validation set data set looks like follows
Train on 44887 samples
validate on 14963 samples

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Please see the vedio for autonomous driving of the car.
