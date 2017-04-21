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
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```
Here, I applied the same pre processing steps to the images in the drive.py file to match the model ( Lines 67 and 68). 

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. You can find 2 functions that train 2 different model architectures. Here, I experimented with a [Lenet Architecture](http://yann.lecun.com/exdb/lenet/) and [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I also have differnt functions written to handle all the steps in the process I followed. 

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

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
