#**Behavioral Cloning** 

##Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test/center_2017_04_24_10_29_37_786.jpg "Centre Lane"
[image2]: ./test/center_2017_04_25_14_09_19_555.jpg "Right Lane"
[image3]: ./test/center_2017_04_25_14_09_21_162.jpg "Recovery In progress"
[image4]: ./test/center_2017_04_25_14_09_23_272.jpg "Recovery Image"
[image5]: ./test/center_2017_04_24_10_29_51_843.jpg "Normal Image"
[image6]: ./test/center_2017_04_24_10_29_51_843_flipped.jpg "Flipped Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I started off with a model similar to model in previous assignment.Then started adding more convolutional layer. When the results weren't satisfactory I started venturing into experiments done already. Came across NVIDIA's published model. My model resembles a [NVIDIA E2E DL model architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Solution takes care of optimizing the solution by pre-processing training data, configuring the params of network which yield better result for our images and adding layers. 

Model is defined in line 114 (train.py)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after every conlutional layer to reduce overfitting.

Validation of data was done to ensure that the model wasn't overfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 102).

####4. Appropriate training data

Training data was cruicial part of this. Not only centre lane driving but recovering was also a concern. I collected data by driving the car centre of the lane most of the time and then on certain occassions deviated from the center to capture recovery too.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Solution had three parts:
#####1. Collecting training data
Collecting training data was a crucial part of the solution. If someone kept on driving into the center of the lane it won't help the model in understanding recovery well.

While collecting data I took care of creating dataset for recovery too.
#####2. Model
After starting from a very basic connected network I moved on to a network similar to last assignment. Then kept on experimenting with changing and adding new layers. Did some brwsing around different models for similar problem. Then picked up the NVIDIA E2E DL model for Self Driving Cars. With this model started experitmenting with params and adding layers.

#####3. Training
While data was collected and model was ready next challenge was how do we avoid overfitting and underfitting. In order to achieve these Images were preprocessed for flip and center alignment.

Model was added with Dropout layer to avoid overfitting. With 0.5 the results yielded was satisfoctory.

At the end to measure the model performance I split the training data into two parts training set and validations set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

Final architecture resembles NVIDIAs E2E DL for Self Driving cars. This model helped me in obtaining the best result. This model had:

|	Layer					|	Description	|
|:--------------------:|:--------------------------:| 
|Noramlization		|	Lambda |
|Cropping2D			|	(60,20) (0,0)|
|Convolution2D		|	Convolutional 24@79x159|
|Activation			|	Relu|
|Dropout				|	0.5|
|Convolution2D		|	Convolutional 36@39x79|
|Activation			|	Relu|
|Dropout				|	0.5|
|Convolution2D		|	Convolutional 48@19x39|
|Activation			|	Relu|
|Dropout				|	0.5|
|Convolution2D		|	Convolutional 64@9x19|
|Activation			|	Relu|
|Dropout				|	0.5|
|Convolution2D		|	Convolutional 64@5x10|
|Activation			|	Relu|
|Dropout				|	0.5|
|Flatten				| 	Flatten|
|Dense					|	Dense(100)|
|Dense					|	Dense(50)|
|Dense					|	Dense(10)|
|Dense					|	Dense(1)|

Final out put is steering angle.

Model is in (train.py 114 - 146).

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when car reaches closer to left or right. These images show what a recovery looks like starting from right end:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After this preprocessing in the model I included a cropping layer to crop the unwanted details in the image I just kept 60, 20 vertical window. This helped a lot in attaining validation accuracy.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I experitmented with multiple different epochs run. Then I settled for 50 because the increase in validation accuracy wasn't much. Close to 50 it started fluctuating.
```
5216/5216 [==============================] - 59s - loss: 0.0227 - val_loss: 0.0519
```
```
Epoch 48/50
5216/5216 [==============================] - 59s - loss: 0.0227 - val_loss: 0.0466
```
```
Epoch 49/50
5216/5216 [==============================] - 59s - loss: 0.0237 - val_loss: 0.0521
```
```
Epoch 50/50
5216/5216 [==============================] - 59s - loss: 0.0214 - val_loss: 0.0444
```
After this I ran the model with drive.py in autonomous mode in similator and this helped me in validating my model well.

As next steps I would like to experiment with pretrained models and see how this helps me reducing the training time.


