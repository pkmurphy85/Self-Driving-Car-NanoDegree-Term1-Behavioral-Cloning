# **Behavioral Cloning**
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/center_2018_04_15_17_34_49_863.jpg "Center Lane Driving"
[image2]: ./Images/cropped.png "Cropped"

[image3]: ./Images/center_2018_04_22_17_53_52_932.jpg "Recovery Right Image 1"
[image4]: ./Images/center_2018_04_22_17_53_53_004.jpg "Recovery Right Image 2"
[image5]: ./Images/center_2018_04_22_17_53_53_073.jpg "Recovery Right Image 3"
[image6]: ./Images/center_2018_04_22_17_53_53_163.jpg "Recovery Right Image 4"
[image7]: ./Images/center_2018_04_22_17_53_53_238.jpg "Recovery Right Image 5"

[image8]: ./Images/center_2018_04_22_17_53_13_297.jpg "Recovery Left Image 1"
[image9]: ./Images/center_2018_04_22_17_53_13_369.jpg "Recovery Left Image 2"
[image10]: ./Images/center_2018_04_22_17_53_13_451.jpg "Recovery Left Image 3"
[image11]: ./Images/center_2018_04_22_17_53_13_523.jpg "Recovery Left Image 4"
[image12]: ./Images/center_2018_04_22_17_53_13_593.jpg "Recovery Left Image 5"



[image13]: ./Images/left_2018_04_22_17_53_13_593.jpg "Left Image"
[image14]: ./Images/right_2018_04_22_17_53_53_238.jpg "Right Image"
[image]: ./examples/placeholder_small.png "Normal Image"
[image]: ./examples/placeholder_small.png "Flipped Image"

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy for deriving a model architecture was to tweak existing image recognition architectures and repurpose them to successfully teach the car to drive around the track autonomously.

My initial model was a convolution neural network model based on [LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I found this to be a good starting point as we previously repurposed this model for traffic sign recognition with success. My implementation of this model performed ok at keeping the car on the road for some time, but struggled to successfully traverse the course.

For my final implementation I utilized the network developed by NVIDIA described in the paper ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). A table of the implementation layers are below:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input           | 160 x 320 x 3 BGR image   					|
| Cropping        | Output 65 x 320 x 3 BGR image   					|
| Convolution 5x5     	| 24 filters, 2x2 stride, relu activation 	|
| Convolution 5x5     	| 36 filters, 2x2 stride, relu activation 	|
| Convolution 5x5     	| 48 filters, 2x2 stride, relu activation 	|
| Convolution 3x3     	| 64 filters, 1x1 stride, relu activation 	|
| Convolution 3x3     	| 64 filters, 1x1 stride, relu activation 	|
| Fully connected		| 1164 to 100   									|
| Fully connected		| 100 to 50   									|
| Fully connected		| 50 to 10   									|
| Fully connected		| 10 to 1   									|

First the data is normalized to the range [0,1] using an lambda layer. Then each image is passed through a cropping layer to isolate the section of the image with the road and remove unnecessary information such as mountains in the background. Below is an example of an input image and a cropped image:

Input image:

![alt text][image1]

Cropped image:

![alt text][image2]

Next the model consists of 5 convolutional layers. The first three use 5x5 filters with depths of 24, 36, and 48 respectively. The 4th and 5th layers use 3x3, both with depths of 64. RELU layers are included after each convolutional layer to introduce nonlinearities.  

After the convolutional layers the model consists of four fully connected layers. The fully connected layers reduce size from 1164 to 100, 100 to 50, 50 to 10, and 10 to 1.

#### 2. Creation of the Training Set

To capture good driving behavior, I first recorded several laps on track one using center lane driving. I then recorded the vehicle traversing the track in reverse. This is like driving the car on a brand new track and helps the model generalize.

In addition to center camera images I used camera images from the left and right cameras. Examples are below:

Left Camera:

![alt text][image13]

Right Camera:

![alt text][image14]

During training the left and right camera images are fed to the network as if they are center camera images. These help teach the car how to veer back to the center when it drifts. The steering angle for each left or right image includes a small correction factor which helps teach the car to correct its position if it is off to one side. Including left and right camera images also provides us with 3 times as much data.

After several attempts my network still struggled to keep the car on the road. To improve my training set I recorded the vehicle recovering from the left and right sides of the road back to center so that the vehicle would learn to how to react if it got off center. These images show what a recovery looks like starting from the right:

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

and the left:

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]



After the collection process, I had 35,976 images.

### 3. Training Process

In order to gauge how well the model was working, I randomly shuffled the data and split 20% into a validation set. I evaluated the mean squared error (MSE) on the training and validation sets to determine if my model was under or overfitting. In training my final model I found both training loss and validation loss to be relatively low, so I did not include additional dropout layers.

After building a several models the car had some trouble staying on the track in certain locations, particularly when it moved towards the edge of the road. To combat this I collected more training data. This data focused on collecting samples where the car was on the left or right side of the track and corrected itself back to center lane driving. This training data proved very useful for training the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Click to watch!

[![Video 1](https://i.imgur.com/NZ6KYJN.jpg)]("https://youtu.be/JIq5rcHnNak")
