
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
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* generators.py containing generators and data augmentation functions.
* model.h5 containing a trained convolution neural network 
* model.json containing the architecture of the model
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolutional neural network with the following layers:
- Input layer: 64x64x3 images
- Convolutional: 16 filters (8x8), strides=(4, 4)
- Convolutional: 32 filters (3x3), strides=(2, 2)
- Max Pooling: 2x2, strides=(2,2), 'valid' padding
- Convolutional 32 filters (5x5), strides=(1, 1)
- Convolutional 64 filters (3x3), strides=(1, 1)
- Max Pooling: 2x2, strides=(2,2), 'valid' padding
- Convolutional 64 filters (5,5), strides=(2, 2)
- Flattening layer
- Fully connected (400)
- Dense (100)
- Dense (20)
- Output: Dense (1)

All layers introduce nonlinearity by using RELU activation(model.py lines 40-64), and the data is normalized in the model using a Keras lambda layer (code line 36). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 51-60). 

The input data was split into training and validation  sets to ensure that the model was not overfitting (code lines 71-80).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Additional validation was performed by running the model in the secondary tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 69).

####4. Appropriate training data

The model was trained on a dataset obtained using an analog wheel input device. The dataset contains around 17 thousand data points. Some recovery driving is included, to enable the network to learn situations where
the car has drifted off the track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a relatively simple architecture first, with 4 or 5 layers, to start getting the feel of the project. I was surprised to see how far I could get with this basic architecture, which after some tuning I chose to use in the final version.

The Nvidia architecture used in the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) linked from the project lesson pages was tested as well. This was the first model which got around the track successfully. I thought it would perform well because it was already tested on a similar task. Dropout was added to this model during testing. Other variants included trying to use some different activation functions like ELU, and also trying to use an activation layer on the output to make it converge quicker to the approximately -1, 1 range. All of these changes were discarded because they would not give the expected results on track.

- after 5 to 7 epochs overfitttng was found, validation error increasing
- went back to basic model and found it worked better without ELU


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