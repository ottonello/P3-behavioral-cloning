
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/augmented.png "Augmented Image"
[image8]: ./img/histogram_before.png "Histogram before augmentation"
[image9]: ./img/histogram_after.png "Histogram after augmentation"

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

####1. An appropriate model archtiecture has been employed

My model consists of a convolutional neural network with 5 convolutional layers with filter sizes ranging from 3x3 to 8x8, followed by 3 fully connected layers.

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

I started with a relatively simple architecture first, with 4 or 5 layers, to start getting the feel of the project. I was surprised to see how far I could get with this basic architecture, but it still would not pass the first track.

Then I started using the architecture used in Nvidia's [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), linked from the project lesson pages was tested as well. I expected this architecture should work well because it was used to solve a similar problem. This was the first model which got around the track successfully.

I found after 5 to 7 epochs overfitting happened in both models, this was noticeable because, while the testing loss was diminishing, validation error increased or stayed around the same value. To avoid this, dropout layers were added to this model during testing.

After improving my results by using different data augmentation methods which I will detail latyer, I went back to fine tuning my architecture.

Other variants included trying to use some different activation functions like ELU, and also trying to use an activation layer on the output to make it converge quicker to the approximately -1, 1 range. All of these changes were discarded because they would not give the expected results on track.

In the end I decided to try a smaller network, so I went back to my original, basic architecture and added a couple more convolutional layers and one more fully connected layer. After some more fine tuning I found this network to perform better.

The final model drives around track 1 without ever leaving the track, after training for 5 or 6 epochs of around 20000 samples. It is verified that it hasn't memorized track 1, by checking that the same model is also capable to finish track no. 2.

####2. Final Model Architecture

The final model architecture (model.py lines 34-64) consisted of a convolution neural network with the following layers and layer sizes:

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

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I did NOT use data from the second track for training the final model.

Some additional augmentation was performed to generate a more regular distribution of steering angle sizes.

The augmentation pipeline includes(in this order):
- Random patches of shadows of triangular shape.
- Shearing to the left or right randomly, with the appropriate correction on the steering angle. This generates a more uniform distribution on the angles.
- Random brightness.
- Random flipping images and angles to compensate for having too many samples of left or right turns.

The images are also cropped to remove the top and bottom parts, and then resized to 64x64.

Here is a sample of the images before and after preprocessing:

![alt text][image6]
![alt text][image7]

After 

![alt text][image8]
![alt text][image9]

After the collection process, I had X number of data points. To preprocess this data I 


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.