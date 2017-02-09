
#Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/center.jpg "Grayscaling"
[image3]: ./img/recover_1.jpg "Recovery Image"
[image4]: ./img/recover_2.jpg "Recovery Image"
[image5]: ./img/recover_3.jpg "Recovery Image"
[image6]: ./img/normal.png "Normal Image"
[image7]: ./img/augmented.png "Augmented Image"
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

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5 convolutional layers with filter sizes ranging from 3x3 to 5x5, followed by 4 fully connected layers.

All layers introduce nonlinearity by using RELU activation(model.py lines 60-75), and the data is normalized in the model using a Keras lambda layer (code line 36). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 60-75). 

The input data was split into training and validation  sets to ensure that the model was not overfitting (code lines 82-91).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Additional validation was performed by running the model in the secondary tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 69).

####4. Appropriate training data

The model was trained on a dataset obtained using an analog wheel input device. The original dataset contains around 10000 
data points. Some recovery driving is included, to enable the network to learn situations where the car has drifted off the track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a relatively simple architecture first, with 6 layers total, to start getting the feel of the project. 
I was surprised to see how far I could get with this basic architecture, but it still would not pass the first track.

After improving my results by using different data augmentation methods which I will detail layer, I thought about trying to modify the architecture
by adding more layers on the smaller network but I always found my changes would either make the network too small to learn enough features, shown in the fact that no matter how many epochs are used the accuracy stays high,
or would make it overfit very quickly.

Because of this I went back to the architecture used by Nvidia in the well known [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I expected this architecture should work well because it was used to solve a similar problem, and indeed I got results very quickly using this model. This architecture performed very well without any further modifications. My alternative model is also included in the model.py file only for documentation purposes.

I found after 5 to 7 epochs overfitting happened in both models, this was noticeable because, while the testing loss was diminishing, validation 
error increased or stayed around the same value. To avoid this, dropout layers were added to this model during testing.

Other variants included trying to use some different activation functions like ELU, and also trying to use an activation layer on the output 
to make it converge quicker to the approximately -1, 1 range. All of these changes were quikly discarded because they would not give the expected results on track. 

####2. Final Model Architecture

The final model architecture (model.py lines 34-75) consisted of a convolution neural network with the following layers and layer sizes:

- Input layer: 64x64x3 images
- Convolutional: 24 filters (5,5), strides=(2, 2), 'same' padding
- Max Pooling: 2x2, strides=(1,1)

- Convolutional: 36 filters (5x5), strides=(2, 2)
- Max Pooling: 2x2, strides=(1,1), 'valid' padding

- Convolutional 48 filters (5x5), strides=(2, 2)
- Max Pooling: 2x2, strides=(1,1), 'valid' padding

- Convolutional 64 filters (3x3), strides=(1, 1)
- Max Pooling: 2x2, strides=(2,2), 'valid' padding

- Convolutional 64 filters (3x3), strides=(1, 1)
- Max Pooling: 2x2, strides=(2,2), 'valid' padding

- Flattening layer
- Fully connected (1164)
- Dense (100)
- Dense (50)
- Dense (10)
- Output: Dense (1)

Here is a visualization of the architecture:

![alt text][image1]

____________________________________________________________________________________________________


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 
recover to the center of the track from these situations. These images show what a recovery looks like starting from the right side
(these images belong to the right camera):

![alt text][image3]
![alt text][image4]
![alt text][image5]

Some samples were removed so the car was not shown when leaving the track, but only joining it instead, since we do not want to induce a bad behaviour into the model.

I did NOT use data from the second track for training the final model.

Some additional augmentation was performed to generate a more regular distribution of steering angle sizes.

The augmentation pipeline includes(in this order):
- Random patches of shadows of triangular shape.
- Shearing to the left or right randomly, with the appropriate correction on the steering angle. This generates a more uniform distribution on the angles.
- Random brightness.
- Random flipping images and angles to compensate for having too many samples of left or right turns.

The images are also cropped to remove the top and bottom parts, and then resized to 64x64.

Here is a sample of the images before and after preprocessing:

</br>

*Before:*

![Image Before preprocessing][image6]   
 
 
</br>

 
  
*After(without resizing to 64x64):*

![Image After preprocessing][image7]

</br>

All of this augmentation results in samples that cover most of the range of steering angles.
as can be seen in these histograms:

*Steering angles before augmentation:*

![alt text][image8]

*Steering angles after augmentation:*

![alt text][image9]

After the collection process, I had 10195 data points. Data is shuffled and 3% of the data is separated into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

The data generator was set to take batches of 64 samples, and `model.fit_generator` is configured to generate epochs of 20000 images.
Under these conditions, the ideal number of epochs was 7(which would) as evidenced by the validation loss not improving anymore.
This can also be confirmed on track number 2, since when the network has memorized track number 1 the performance there is
very bad. The final model can finish both tracks.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
