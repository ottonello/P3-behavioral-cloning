# P3-behavioral-cloning
Behavioral cloning project - part of the Self Driving Car Engineer Nanodegree

## Prerequisites
* Python 3.5
* [Anaconda](https://www.continuum.io/downloads)
* Udacity's Self Driving Car simulator

### Training
Training requires data generated using the simulator:
- Images under the IMG directory
- driving_log.csv file with the training data

To start the training:
* Create the environment from the .yml file: `conda env create -f environment.yml`
* Activate the environment: `source activate drive`
* Run `python train.py`

When training has finished, model and weights files are written: 
  * `model.json`
  * `model.h5`

## Driving 
* Open Udacity's simulator
* Run `python drive.py`

## Obtaining training data
The training data is obtained from the Self Driving Car Engineer Nanodegree simulator, samples of centered driving are taken
first:
