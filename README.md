# P3-behavioral-cloning
Behavioral cloning project - part of the Self Driving Car Engineer Nanodegree

## Prerequisites
* Python 3.5
* [Anaconda](https://www.continuum.io/downloads)
* An Nvidia GPU
* CUDA 8
* Udacity's Self Driving Car simulator

## Training prerequisites
Training data generated using the simulator:
- Images under the IMG directory
- driving_log.csv file with the training data

## Training
* Create the environment from the .yml file: `conda env create -f environment.yml`
* Activate the environment: `source activate drive`
* Run `python train.py`

When training has finished, model and weights files are written: 
  * `model.json`
  * `model.h5`

## Driving
* Run `python drive.py`
