# Location Rating (Supervised Learning)

## Overview

Location Rating

### Application Description
The purpose of this application is to let a trained AI model help predicting if a location is good or bad based. \
There are two distinct parts:
1. Training. The model is trained with location data and when you are satisfied the model can be store. 
2. Predicting. A stored model can be loaded and by giving som location data it can predict if it will be classified as good, fair or bad 

### Data
Location features: Parking fee, Number of people, Number of apartments, Age of apartments, Average income, Shops, Competition, Signage

### Data source
A training set 'location_training_data.csv' is part of this project
This file can be replaced with production data when you want to run it in production setup.

### Model
TensorFlow Keras sequential \
See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

### Functionality
1. Loading and preprocessing of data
2. Feature extraction, encoding, normalization
3. Model training (TensorFlow Keras Sequential)
4. Model evaluation
5. Model prediction with user data
6. Command line interface to manage the functionality

### Limitations
* Small dataset
* It will only be possible to run from command line 

### Requirements

#### Data requirements
1. Location features:
   * Parking fee
   * Number of people
   * Number of apartments
   * Age of apartments
   * Average income
   * Shops
   * Competition
   * Signage
2. Location label:
   * GOOD, FAIR, BAD
3. User data (input):
   * Parking fee
   * Number of people
   * Number of apartments
   * Age of apartments
   * Average income
   * Shops
   * Competition
   * Signage

#### Software requirements

##### Modules #####

* pandas: Data manipulation and analysis
* scikit-learn: machine learning algorithms and preprocessing
* numpy: numerical operations
* matplotlib: 
* tensorflow: 

##### Classes and methods #####

1. `LocationData`
    * Responsible for loading, cleaning and preprocessing data
    * `load()`
    * `preprocess()`
2. `UserData`
    * Handles user input of location data to rate
    * `get_user_input()`
3. `ClassificationModel`
    * `setup()`: Create the model and compile it 
    * `train()`: Train the model with Location data
    * `save()`: Save the trained model
    * `load()`: Load a stored model
    * `evaluation_graph()`: Display accurate/loss graphs
    * `evaluate()`: Evaluate the training result
    * `rate()`: Generate the prediction base on user input 
4. `UserInterface`
    * `main_menu()`: Displays the menu items 
    * `user_input()`: Prompt user for location values

