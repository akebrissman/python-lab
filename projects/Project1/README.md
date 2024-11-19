# Location Rating (Supervised Learning)

## Overview

Location Rating

### Application Description
The purpose of this application is to let a trained AI model help predicting if a location is good or bad based on some critera.
There are two distinct parts:
1. Training. The model is trained with location data and when you are satisfied the model is stored. 
2. Predicting. A stored model can be loaded, and by giving som location data it can predict (classify) if this location is bad, poor, fair, good or excellent. 

### Data
Location features: Number of people, Avg income, Percentage of apartments, Percentage of flats, Percentages of EV,  Age of apartments, Type, Competition, Signage.\
Location labels: BAD, POOR, FAIR, GOOD, or EXCELLENT

### Data source
A training set 'location_training_data.csv' is part of this project
This file can be replaced with production data when it is time to run it in a production setup.

### Model
TensorFlow Keras Sequential \
See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

### Functionality
* Data load from CSV
* Data preprocessing. (Feature extraction, encoding, normalization)
* Model creation (TensorFlow Keras Sequential)
* Model training 
* Model evaluation
* Model graph
* Model store and load 
* Model prediction with user data
* Command line interface to manage the functionality

### Limitations
* Small dataset, synthetic data must be used
* Only support of command line interface (No web UI) 

### Requirements

#### Data requirements
1. Location features (CSV File with the following data per location)
   * Number of people (Number of people living in the area)
   * Avg income (The avarage income of the people in the area)
   * Percentage of apartments (% of apartments)
   * Percentage of flats (% of owned flats)
   * Percentages of EV (% of Electric Vehicles)
   * Age of apartments (1-4)
   * Area type (1-5)
   * Competition (1-3)
   * Signage (1-3)
2. Location data label: (Each location is ranked as)
   * EXCELLENT GOOD, FAIR, POOR, BAD
3. User data (input):
   * Number of people (Number)
   * Avg income (Number)
   * Percentage of apartments (0-100)
   * Percentage of flats (0-100)
   * Percentages of EV (0-100)
   * Age of apartments (1-4)
   * Area type (1-5)
   * Competition (1-3)
   * Signage (1-3)

#### Software requirements

##### Modules #####

* pandas: Data manipulation and analysis
* numpy: numerical operations
* scikit-learn: machine learning algorithms and preprocessing
* matplotlib: 
* tensorflow: 
* MinMaxScaler: 

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

##### Links #####
* [Tensor Flow Sequential](https://www.tensorflow.org/guide/keras/sequential_model)
* [Scikit learn - MinMaxScaler](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
* [Scikit learn - Encoding of categorical variables](https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html)
* [Pandas - Convert categorical variables](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
* [Pandas - Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)