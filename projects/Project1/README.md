# Location Rating (Supervised Learning)

## Overview

Location Rating

### Application Description
The purpose of this application is to train an AI model with location data and then use it  \
There are two distinct parts:
1. Train. The model can be trained with location data and then stored, to be used later. 
2. Predict. A stored model can be loaded, and by entering new location data it can predict (classify) the new location as bad, poor, fair, good or excellent. 

### Functionality
When running the application main.py you will get a command line interface from where you 
can select different options. Only the options which are allowed, at the given time, will be displayed.

_Note: When loading data (option1.) it is recommended to also create synthetic data by entering 1s._ 

Functions

* Data load from CSV (With or without synthetic data)
* Data preprocessing. (Feature extraction, encoding, normalization)
* Model creation (TensorFlow Keras Sequential)
* Model training 
* Model evaluation
* Model graph
* Model store and load (Stored in folder storage)
* Model prediction (Data entered by the user)
* Command line interface to manage the functionality

### Setup
      # Install dependencies 
      cd python-lab/projects/project1
      python -m venv venv
      source venv/bin/activate
      # venv/Scripts/activate
      pip install -r requirements.txt

      # Run
      python main.py

### Data source 
The training set 'location_training_data.csv' is a manually created dataset.
The columns (features) are decided by the developer but the values are collected from internet and from 
own domain knowledge. \
It is quite small dataset which is checked in as part of this project, but it is good enough to 
run the project and validate the model. 
The datafile can be replaced with a much bigger and more accurate data set when it is time to run it in 
for production use.

### Data
The following columns, from the file 'location_training_data.csv', are used as features and label. \
Features columns: Number of people, Avg income, Percentage of apartments, Percentages of EV, Type apartments, Area type, Competition, Signage.\
Label column: Rating with the distinct values BAD, POOR, FAIR, GOOD, EXCELLENT

### Model
The model TensorFlow Keras Sequential is used in this project. \
It fits well the purpose of doing classification with supervised learning  
See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

### Limitations
* Small dataset, synthetic data must be used for a good training result
* Only support of command line interface (No web UI)
* All features are weighted equally, but there are comments in the code where to change if weighting is needed.

### Requirements

#### Data requirements
1. Location features (CSV File with the following data per location)
   * Number of people (Number of people living in the area)
   * Avg income (The average income of the people in the area)
   * Percentage of apartments (% of apartments)
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
   * Percentages of EV (0-100)
   * Age of apartments (1-4)
   * Area type (1-5)
   * Competition (1-3)
   * Signage (1-3)

#### Software requirements

##### Modules #####

* pandas: Data manipulation and analysis
* numpy: Numerical operations
* scikit-learn: Machine learning algorithms and preprocessing
* matplotlib: Used for the evaluation graphs
* tensorflow: Model
* MinMaxScaler: Data normalization

##### Classes and methods #####

1. `LocationData`
    * Responsible for loading, cleaning and preprocessing data
    * `load()`
    * `preprocess()`
2. `UserData`
    * Handles user input of location data to rate (predict)
    * `get_user_input()`
3. `ClassificationModel`
    * `setup()`: Create the model and compile it 
    * `train()`: Train the model with Location data
    * `save()`: Save the trained model
    * `load()`: Load a stored model
    * `evaluation_graph()`: Display accurate/loss graphs
    * `evaluate()`: Evaluate the training result
    * `predict()`: Generate the prediction base on user input 
4. `UserInterface`
    * `main_menu()`: Displays the menu items 
    * `user_input()`: Prompt user for location values

##### Links #####
* [Tensor Flow Sequential](https://www.tensorflow.org/guide/keras/sequential_model)
* [Scikit learn - MinMaxScaler](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
* [Scikit learn - Encoding of categorical variables](https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline.html)
* [Pandas - Convert categorical variables](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
* [Pandas - Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)