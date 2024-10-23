# Location Rank (Supervised Learning)

## Overview

Location Rank

### Application Description
A system where you can train a model to get a rank of a location by entering location parameters. 

### Data
Location features: Parking fee (y/n), Antal boenden, Antal lägenheter, Lägenheternas ålder, Medelinkomst, Butiker (y/n), Konkurrens, Skyltning (y/n)

### Data source
A training set is checked in together with this project 'location_training_data.csv'

### Model
Alternative 1: TensorFlow Keras sequential (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
Alternative 2: ????

### Functionality

1. Data load and preprocessing
2. Feature extraction and encoding
3. Model training (TensorFlow Keras Sequential)
4. Model evaluation
5. User interface for entering location data
6. Location raking based on user input

#### Limitations
* Small dataset
* It will only be possible to run from command line 

### Requirements

#### Data requirements
1. Location features:
    * Parking fee (y/n)
    * Antal boenden
    * Antal lägenheter
    * Lägenheternas ålder
    * Medelinkomst
    * Butiker (y/n)
    * Konkurrens
    * Skyltning (y/n)
    * Rank (Target)
2. User data (input):
    * Parking fee (y/n)
    * Antal boenden
    * Antal lägenheter
    * Lägenheternas ålder
    * Medelinkomst
    * Butiker (y/n)
    * Konkurrens
    * Skyltning (y/n)

#### Software requirements

**Modules**

* pandas: Data manipulation and analysis
* scikit-learn: machine learning algorithms and preprocessing
* numpy: numerical operations
* matplotlib: 
* tensorflow: 

**Classes and methods**

1. `LocationData`
    * Responsible for loading, cleaning and preprocessing data
    * `load()`
    * `clean()`
    * `preprocess()`
2. `UserData`
    * Handles user information and ratings
    * `load_user_data()`
    * `update_user_preferences()`
3. `ClassificationModel`
    * `setup()`: Create the model and compile it 
    * `train()`: Train the model with Location data
    * `save()`: Save the trained model
    * `load()`: Load a stored model
    * `evaluate()`: Evaluate the training result
    * `get_rank()`: Generate the classification base on user parameters 
4. `UserInterface`
    * Manages user interactions and displays the result 
    * `get_user_input()`: Prompt user for preferences
    * `display_recommendations()`: Show recommended books to the user

