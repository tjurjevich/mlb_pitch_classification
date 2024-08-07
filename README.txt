This project provides a model that will predict Major League Baseball (MLB) pitch types. 
Files in this repository and a brief description of what they do...

'pitch_classification_analysis.ipynb': This notebook provides some brief analysis and visualizations of different variables.

'statcast_data_pull.py': A python script that saves out a .csv file used for eventual model training. Statcast pitch data ranges from March 28th, 2024 (MLB opening day) to July 24th, 2024 (date last ran). This can technically be rerun to pull in additional pitch data, if desired.

'statcast_data_TRAIN.csv': A csv file containing all returned variables from pybaseball's statcast() function (446,441 rows).

'statcast_data_TEST.csv': A csv file containing all returned variables from pybaseball's statcast() function. This data is comprised of MLB statcast pitch data ranging from July 25th, 2024 to August 4th, 2024 (42,949 rows).

'requirements.txt': A text file containing packages and their versions needed for working through this repository.

'pitch_classification_train.py': A python script that contains code for the NNPitchClassification class--a neural network designed to predict MLB pitch type. 7 different input variables are used to predict 'pitch_type' from pybaseball statcast() data.

'pitch_classifier_neural_network.h5': The trained tensorflow/keras model from pitch_classification_train.py.


----------MODEL ACCURACY----------
The trained deep learning model produced validation accuracies of ~98.5% throughout its epochs, and produced an accuracy on the statcast_data_TEST of ~96.5%.

