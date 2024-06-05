# Bank-Churn-Classification-Model
Classic churn classification model for a bank

### Pipelines and main
This folder contains to classes, Trainer and Scorer that will train, evaluate and save the best model. The Scorer class is used to predict with the test data.
The main.py file will orchestrate both classes, training and testing by saving best models in the Pickle folder

### 2do parcial
This folder contains a modelling file where it was performed a Exploratory Data Analysis and models were tested. It also contains the notebooks train and test and a Pickle_ipynb folder. The train notebook will train a LGBM model with tuned hyperparameters, saving the encoders, the scaler and the model itself as pickles to the Pickle_ipynb folder. The test notebook will load this files and predict with test data.

