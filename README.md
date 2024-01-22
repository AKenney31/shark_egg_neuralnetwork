# Shark Egg Female Predictor

## Overview
This project involves the development of a neural network-based model to predict the female shark from which a given egg came. The model is built using TensorFlow and employs a Multilayer Perceptron (MLP) to handle the high-dimensional data set composed of various measurements of shark eggs.

## Requirements
- [Python 3.11](https://www.python.org/downloads/).
- TensorFlow
- Scikit-learn
- Pandas
- Numpy
- It is recommended that you use a python virtual environment. [Setup venv in VSCode](https://code.visualstudio.com/docs/python/environments), [Setup venv in PyCharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html), [Venv documentation](https://docs.python.org/3/library/venv.html). This allows you to manage separate package installations for different projects. It creates a “virtual” isolated Python installation, guaranteeing proper package versions and proper Python versions unaffected by new extraneous installations.

After setting up your venv, you can install the necessary libraries using:
```bash
pip install tensorflow scikit-learn pandas numpy openpyxl
```
If you are using pycharm, see this [Installing packages in PyCharm](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html#install-in-tool-window).

### Data Format
- The dataset is an Excel file.
- Each row represents a shark egg instance.

## Usage
1. **Data Preparation**: Make sure the dataset, Genetic_SingletonEggsMatch_Udel.xlsx, is in the project directory.
2. **Train and Run Model**: Run the Python script to build the dataset and train the model. The script will automatically scrape the data from the Excel file, clean the data by handling null values and dropping columns, split the data into training and testing sets, scale the features, define the MLP model, and train it.
3. **Evaluation**: After training, the model's accuracy on the test set will be printed in the console. The output dataset will be displayed in a new Excel file entitled "Results.xlsx". The predicted female shark for that egg instance will be printed in the last column of the results spreadsheet.

## Data Preparation
Starting train dataset pulls in 22 features and no dropped rows.
1. **Drop Bad Rows**: First we drop the rows that have more than 6 null values. These are deemed as rows that have insufficient data.
2. **Column Null Value Check**: Then we run a script to see which features have the most null values. The features with more than 10 null values remaining are dropped from the dataset.
3. **KNN Imputer**: Here are the [docs](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html). This is meant to accurately predict the remaining null values in a given row with a good guess that depends on the values of the rows with the closest matching known values.
4. **Convert Y Data To One-Hot Encodings**: The y dataset, which is the column of known females in which we are trying to predict, is converted to categorical one-hot encoding using tensorflow's to_categorical function. The resulting y dataset has 20 columns for each of the 20 females.
5. **Scaling**: The X data is scaled using standard scalar. This helps to balance the impact of different features that have different scales. (Features with larger values won't have a larger impact than features with smaller values)
6. **Split To Training and Testing Datasets**: The X data is split into a testing set with 20% of the data and a training set with 80% of the data.

## Model Architecture
The implemented MLP model has the following architecture:
- Input Layer: Matches the number of features (19 neurons)
- Two Hidden Layers: The first with 64 neurons with relu activation, and the second with 128 neurons and swish activation
- Dropout Layers: With a dropout rate of 0.2 after the first hidden layer, and .3 after the second hidden layer to prevent overfitting
- Output Layer: With 20 neurons to match the 20 female names. This has softmax activation.
- Compilation: "Adam" optimizer, "categorical_Crossentropy" loss function, "accuracy" testing metric.

## Scripts
- `main.py`: Main script for the MLP model implementation and training.

