# Shark Egg Female Predictor

## Overview
This project involves the development of a neural network-based model to predict the female shark from which a given egg came. The model is built using TensorFlow and employs a Multilayer Perceptron (MLP) to handle the high-dimensional data set composed of various measurements of shark eggs.

## Requirements
- Python 3.x
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

## Dataset
The dataset consists of hundreds of rows of shark eggs, each with 17 different measurements. The dataset is split into input features (17 columns) and the target output (1 column indicating the female shark).

### Data Format
- The dataset is an Excel file.
- Each row represents a shark egg instance.

## Usage
1. **Data Preparation**: Place your dataset in the project directory and update the file path in the code if necessary.
2. **Model Training**: Run the Python script to build the dataset and train the model. The script will automatically scrape the data from the Excel file, split the data into training and testing sets, scale the features, define the MLP model, and train it.
3. **Evaluation**: After training, the model's accuracy on the test set will be printed. The output dataset will be displayed in a new Excel file entitled "Results.xlsx".

## Model Architecture
The implemented MLP model has the following architecture:
- Input Layer: Matches the number of features (17 neurons)
- Two Hidden Layers: The first with 64 neurons and the second with 128 neurons, both with ReLU activation
- Dropout Layers: With a dropout rate of 0.5 after each hidden layer to prevent overfitting
- Output Layer: With a single neuron using the sigmoid activation function (for binary classification)

## Scripts
- `main.py`: Main script for the MLP model implementation and training.

