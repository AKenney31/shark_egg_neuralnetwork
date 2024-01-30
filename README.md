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
1. **Drop Rows With Null Shell.Thickness**: Shell Thickness is a column specifically mentioned to include in the training data. Therefore, we will first drop the rows where shell thickness was not measured. 
2. **Drop Bad Rows**: Then we drop the remaining rows that have more than 6 null values. These are deemed as rows that have insufficient data.
3. **Column Null Value Check**: Then we run a script to see which features have the most null values. The features with more than 10 null values remaining are dropped from the dataset.
4. **KNN Imputer**: Here are the [docs](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html). This is meant to accurately predict the remaining null values in a given row with a good guess that depends on the values of the rows with the closest matching known values.
5. **Convert Y Data To One-Hot Encodings**: The y dataset, which is the column of known females in which we are trying to predict, is converted to categorical one-hot encoding using tensorflow's to_categorical function. The resulting y dataset has 20 columns for each of the 20 females.
6. **Scaling**: The X data is scaled using standard scalar. This helps to balance the impact of different features that have different scales. (Features with larger values won't have a larger impact than features with smaller values)
7. **Split To Training and Testing Datasets**: The X data is split into a testing set with 20% of the data and a training set with 80% of the data.

## Model Architecture
The implemented MLP model has the following architecture:
- Input Layer: Matches the number of features (20 neurons)
- Two Hidden Layers: The first with 64 neurons with relu activation, and the second with 128 neurons and swish activation
- Dropout Layers: With a dropout rate of 0.2 after the first hidden layer, and .3 after the second hidden layer to prevent overfitting
- Output Layer: With 20 neurons to match the 20 female names. This has softmax activation.
- Compilation: "Adam" optimizer, "categorical_Crossentropy" loss function, "accuracy" testing metric.

## Model Experimentation
The optimal model was achieved through a meticulous process of experimentation and trial and error. Given the dataset's complexity, characterized by its large number of diverse input features, it was imperative to incorporate an adequate number of neurons across several hidden layers. This configuration is crucial for empowering the neural network with the capability to synthesize information effectively. Insufficient layers and neurons might impede the network's ability to discover patterns within the data. Conversely, an excessive number of neurons and layers might result in overfitting, where the model becomes too tailored to the training data, losing its predictive power on unseen data.

In the initial phase of model tuning, I incrementally added hidden layers to gauge their impact on model accuracy. Hidden layers are often the differentiators between different "logic" steps a neural network may take to arrive at a conclusion. Through this iterative process, it became evident that configurations with two or three hidden layers were most effective, with diminishing returns observed beyond four layers.

Subsequent experiments focused on fine-tuning the neuron count within these layers, evaluating configurations of 64, 128, 256, and 512 neurons. I tried many combinations of these numbers over networks of two and three hidden layers. This exploration led to the identification of an optimal architecture comprising two hidden layers outfitted with 64 and 128 neurons, respectively.

The exploration extended to activation functions, where combinations of ReLU (Rectified Linear Unit) and Swish were trialed across different layers. The pairing of ReLU for the initial hidden layer and Swish for the subsequent one emerged as the most efficacious, striking a balance between linearity and non-linearity conducive to model learning.

Moreover, I delved into optimizing dropout rates, ranging from 0.1 to 0.5, to mitigate the risk of overfitting. The findings suggested that higher dropout rates were particularly beneficial for layers with a greater neuron count, enhancing the model's generalization capabilities.

Lastly, the selection of the optimizer was critical. After experimenting with Stochastic Gradient Descent, AdaGrad, Nadam, and Adam, the Adam optimizer was distinguished as the superior choice, offering the best convergence speed and accuracy.

This comprehensive approach to model development underscored the importance of iterative testing and adjustment of network parameters, culminating in a robust model adept at navigating the complexities of the dataset.

## Scripts
- `main.py`: Main script for the MLP model implementation and training.

