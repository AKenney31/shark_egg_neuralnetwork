from sklearn.impute import KNNImputer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import activations
import pandas as pd


def load_data():
    file = "Genetic_SingletonEggsMatch_Udel.xlsx"
    shark_data_1 = pd.read_excel(file, sheet_name='Data 2023')
    shark_data_2 = pd.read_excel(file, sheet_name='Data 2021')
    shark_data_3 = pd.read_excel(file, sheet_name='Data 2022')
    shark_data_concat = pd.concat([shark_data_1, shark_data_2, shark_data_3])
    shark_data = shark_data_concat[['Record', 'Egg.ID', 'Local.ID', 'Female.by.Aquarist', 'Female.Mid.Confidence',
                                    'Institution', 'Exhibit', 'Female.Known', 'Yolk', 'Left.Total.Length', 'Right.Total.Length', 
                                    'Posterior.Apron.Length', 'Anterior.Apron.Length', 'Apron.to.Apron.length', 'Anterior.Waist.to.Posterior.Apron.Length', 
                                    'Hatching.End.Widest', 'Anterior.End.Width', 'Anterior.Width.Apron.Start', 'Width.Central.Waist.Max', 'Width.Central.Waist', 
                                    'Max.Width', 'Central.Body.Width.Max.Width', 'Posterior.Width.Apron.Start', 'Posterior.End.Width', 'Max.Depth', 
                                    'Top.Anterior.Respiratory.Slit', 'Top.Posterior.Respiratory.Slit', 'Bottom.Anterior.Respiratory.Slit', 
                                    'Bottom.Posterior.Respiratory.Slit', 'Shell.Thickness']]        

    # Clean up categorical data
    female_names = {'183': 0, '184': 1, '259': 2, '262': 3, '8': 4, 'Baby': 5, 'Blank': 6, 'Cleo': 7, 'Dalmatian': 8,
                    'Fern': 9,  'Gatsby': 10,  'Giraffe': 11,  'Lilly': 12,  'Mena': 13,  'SB173': 14,  'ST10': 15,
                    'Sierra': 16, 'Valentine': 17, 'Vera': 18,  'Yang': 19, 'nan': None, '': None, ' ': None}
    yolk_options = {'microyolk': 0, 'yes': 1, 'no': 2, 'nan': 3, '': 3, ' ': 3}

    # Replace Categorical Columns with Numerical Columns
    pd.set_option('mode.chained_assignment', None)
    female_num = [female_names[str(name).strip()] for name in shark_data['Female.Known']]
    shark_data['Yolk.Num'] = [yolk_options[str(name).strip().lower()] for name in shark_data['Yolk']]

    # Insert Female.Num into data
    shark_data['Female.Num'] = female_num

    return shark_data

def main():
    full_data_set = load_data()

    # Create a table with all of the train columns and drop the columns without known females. This will become our training dataset
    train_data = full_data_set[['Left.Total.Length', 'Right.Total.Length', 'Posterior.Apron.Length', 'Anterior.Apron.Length', 'Apron.to.Apron.length', 
                                'Anterior.Waist.to.Posterior.Apron.Length', 'Hatching.End.Widest', 'Anterior.End.Width', 'Anterior.Width.Apron.Start', 
                                'Width.Central.Waist.Max', 'Width.Central.Waist', 'Max.Width', 'Central.Body.Width.Max.Width', 'Posterior.Width.Apron.Start', 
                                'Posterior.End.Width', 'Max.Depth', 'Top.Anterior.Respiratory.Slit', 'Top.Posterior.Respiratory.Slit', 
                                'Bottom.Anterior.Respiratory.Slit', 'Bottom.Posterior.Respiratory.Slit', 'Shell.Thickness', 
                                'Yolk.Num', 'Female.Num']].dropna(subset='Female.Num')
    
    # Clean out all rows with more than 6 null values
    total_columns = len(train_data.columns)
    train_data_cleaned = train_data.dropna(thresh=total_columns - 6)

    '''
    In the cleaned data, which columns are still carrying a lot of null values? 

    null_counts = train_data_cleaned.isnull().sum()
    print(null_counts)
    
    Results of cleaned data null counts per column:

    Left.Total.Length                            0
    Right.Total.Length                           1
    Posterior.Apron.Length                       0
    Anterior.Apron.Length                        0
    Apron.to.Apron.length                       82      -- drop
    Anterior.Waist.to.Posterior.Apron.Length    81      -- drop
    Hatching.End.Widest                          5
    Anterior.End.Width                           1
    Anterior.Width.Apron.Start                   1
    Width.Central.Waist.Max                      0
    Width.Central.Waist                          0
    Max.Width                                    0
    Central.Body.Width.Max.Width                 0
    Posterior.Width.Apron.Start                  1
    Posterior.End.Width                          0
    Max.Depth                                    2
    Top.Anterior.Respiratory.Slit                3
    Top.Posterior.Respiratory.Slit               3
    Bottom.Anterior.Respiratory.Slit             4
    Bottom.Posterior.Respiratory.Slit            5
    Shell.Thickness                             86      -- drop
    '''

    train_data_cleaned_drop_cols = train_data_cleaned.drop(['Apron.to.Apron.length', 'Anterior.Waist.to.Posterior.Apron.Length', 'Shell.Thickness'], axis=1)

    # For the remaining features, we want to fill in our null values using KNN imputer. 
    # Accurately predict missing values by using the closest matching row information.
    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=6, weights='distance', metric='nan_euclidean')

    # Apply imputer to cleaned dataset
    train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data_cleaned_drop_cols), columns=train_data_cleaned_drop_cols.columns)
    
    # Split data into X for feature columns and Y for target column
    X = train_data_imputed.iloc[:, : -1].values
    y_base = train_data_imputed.iloc[:, -1].values

    # Convert labels to categorical one-hot encoding
    y = to_categorical(y_base, num_classes=20)  # 20 females to categorize

    # This block is to few the shape of the X and y to see how many training rows there are
    print("_____________________________________________________________")
    print("Shape:")
    print(X.shape, y.shape)
    print("_____________________________________________________________")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the MLP model
    # Input length is equivalent to the number of columns in the cleaned data
    model = Sequential([
        Dense(64, activation=activations.relu, input_shape=(len(train_data_cleaned_drop_cols.columns) - 1,)),
        Dropout(0.2),
        Dense(128, activation=activations.swish),
        Dropout(0.3),
        Dense(20, activation=activations.softmax)  # 20 neurons for 20 categories
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print('____________________________________________________________________________')

    # Next we run the whole dataset through the model and produce our predictions
    predict_dataset = full_data_set[['Left.Total.Length', 'Right.Total.Length', 'Posterior.Apron.Length', 'Anterior.Apron.Length',  
                                     'Hatching.End.Widest', 'Anterior.End.Width', 'Anterior.Width.Apron.Start', 'Width.Central.Waist.Max', 
                                     'Width.Central.Waist', 'Max.Width', 'Central.Body.Width.Max.Width', 'Posterior.Width.Apron.Start', 'Posterior.End.Width', 
                                     'Max.Depth', 'Top.Anterior.Respiratory.Slit', 'Top.Posterior.Respiratory.Slit', 'Bottom.Anterior.Respiratory.Slit', 
                                     'Bottom.Posterior.Respiratory.Slit', 'Yolk.Num']]
    # Clean out all rows with more than 6 null values
    total_columns_pre = len(predict_dataset.columns)
    predict_data_cleaned = predict_dataset.dropna(thresh=total_columns_pre - 6)
    
    # Apply imputer
    predict_data_imputed = pd.DataFrame(imputer.fit_transform(predict_data_cleaned), columns=predict_data_cleaned.columns)

    # Produce predictions
    predict_X = predict_data_imputed.iloc[:, :].values

    # Scale
    predict_X = scaler.fit_transform(predict_X)

    predictions_probabilities = model.predict(predict_X)

    # convert numeric predictions to female name
    female_numeric = {0: '183', 1: '184', 2: '259', 3: '262', 4: '8', 5: 'Baby', 6: 'Blank', 7: 'Cleo', 8: 'Dalmatian',
                    9: 'Fern', 10: 'Gatsby', 11: 'Giraffe', 12: 'Lilly', 13: 'Mena', 14: 'SB173', 15: 'ST10',
                    16: 'Sierra', 17: 'Valentine', 18: 'Vera', 19: 'Yang'}
    predictions = [female_numeric[pred.argmax()] for pred in predictions_probabilities]

    # put predictions into cleaned full dataset
    output_ds = full_data_set.dropna(subset=['Left.Total.Length', 'Right.Total.Length', 'Posterior.Apron.Length', 'Anterior.Apron.Length',  
                                             'Hatching.End.Widest', 'Anterior.End.Width', 'Anterior.Width.Apron.Start', 'Width.Central.Waist.Max', 
                                             'Width.Central.Waist', 'Max.Width', 'Central.Body.Width.Max.Width', 'Posterior.Width.Apron.Start', 
                                             'Posterior.End.Width', 'Max.Depth', 'Top.Anterior.Respiratory.Slit', 'Top.Posterior.Respiratory.Slit', 
                                             'Bottom.Anterior.Respiratory.Slit', 'Bottom.Posterior.Respiratory.Slit', 'Yolk.Num'], thresh=total_columns_pre - 6)
    output_ds['Predicted_Female'] = predictions

    # put new updated full dataset into excel file
    output_ds.to_excel("Results.xlsx", sheet_name="Data")


if __name__ == "__main__":
    main()
