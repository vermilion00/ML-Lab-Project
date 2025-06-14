import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import *
from keras.api.layers import *

#Label column index
FILENAME_INDEX = 0
LENGTH_INDEX = 1
LABEL_INDEX = 59
class Classifier:
    def __init__(self, learning_rate=0.00011, epochs=120, test_size=0.1, random_state=111, batch_size=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
    
    def prepareData(self, data_list):
        #Encode the labels into integers
        label_encoder = LabelEncoder()
        df = pd.DataFrame(data_list)
        df[LABEL_INDEX] = label_encoder.fit_transform(df[LABEL_INDEX])
        y = df[LABEL_INDEX]
        #Drop label, length and filename columns
        #Also remove the harmony and perceptr columns while they can't be extracted
        #TODO: Remove the last indexes when harmony and perceptr can be extracted
        x = df.drop([FILENAME_INDEX, LENGTH_INDEX, LABEL_INDEX, 14, 15, 16, 17], axis=1)
        #Scale the data
        columns = x.columns
        minmax = MinMaxScaler()
        scaled_data = minmax.fit_transform(x)
        #Save the scaled data as a dataframe
        x = pd.DataFrame(scaled_data, columns=columns)
        #Split the model into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        #TODO: What does this do?
        x_train.shape, x_test.shape, y_train.shape, y_test.shape
        #TODO: Does this work?
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        print("Prepared Data")

    def buildModel(self):
        model = Sequential()
        #Input shape is 57, since the table has 60 columns and we dropped 3
        #TODO: Play around with layers
        model.add(Input((53,), batch_size=self.batch_size))
        # model.add(Input((57,), batch_size=self.batch_size))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        # model.summary()
        #Compile the model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
        print("Compiled Model")
        self.model = model
    
    def trainModel(self):
        history = self.model.fit(
                    x=self.x_train,
                    y=self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(self.x_test, self.y_test)
                )
        print("Fitted model")
        test_error, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=True)
        print(f'Accuracy: {test_accuracy}')
        print(f'Error: {test_error}')