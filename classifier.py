import pandas as pd
import tensorflow as tf
from numpy import array as np_array
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import *
from keras.api.layers import *

#Label column index
FILENAME_INDEX = 0
LENGTH_INDEX = 1
LABEL_INDEX = 59

progress = 0
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
        # self.history = None
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
    
    def prepareData(self, data_list):
        #Encode the labels into integers
        df = pd.DataFrame(data_list)
        #Label encoder is initialised in the __init__ function
        df[LABEL_INDEX] = self.label_encoder.fit_transform(df[LABEL_INDEX])
        y = df[LABEL_INDEX]
        #Drop label, length and filename columns
        #Also remove the harmony and perceptr columns while they can't be extracted
        #TODO: Remove the last indexes when harmony and perceptr can be extracted
        x = df.drop([FILENAME_INDEX, LENGTH_INDEX, LABEL_INDEX, 14, 15, 16, 17], axis=1)
        #Scale the data
        columns = x.columns
        #The scaler is the MinMaxScaler
        scaled_data = self.scaler.fit_transform(x)
        #Save the scaled data as a dataframe
        x = pd.DataFrame(scaled_data, columns=columns)
        #Split the model into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        #TODO: What does this do?
        x_train.shape, x_test.shape, y_train.shape, y_test.shape
        #Assign the split data to the model variables
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        print("Prepared Data")

    def buildModel(self):
        model = Sequential()
        #Input shape is 53, since the table has 60 columns and we dropped 7
        #TODO: Play around with layers
        #TODO: When perceptr and harmony are being used, set input shape to (57,)
        model.add(Input(shape=(53,), batch_size=self.batch_size))
        # model.add(Input((57,), batch_size=self.batch_size))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=10, activation='softmax'))
        # model.summary()
        #Compile the model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )
        print("Compiled Model")
        self.model = model
    
    def trainModel(self):
        #Fit the model with the parameters set in the gui
        history = self.model.fit(
                    x=self.x_train,
                    y=self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(self.x_test, self.y_test),
                    callbacks=[Callback()]
                )
        print("Fitted model")
        #Run a test simulation to get an accuracy reading
        test_error, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f'Accuracy: {test_accuracy}')
        print(f'Error: {test_error}')

    def predictGenre(self, data):
        print("Predicting Genres")
        stripped_data = []
        #Drop the filename, length and label columns + harmony and perceptr features
        try:
            #Loop through all selected files
            for i in data:
                #Remove the filename, length, harmony, perceptr and label values
                stripped_list = i[2:14] + i[18:-1]
                #Shape the features to (-1, 1)
                shaped_list = np_array(stripped_list).reshape(1, -1)
                #Scale the features using the MinMaxScaler
                scaled_list = self.scaler.transform(shaped_list)
                #Add the feature list to the list of songs to predict
                stripped_data.append(scaled_list)
            #Predict the genre
            #TODO: Currently only predicts the first in the list
            result_list = []
            for song_data in stripped_data:
                prediction = self.model.predict(x=song_data)[0]
                prediction = [f"{item*100:.2f}%" for item in prediction]
                #Multiply prediction by 100 to get the percentage
                result = list(zip(self.label_encoder.classes_, prediction))
                result.sort(key=lambda a: a[1], reverse=True)
                print(f"Predicted Genres: {result}")
                #TODO: When the result is displayed in a better way, rework this
                #Return only the most likely result to display on the hint text
                result_list.append(result[0])
            #If only one result is available, return the result as a string
            if len(result_list) == 1:
                return f"{result_list[0][0]} with a probability of {result_list[0][1]}"
            #If more, then loop through all of them and prepend their number
            else:
                result_string = ""
                for idx, genre in enumerate(result_list):
                    result_string += f"{idx+1}: {genre[0]} with a probability of {genre[1]}\n"
                #Cut off the last newline when returning
                return result_string[:-1]
        except:
            print("Prediction failed, likely due to a wrong format")
    
    def loadModel(self, file_path):
        self.model = saving.load_model(file_path)
        # return saving.load_model(file_path)
    
class Callback(callbacks.Callback):
    def __init__(self):
        self.progress = 0

    #Increment the progress counter after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # self.progress += 1
        global progress
        progress += 1
        # return super().on_epoch_end(epoch, logs)

    #Reset the progress counter when the training is finished
    # def on_train_end(self, logs=None):
    #     global progress
    #     progress = 0
    #     # return super().on_train_end(logs)
