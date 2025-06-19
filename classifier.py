import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import *
from keras.api.layers import *
import joblib

#Label column index
FILENAME_INDEX = 0
LENGTH_INDEX = 1
LABEL_INDEX = 59

progress = 0
class Classifier:
    #MARK: Init
    #High epoch amount is fine since early stopping is available
    def __init__(self, learning_rate=0.00011, epochs=300, test_size=0.1, random_state=111, batch_size=20, patience=40):
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
        self.history = None
        # self.history = None
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.patience = patience
        self.test_acc = None
        self.test_loss = None
    
    #MARK: Prepare Data
    def prepareData(self, data_list):
        #Encode the labels into integers
        df = pd.DataFrame(data_list)
        #Label encoder is initialised in the __init__ function
        df[LABEL_INDEX] = self.label_encoder.fit_transform(df[LABEL_INDEX])
        y = df[LABEL_INDEX]
        #Drop label, length and filename columns
        x = df.drop([FILENAME_INDEX, LENGTH_INDEX, LABEL_INDEX], axis=1)
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

    #MARK: Build Model
    def buildModel(self):
        model = Sequential()
        #Input shape is 57, since the table has 60 columns and we dropped 3
        #TODO: Play around with layers
        model.add(Input(shape=(57,), batch_size=self.batch_size))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(rate=0.3))
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
    
    #MARK: Train Model
    def trainModel(self):
        #Define an early stopping callback to avoid wasting time training without progress
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        #Fit the model with the parameters set in the gui
        self.history = self.model.fit(
                    x=self.x_train,
                    y=self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(self.x_test, self.y_test),
                    callbacks=[Callback(), early_stopping]
                    # callbacks=[Callback()]
                )
        print("Fitted model")
        #Run a test simulation to get an accuracy reading
        #Assign the accuracy and loss values to be able to save them later
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f'Test Accuracy: {self.test_acc*100:.2f}%')
        print(f'Test Loss: {self.test_loss:.4f}')
        #Return the accuracy and loss values to display them in the gui
        return self.test_acc, self.test_loss

    #MARK: Predict Genre
    def predictGenre(self, data):
        print("Predicting Genres")
        stripped_data = []
        #Drop the filename, length and label columns + harmony and perceptr features
        try:
            #Loop through all selected files
            for i in data:
                #Remove the filename, length and label values
                stripped_list = i[2:-1]
                #Shape the features to (-1, 1)
                shaped_list = np.array(stripped_list).reshape(1, -1)
                #Scale the features using the MinMaxScaler
                scaled_list = self.scaler.transform(shaped_list)
                #Add the feature list to the list of songs to predict
                stripped_data.append(scaled_list)
            #Predict the genre
            result_list = []
            for song_data in stripped_data:
                #Make the prediction
                prediction = self.model.predict(x=song_data)[0]
                #Combine the prediction with the corresponding class
                result = list(zip(self.label_encoder.classes_, prediction))
                #Sort the list so that the highest probability is first
                result.sort(key=lambda a: a[1], reverse=True)
                new_result = []
                for item in result:
                    #Format the probability to show a percentage
                    #Capitalize the genre because the labels are lower case
                    new_result.append((item[0].capitalize(), f'{item[1]*100:.2f}%'))
                print(f"Predicted Genres: {new_result}")
                #TODO: When the result is displayed in a better way, rework this
                #Return only the most likely result to display on the hint text
                result_list.append(new_result[0])
            #If only one result is available, return the result as a string
            if len(result_list) == 1:
                #Double indexing because result_list is a tuple within a list
                return f"Predicted Genre: {result_list[0][0]} with a probability of {result_list[0][1]}"
            #If more, then loop through all of them and prepend their number
            else:
                result_string = ""
                for idx, genre in enumerate(result_list):
                    result_string += f"File {idx+1}: {genre[0]} with a probability of {genre[1]}\n"
                #Cut off the last newline when returning
                return "Predicted Genres:\n" + result_string[:-1]
        except Exception as e:
            print(e)
            return e
    
    #MARK: Load Model
    def loadModel(self, file_path):
        #Load the keras file containing the model, scaler and label encoder
        obj_list = joblib.load(file_path)
        #Assign the read objects to the respective variables
        self.model, self.scaler, self.label_encoder = obj_list[0], obj_list[1], obj_list[2]
        #Get the accuracy and loss from the saved list
        self.test_acc, self.test_loss = obj_list[3], obj_list[4]
    
#MARK: Callback
class Callback(callbacks.Callback):
    def __init__(self):
        self.progress = 0

    #Increment the progress counter after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # self.progress += 1
        global progress
        progress += 1
        # return super().on_epoch_end(epoch, logs)

    #Incase the training is cancelled early, add to the progress so it doesn't break the program
    # def on_train_end(self, logs=None):
    #     global progress
    #     progress = 
        # return super().on_train_end(logs)
