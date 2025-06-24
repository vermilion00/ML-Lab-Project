import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import *
# from keras.api.layers import *
from keras.layers import *
import joblib
import matplotlib.pyplot as plt

#Label column index
FILENAME_INDEX = 0
LENGTH_INDEX = 1
LABEL_INDEX = 59
#Harmony, perceptr = [14-17]
#Tempo = 18
EXTENDED_DROPPED_IDX = [0, 1, 14, 15, 16, 17, 18, 59]
DROPPED_IDX = [0, 1, 59]

progress = 0
class Classifier:
    #MARK: Init
    #High epoch amount is fine since early stopping is available
    # def __init__(self, learning_rate=0.0009, epochs=1000, test_size=0.2, random_state=42, batch_size=40, patience=40):
    def __init__(self, learning_rate=0.00011, epochs=300, test_size=0.2, random_state=111, batch_size=20, patience=40):
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
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.patience = patience
        self.test_acc = None
        self.test_loss = None
        self.used_slow_features = None
        self.train_matrix = None
        self.test_matrix = None
        self.matrices = []
    
    #MARK: Prepare Data
    def prepareData(self, data_list, use_slow_features=True):
        #Encode the labels into integers
        df = pd.DataFrame(data_list)
        #Label encoder is initialised in the __init__ function
        df[LABEL_INDEX] = self.label_encoder.fit_transform(df[LABEL_INDEX])
        y = df[LABEL_INDEX]
        #Drop label, length and filename columns
        if use_slow_features:
            x = df.drop(DROPPED_IDX, axis=1)
        else:
            x = df.drop(EXTENDED_DROPPED_IDX, axis=1)
        columns = x.columns
        #The scaler is the MinMaxScaler
        scaled_data = self.scaler.fit_transform(x)
        #Save the scaled data as a dataframe
        x = pd.DataFrame(scaled_data, columns=columns)
        #Split the model into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        # x_train.shape, x_test.shape, y_train.shape, y_test.shape
        #Assign the split data to the model variables
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.used_slow_features = use_slow_features
        print("Prepared Data")

    # MARK: Build Neural
    def buildModel(self):
        model = Sequential()
        #Input shape is 57, since the table has 60 columns and we dropped 3
        # if self.used_slow_features:
        model.add(Input(shape=(self.x_train.shape[1],), batch_size=self.batch_size))
        # else:
        #     #Since we also dropped harmony, perceptr and tempo, we use 52 features
        #     model.add(Input(shape=(52,), batch_size=self.batch_size))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation='tanh'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=10, activation='softmax'))
        #Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )
        print("Compiled Model")
        self.model = model    

    #MARK: Build LSTM
    def buildModelLSTM(self):
        model = Sequential()
        self.patience = self.epochs/2
        #Input shape is 57, since the table has 60 columns and we dropped 3
        if self.used_slow_features:
            model.add(Input(shape=(self.x_train.shape[1], 1), batch_size=self.batch_size))
        else:
            #Since we also dropped harmony, perceptr and tempo
            model.add(Input(shape=(self.x_train.shape[1], 1), batch_size=self.batch_size))
        # model.add(Flatten())
        # model.add(LSTM(units=512, activation='relu'))
        # model.add(Dropout(rate=0.3))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(rate=0.2))
        # model.add(BatchNormalization())
        model.add(LSTM(units=128))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=10, activation='softmax'))
        #Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
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
                            # callbacks=[Callback()]
                            callbacks=[Callback(), early_stopping]
                        )
        print("Fitted model")
        #Run a test simulation to get an accuracy reading
        #Assign the accuracy and loss values to be able to save them later
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f'Test Accuracy: {self.test_acc*100:.2f}%')
        print(f'Test Loss: {self.test_loss:.4f}')
        #Generate the confusion matrices
        self.generateMatrix("neural")
        #Return the accuracy and loss values to display them in the gui
        return self.test_acc, self.test_loss
    
    #MARK: Train Cat
    def trainModelCat(self, depth=6, task_type="CPU"):
        global progress
        #Only import catboost when it's needed
        from catboost import CatBoostClassifier, Pool
        train_data = Pool(data=self.x_train, label=self.y_train)
        test_data = Pool(data=self.x_test, label=self.y_test)
        test = CatCallback()
        model = CatBoostClassifier(
                    iterations=self.epochs,
                    learning_rate=self.learning_rate,
                    depth=depth,
                    eval_metric='Accuracy',
                    loss_function='MultiClass',
                    early_stopping_rounds=200,
                    task_type=task_type
                )
        if task_type == "CPU":
            model.fit(train_data, eval_set=test_data, callbacks=[test])
        else:
            #User defined callbacks aren't supported on the GPU
            model.fit(train_data, eval_set=test_data)
        self.model = model
        y_pred = model.predict(self.x_test)
        #Get the accuracy and loss scores
        self.test_acc = accuracy_score(self.y_test, y_pred)
        # self.test_loss = log_loss(self.y_test, y_pred)
        self.test_loss = 0
        self.model = model
        #Generate the matrices
        self.generateMatrix("cat")
        #Set the progress to max in case of early stop
        progress = self.epochs
        #Return the accuracy and loss to display on the GUI
        return self.test_acc, self.test_loss

    #MARK: Predict Genre
    def predictGenre(self, data, model_type="neural", use_slow_features=True):
        print("Predicting Genres")
        stripped_data = []
        #Drop the filename, length and label columns + harmony and perceptr features
        try:
            #Loop through all selected files
            for i in data:
                #Remove the filename, length and label values
                if use_slow_features:
                    stripped_list = i[2:-1]
                #Leave out the slow features as well
                else:
                    stripped_list = i[2:14] + i[19:-1]
                #Shape the feature list
                shaped_list = np.array(stripped_list).reshape(1, -1)
                #Scale the features using the MinMaxScaler
                scaled_list = self.scaler.transform(shaped_list)
                #Add the feature list to the list of songs to predict
                stripped_data.append(scaled_list)
            #Predict the genre
            result_list = []
            for song_data in stripped_data:
                #Make the prediction
                match model_type:
                    case "cat":
                        prediction = self.model.predict(song_data, prediction_type="Probability")[0]
                    case _:
                        prediction = self.model.predict(song_data)[0]
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
                result_list.append(new_result)
            #If only one result is available, return the result as a string
            if len(result_list) == 1:
                #Double indexing because result_list is a tuple within a list within a list
                return f"{result_list[0][0][0]} with a probability of {result_list[0][0][1]}, {result_list[0][1:]}"
            #If more, then loop through all of them and prepend their number
            else:
                result_string = ""
                for idx, song in enumerate(result_list):
                    result_string += f"File {idx+1}: {song[0][0]} with a probability of {song[0][1]}, {result_list[idx][1:]}\n"
                #Cut off the last newline when returning
                return result_string[:-1]
        except Exception as e:
            print(f"Prediction failed: {e}")
            return e
        
    #MARK: Save Model
    def saveModel(self, file_path):
        #Combine the model, scaler and label encoder etc into one list
        obj_list = [self.model,
                    self.scaler,
                    self.label_encoder,
                    self.test_acc,
                    self.test_loss,
                    self.used_slow_features,
                    self.matrices]
        #Attempt to save the list
        try:
            #Save the list to the selected path
            joblib.dump(obj_list, file_path)
            return True
        except:
            return False

    #MARK: Load Model
    def loadModel(self, file_path):
        #Load the keras file containing the model, scaler and label encoder
        obj_list = joblib.load(file_path)
        #Assign the read objects to the respective variables
        self.model, self.scaler, self.label_encoder = obj_list[0], obj_list[1], obj_list[2]
        #Get the accuracy and loss from the saved list
        self.test_acc, self.test_loss, self.used_slow_features = obj_list[3], obj_list[4], obj_list[5]
        #Get the confusion matrices from the list
        self.matrices = obj_list[6]
    
    #MARK: Generate Matrix
    def generateMatrix(self, model_type="neural"):
        train_matrix = None
        test_matrix = None
        #Reset matrix list when new ones are being generated
        self.matrices = []
        #Since two matrices need to be generated, put their params in a list to loop through
        matrix_list = [(train_matrix, self.x_train, self.y_train),
                       (test_matrix, self.x_test, self.y_test)]
        #Loop through the list
        for matrix, x, y in matrix_list:
            #Generate the prediction
            match model_type:
                case "neural":
                    y_pred = self.model.predict(x).argmax(axis=1)
                case "cat":
                    y_pred = self.model.predict(x, prediction_type="Probability").argmax(axis=1)
            #Capitalize the class names
            labels = [item.capitalize() for item in self.label_encoder.classes_]
            #Calculate the confusion matrix
            matrix_data = confusion_matrix(y, y_pred)
            matrix = ConfusionMatrixDisplay(matrix_data, display_labels=labels)
            #Save the confusion matrix to class variable
            self.matrices.append(matrix)

    #MARK: Show Matrix
    def showMatrix(self):
        try:
            #Plot the training matrix
            self.matrices[0].plot(xticks_rotation='vertical', cmap='RdPu')
            plt.title("Training Confusion Matrix")
            #Plot the testing matrix
            self.matrices[1].plot(xticks_rotation='vertical', cmap='RdPu')
            plt.title("Testing Confusion Matrix")
            #Show the matrices on screen
            plt.show()
        except Exception as e:
            print(f"Failed to show confusion matrices. \n{e}")


#MARK: Callback class
class Callback(callbacks.Callback):
    # def __init__(self):
    #     pass
        # self.test = 0

    #Increment the progress counter after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # self.progress += 1
        global progress
        progress += 1
        # return super().on_epoch_end(epoch, logs)

class CatCallback:
    def after_iteration(self, info):
        global progress
        progress += 1
        return True
