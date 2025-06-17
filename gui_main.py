#Show spectrogram of file to be classified?
#Allow training the model via gui
#Input training parameters, add button to reset to default params
#Choose between different models
#Use askopenfilenames to allow selecting multiple files, then iterate through all of them

# TODO:
# What is harmony and perceptr?
# How is the duration calculated in the dataset?
# Is it important that the calculated values differ slightly from the dataset values?
# Writing 100 rows of csv is instant, but maybe add progress bar to saveCSV function
# Rewrite functions to not need global variables
# Play around with ttk styling
# Glob files in single lookup instead of looping over each ending
# Fix being able to scroll up past first path for some reason
# Add option to append to data with new selection, instead of overwriting it
# Add option to show waveplots/spectrograms
# Add button to save new default parameters
# Add button to show/hide model parameter options
# Add a message showing which files have been extracted
# Add a message showing that the model is loaded or not

from constants import *
import classifier
from numpy import mean, var
import pandas as pd
import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkscrolledframe import ScrolledFrame
import threading
import csv
import librosa
from librosa import feature
from glob import glob

#TODO: Remove this if not needed
# #Make save folder if it doesn't exist
# from os import makedirs, path, remove, name
# if not path.exists(SAVE_DIR):
#     makedirs(SAVE_DIR)
# #Check if running on windows
# if name == 'nt':
#     #Only import this when it's needed
#     from ctypes import windll
#     #Set folder to be hidden
#     ret = windll.kernel32.SetFileAttributesW(SAVE_DIR, 0x02)

#Function list for use in startExtraction function
FEATURE_FUNCTION_LIST = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
    #TODO:
    #harmony is not a thing, librosa.effects.harmonic?
    #What is perceptr?
]

#MARK: Threading events
extract_flag = threading.Event()
load_file_flag = threading.Event()
save_file_flag = threading.Event()
train_model_flag = threading.Event()
predict_genre_flag = threading.Event()
save_model_flag = threading.Event()
load_model_flag = threading.Event()

#MARK: Progress func
#Handles showing and hiding the progress bar as well as updating it
def updateProgress(process, progress):
    global total_progress
    global paths
    global model_available
    #Check if this is the first call for this progress
    if progress == 1:
        match process:
            case "extraction":
                #Set total progress to the amount of steps per path * amount of paths
                total_progress = EXTRACTION_STEPS * len(paths)
                progress_text.set("Running extraction: ")
                #Lock the extraction button for the duration of the process
                extract_button.config(state=DISABLED)
                #Lock the model buttons for the duration of the process
                train_model_button.config(state=DISABLED)
                predict_genre_button.config(state=DISABLED)
            case "training":
                #Set total progress to the amount of steps per path * amount of paths
                total_progress = epochs.get()
                progress_text.set("Training model: ")
                #Lock the model buttons for the duration of the process
                train_model_button.config(state=DISABLED)
                predict_genre_button.config(state=DISABLED)
        #Replace the hint label with the progress label and bar
        hint_label.pack_forget()
        progress_frame.pack(side=BOTTOM, padx=2, pady=(0,2), fill=X)
        progress_number.set("0%")

    num = int(progress/total_progress*100)
    #Process is still ongoing
    if num < 100:
        progress_number.set(str(num) + '%')
        progress_bar['value'] = num
    #Process is done
    else:
        #Limit the max progress value to 100
        # progress_number.set("100%")
        # progress_bar["value"] = 100
        #Sleep for a short time keep the progress completed messages up
        # time.sleep(0.5)
        #Hide progress label and bar
        progress_frame.pack_forget()
        #Set the hint label to say that the process is done
        hint_text.set(HINT_TEXT[process])
        #Show the hint label again
        hint_label.pack(padx=2, pady=(0,2), fill=X)
        #Reset the progress
        # progress = 0
        #Do something when process is finished
        match process:
            case "extraction":
                #Unlock save button only after extraction has finished
                save_button.config(state=NORMAL)
                #Unlock extract button again
                extract_button.config(state=NORMAL)
                #Unlock the model buttons for the duration of the process
                train_model_button.config(state=NORMAL)
                #Only unlock the prediction button if a model is loaded
                if model_available:
                    predict_genre_button.config(state=NORMAL)
            case "training":
                #Unlock the model buttons after the process
                train_model_button.config(state=NORMAL)
                predict_genre_button.config(state=NORMAL)

#MARK: Thread 1 handler
#Handles several functions running on one thread
#Since all these functions can't run concurrently, they can share a thread
def thread1_handler():
    global file_type
    global result_list
    while True:
        if load_file_flag.is_set():
            load_file_flag.clear()
            match file_type:
                case "audio":
                    loadAudio()
                case "csv":
                    loadCSV()
                case "folder":
                    loadFolder()
        elif extract_flag.is_set():
            extract_flag.clear()
            startExtraction()
        elif save_file_flag.is_set():
            save_file_flag.clear()
            saveCSV()
        elif train_model_flag.is_set():
            train_model_flag.clear()
            trainModelHelper(result_list)
        elif predict_genre_flag.is_set():
            predict_genre_flag.clear()
            hint_text.set(c.predictGenre(result_list))
        elif save_model_flag.is_set():
            save_model_flag.clear()
            saveModel(c.model)
        elif load_model_flag.is_set():
            load_model_flag.clear()
            loadModelHelper()
        #Add other functions that run on the same thread as elif
        else:
            #If thread is not needed, sleep for 100ms before polling again
            time.sleep(0.1)

#MARK: Thread 2 Handler
def thread2_handler():
    global progress
    local_extraction_progress = 0
    local_training_progress = 0
    progress = 0
    while True:
        if progress != local_extraction_progress:
            local_extraction_progress = progress
            updateProgress("extraction", progress)
        elif classifier.progress != local_training_progress:
            local_training_progress = classifier.progress
            updateProgress("training", classifier.progress)
        else:
            time.sleep(0.1)

#MARK: Load helper
#A helper function, since I can't call a function and set the flag in a button action
def loadFileHelper(type):
    global file_type
    file_type = type
    load_file_flag.set()

#MARK: Load audio
def loadAudio():
    global paths
    global already_extracted
    #Which file types should be selectable?
    filetypes = (
        ('Audio files', '*.wav *.mp3 *.flac *.ogg *.mat'),
        ('All files', '*.*')
    )
    #Opens an explorer window and returns the file path and names
    file_paths = fd.askopenfilenames(
        title = 'Select the audio file(s) you wish to load',
        filetypes = filetypes
    )
    #If multiple files are added, show them separated by newlines
    #Only update file path if selection is made
    #Check for empty string since it returns nothing when cancelled, instead of an empty list
    if file_paths != "":
        file_path_str = ""
        #Add newlines at the end of each path for the gui display
        for i in file_paths:
            file_path_str += i + "\n"
        #Remove the last newline character
        if len(file_path_str) > 0:
            file_path_str = file_path_str[:-1]
        file_path.set(file_path_str)
        #Save the tuple globally to make iterating easier
        paths = file_paths
        #Since new data is available, reset extracted flag
        already_extracted = False
        #Enable the extract button, since audio paths are now available
        extract_button.config(state=NORMAL)
        #Disable the save button, to avoid confusion, since the new audio files haven't been extracted
        save_button.config(state=DISABLED)
        #Lock the model buttons, to avoid confusion, since the new audio files haven't been extracted
        train_model_button.config(state=DISABLED)
        predict_genre_button.config(state=DISABLED)

#MARK: Load CSV
def loadCSV():
    global paths
    filetypes = (
        ('Pre-extracted features', '*.csv'),
        ('All files', '*.*')
    )
    #Only allow selecting one file here
    file_path_str = fd.askopenfilename(
        title = 'Select the pre-extracted feature file you wish to classify',
        filetypes = filetypes
    )
    if file_path_str != "":
        file_path.set(file_path_str)
        paths = []
        #Reset selected audio file paths if csv is selected
        readCSV(file_path_str)

#MARK: Load Folder
def loadFolder():
    global paths
    global already_extracted
    #Unfortunately tkinter can't open a window to select either a file or a folder
    file_path_str = fd.askdirectory(
        title='Select the directory containing the audio files you wish to classify'
    )
    #Check if the file path isn't empty
    if file_path_str != "":
        file_path_str += '/'
        # file_path.set(file_path_str)
        file_paths = []
        #Look for all allowed filetypes in the folder and add them to path list
        #TODO: glob files in single lookup instead of looping over each ending
        for filetype in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
            file_paths += glob(file_path_str + filetype)
        #If file path list is not empty, save and display new paths
        if file_paths != []:
            temp_paths = []
            #Replace the \ with a / for aesthetic reasons
            for i in file_paths:
                new_path = i.replace('\\', '/')
                temp_paths.append(new_path)
            file_paths = temp_paths
            #Save the tuple globally to make iterating easier
            paths = file_paths
            #Since new data is available, reset extracted flag
            already_extracted = False
            #Enable the extract button, since audio paths are now available
            extract_button.config(state=NORMAL)
            #Disable the save button, to avoid confusion, since the new audio files haven't been extracted
            save_button.config(state=DISABLED)
            #Lock the model buttons, to avoid confusion, since the new audio files haven't been extracted
            train_model_button.config(state=DISABLED)
            predict_genre_button.config(state=DISABLED)

            file_path_str = ""
            #Add newlines add the end of each path for the gui display
            for i in file_paths:
                file_path_str += i + "\n"
            #Remove the last newline character
            if len(file_path_str) > 0:
                file_path_str = file_path_str[:-1]  
            #Set the file path string in the gui
            file_path.set(file_path_str)
        #If file path is empty, look for csv file instead
        else:
            #Look for the csv file in the directory
            file_paths += glob(file_path_str + '*.csv')
            #Check if a csv file has been found
            if file_paths != []:
                #Reset audio file paths since features are now read from csv instead
                paths = []
                #Only take the first csv, since all tracks should be in one
                file_path_str = file_paths[0].replace('\\', '/')
                file_paths = list(file_path_str)
                #Read the contents of the csv file into memory
                readCSV(file_path_str)
                #Set the file path string in the gui
                file_path.set(file_path_str)
            #No audio or csv files found
            else:
                #Look if folder contains subfolders, empty list if none available
                if glob(file_path_str) != []:
                    #If no audio files are detected but subfolders are available,
                    #ask if they should be scanned recursively to add all audio to path
                    if mb.askyesno(title="Search subfolders?", message=SEARCH_SUBFOLDERS_MSG):
                        #Answered yes
                        #TODO: Glob once instead of once for each filetype
                        for filetype in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
                            file_paths += glob(file_path_str + '**/*' + filetype, recursive=True)
                        #If file path list is not empty, save and display new paths
                        if file_paths != []:
                            temp_paths = []
                            #Replace the \ with a / for aesthetic reasons
                            for i in file_paths:
                                new_path = i.replace('\\', '/')
                                temp_paths.append(new_path)
                            file_paths = temp_paths
                            #Save the tuple globally to make iterating easier
                            paths = file_paths
                            #Since new data is available, reset extracted flag
                            already_extracted = False
                            #Enable the extract button, since audio paths are now available
                            extract_button.config(state=NORMAL)
                            #Disable the save button, to avoid confusion, since the new audio files haven't been extracted
                            save_button.config(state=DISABLED)
                            #Lock the model buttons, to avoid confusion, since the new audio files haven't been extracted
                            train_model_button.config(state=DISABLED)
                            predict_genre_button.config(state=DISABLED)
                            
                            file_path_str = ""
                            #Add newlines add the end of each path for the gui display
                            for i in file_paths:
                                file_path_str += i + "\n"
                            #Remove the last newline character
                            if len(file_path_str) > 0:
                                file_path_str = file_path_str[:-1]
                            #Set the file path string in the gui
                            file_path.set(file_path_str)
                        #No files found in subfolders
                        else:
                            mb.showwarning(title="No files found", message="No files have been found in all subfolders.")

                hint_text.set(HINT_TEXT["no_files_found"])

#MARK: Read CSV
#Reads the features in a csv into memory
def readCSV(csv_path):
    global result_list
    global already_saved
    global model_already_trained
    #Reset the result list
    result_list = []
    #Catch possible error
    #TODO: Fix
    paths = []
    try:
        #Open the csv file
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #Basic check if csv is formed correctly
                #TODO: Implement more checks
                if len(row) != len(HEADER):
                    print(MALFORMED_CSV_MSG)
                    mb.showerror(title="Malformed CSV", message=MALFORMED_CSV_MSG)
                    #Reset the already made result list, since it's malformed
                    result_list = []
                    #Break out of loop to abort reading csv
                    break
                #Ignore header row
                elif line_count != 0:
                    #Each row is a list of all features of that track, so just add it to list
                    result_list.append(row)
                line_count += 1
            #Show a message that the features have been loaded
            hint_text.set(HINT_TEXT["read_csv"])
            #Since new features have been loaded, reset the saved flag and trained flag
            already_saved = False
            model_already_trained = False
            #Enable the save button, since features are now available to save
            save_button.config(state=NORMAL)
            #Disable the extract button, since old audio files have now been unloaded
            extract_button.config(state=DISABLED)
            #Unlock the model buttons, since data is now available
            train_model_button.config(state=NORMAL)
            predict_genre_button.config(state=NORMAL)
    except:
        #Error when opening the csv file
        print(READ_CSV_FAILED_MSG)
        mb.showerror(title="Failed to read file", message=READ_CSV_FAILED_MSG)
    #TODO: Remove this
    # print(result_list)


#MARK: Save csv
#Opens file dialog to choose the directory and name to save to
def saveCSV():
    global already_saved
    global result_list
    #If the features have already been saved, ask if they should be saved again
    if already_saved and not mb.askyesno(title="Save again?", message=ALREADY_SAVED_MSG):
        #If the answer is No, abort the function call
        return
    if result_list != []:
        filetype = [('CSV file', '*.csv'),
                    ('All files', '*.*')]
        #Set the title of the folder window
        title = "Choose a directory and file name to save the feature list to"
        #Set the default name of the saved file
        default_name = "features.csv"
        filename = fd.asksaveasfilename(filetypes=filetype, defaultextension=filetype, title=title, initialfile=default_name)
        #If filename is empty, the save prompt has been canceled
        if filename != "":
            #Catch errors when writing to file
            try:
                #Open save file in write mode, creates file if it doesn't exist
                with open(filename, '+w', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    #Header is set in constants.py to avoid clutter
                    writer.writerow(HEADER)
                    writer.writerows(result_list)
                    print("Done writing file")
                #Set the saved flag
                already_saved = True
            except:
                print(OVERWRITE_FAILED_MSG)
                mb.showerror(title="Failed to save file", message=OVERWRITE_FAILED_MSG)
    else:   #If no data is available to save
        print(NO_DATA_MSG)
        mb.showerror(title="No data available", message=NO_DATA_MSG)

#MARK: Save Model
def saveModel(model):
    global model_already_saved
    #If the model has already been saved, ask if it should be saved again
    if model_already_saved and not mb.askyesno(title="Save again?", message=MODEL_ALREADY_SAVED_MSG):
        #If the answer is No, abort the function call
        return
    filetype = [('Keras model file', '*.keras'),
                ('All files', '*.*')]
    #Set the title of the folder window
    title = "Choose a directory and file name to save the model to"
    #Set the default name of the saved file
    default_name = "model.keras"
    filename = fd.asksaveasfilename(filetypes=filetype, defaultextension=filetype, title=title, initialfile=default_name)
    #If filename is empty, the save prompt has been canceled
    if filename != "":
        #Catch errors when writing to file
        try:
            #Save the model to the selected file path
            model.save(filename)
            #Set the saved flag
            model_already_saved = True
            hint_text.set("Saved model")
        except:
            #TODO: Adjust this message
            print(OVERWRITE_FAILED_MSG)
            mb.showerror(title="Failed to save file", message=OVERWRITE_FAILED_MSG)

#MARK: Load Model
def loadModelHelper():
    global model_already_saved
    global model_available
    filetypes = (
        ('Keras model file', '*.keras'),
        ('All files', '*.*')
    )
    #Only allow selecting one file here
    file_path_str = fd.askopenfilename(
        title = 'Select the keras model file you wish to load',
        filetypes = filetypes
    )
    #Check if a file has been selected
    if file_path_str != "":
        try:
            #Load the selected model
            c.model = c.loadModel(file_path_str)
            #Reset the Model Saved flag
            model_already_saved = False
            #Set the model_available flag
            model_available = True
            hint_text.set(f"Loaded model \"{file_path_str}\"")
        except:
            print("Failed to load model")
            mb.showerror(title="Failed to load model", message=MODEL_LOADING_FAILED_MSG)

#MARK: Extract
#Extracts the features from the loaded audio files
def startExtraction():
    global paths
    global result_list
    global already_extracted
    global model_already_trained
    global progress
    #If the files have already been extracted, ask if they should be extracted again
    if already_extracted and not mb.askyesno(title="Extract again?", message=ALREADY_EXTRACTED_MSG):
        #If the answer is no, abort the function call
        return
    #Check if audio files are selected
    #TODO: Check if paths can be a tuple here - prob not relevant since button is locked anyway
    if paths != []:
        result_list = []
        #Iterate over every selected path and extract the audio features
        for i in paths:
            #Load the audio file using librosa
            #y is a time-series-array, sr is the sample rate
            try:
                y, sr = librosa.load(i, sr=None)
                #Update the progress text and bar (needs to be called each progress step)
                # updateProgress("extraction")
                progress += 1
                #Loop through list of functions and add result to list
                feature_list_mean = [mean(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
                feature_list_var = [var(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
                # updateProgress("extraction")
                progress += 1
                #Calculate the mfccs separately, as it returns a list of 20 results, 
                #for which we need to calculate mean and var separately
                feature_list_mean += [mean(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
                feature_list_var += [var(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
                # updateProgress("extraction")
                progress +=1
                #Make a feature list, extract the filename from the path, and get length
                feature_list = [i[i.rindex('/')+1:], librosa.get_duration(y=y, sr=sr)]
                #Combine the lists so that mean and var alternate
                feature_list += [feat for pair in zip(feature_list_mean, feature_list_var) for feat in pair]
                # updateProgress("extraction")
                progress += 1
                #Insert the rms, tempo and crossing rate values here, since they don't take the sr as a parameter
                feature_list.insert(4, mean(feature.rms(y=y)))
                feature_list.insert(5, var(feature.rms(y=y)))
                feature_list.insert(12, mean(feature.zero_crossing_rate(y=y)))
                feature_list.insert(13, var(feature.zero_crossing_rate(y=y)))
                # updateProgress("extraction")
                progress += 1
                #Use index 0 since function gives back a list with one element
                feature_list.insert(14, feature.tempo(y=y, sr=sr)[0])
                #TODO: Remove this
                #Since harmony and perceptr can't currently be extracted, set them to 0 to match dataset length
                for i in range(4): feature_list.insert(14, 0.0)
                # updateProgress("extraction")
                progress += 1
                #Get the label from the filename
                feature_list.append(feature_list[0][:feature_list[0].index('.')])
                #Append feature list of current track to complete result list, incase multiple tracks are selected
                result_list.append(feature_list)
                # updateProgress("extraction")
                progress += 1
                #Set the extracted flag
                already_extracted = True
                #Since new data is available that the model can be trained on, clear the flag
                model_already_trained = False
            #A file in the list could not be opened
            except:
                #Add the progress, since the updateProgress function expects 7 more calls to happen
                progress += EXTRACTION_STEPS
                print(f"A file at {i} could not be opened.")
                mb.showerror(title="Invalid file", message=f"The file \"{i[i.rindex('/')+1:]}\" at path \"{i[:i.rindex('/')+1]}\" could not be opened. The extraction will continue without it.")
    #No files are selected
    else:
        mb.showerror(title="No files selected", message=NO_FILES_MSG)
        print(NO_FILES_MSG)

#MARK: Train Model
def trainModelHelper(result_list):
    global model_already_saved
    global model_already_trained
    global model_available
    params_changed = False
    try:
        #Check if the parameters have been changed
        # if learning_rate.get() != c.learning_rate or c.epochs != epochs.get() or c.batch_size != batch_size.get() or c.test_size != test_size.get() or c.random_state != random_state.get():
        #Update the parameters with the respective entry values
        if learning_rate.get() != c.learning_rate:
            c.learning_rate = learning_rate.get()
            params_changed = True
        if epochs.get() != c.epochs:
            c.epochs = epochs.get()
            params_changed = True
        if batch_size.get() != c.batch_size:
            c.batch_size = batch_size.get()
            params_changed = True
        if test_size.get() != c.test_size:
            c.test_size = test_size.get()
            params_changed = True
        if random_state.get() != c.random_state:
            c.random_state = random_state.get()
            params_changed = True
        #If the parameters haven't been changed and the model is still loaded, ask if it should be retrained
        if model_already_trained and not params_changed and not mb.askyesno(title="Train model again?", message=MODEL_ALREADY_TRAINED_MSG):
            #Abort the function call if the model has been trained and the answer is no
            return

    except:
        print(INVALID_INPUT_MSG)
        #Show a warning box if an input field contains invalid input
        mb.showerror(title="Invalid input", message=INVALID_INPUT_MSG)
        #Abort the function call
        return
    #Prep the data
    c.prepareData(result_list)
    #Build the model
    c.buildModel()
    #TODO: Update progress bar, lock buttons
    #Post training updates in path screen, update descriptor
    #Update wrap length of path screen according to dimensions
    #Train the model
    c.trainModel()
    #Enable the Save Model button
    save_model_button.config(state=NORMAL)
    hint_text.set("Model training complete")
    #Reset the saved model flag, since it's been trained again
    model_already_saved = False
    #Set the model available flag
    model_available = True
    #Set the model already trained flag
    model_already_trained = True
    #Reset the training progress counter
    classifier.progress = 0

#MARK: Wraplength
#Set the wraplength of the label based on the window size
def setWraplength(event):
    #Subtract 6 from the width for padding
    event.widget.configure(wraplength=event.width-6)

#MARK: Threading
#Put long functions on a different thread so the GUI can update still
thread1 = threading.Thread(target=thread1_handler, daemon=True)
thread1.start()
thread2 = threading.Thread(target=thread2_handler, daemon=True)
thread2.start()

#MARK: Tk config
#The base state of every gui element is configured here
root = Tk()
root.title('Music Genre Classifier')
#Set the minimum size so that all vital elements are still visible
root.minsize(width=428, height=221)
root.geometry("428x290")

c = classifier.Classifier()
file_path = StringVar()
paths = []
result_list = []
file_type = ""
#Initialise the hint text with the startup message
hint_text = StringVar(value=HINT_TEXT["program_start"])
progress_text = StringVar()
progress_number = StringVar()
#Progress bar progress
progress = 0
total_progress = 0
#already done flags to ask if they should happen again
already_extracted = False
already_saved = False
model_already_saved = False
model_already_trained = False
model_available = False
#Model parameter variables
learning_rate = DoubleVar(value=c.learning_rate)
epochs = IntVar(value=c.epochs)
test_size = DoubleVar(value=c.test_size)
batch_size = IntVar(value=c.batch_size)
random_state = IntVar(value=c.random_state)

#Load a model at startup if it is saved in the starting directory
try:
    #If multiple models are in the starting directory, load the first one found
    model_path = glob("*.keras")[0]
    c.model = c.loadModel(model_path)
    hint_text.set(f"Loaded model \"{model_path}\" from the default directory.")
    print(f"Loaded model \"{model_path}\" from the default directory.")
    #Set the model available flag
    model_available = True
#No model found/malformed file
except:
    pass

#MARK: Buttons
#Make a new frame to group the related buttons together
button_frame = ttk.Frame(root, padding="2 0 2 0")
button_frame.pack(anchor=W)
#Create a button widget
load_file_button = ttk.Button(button_frame, text="Load Audio", command=lambda:loadFileHelper("audio"), padding="2 2 2 2")
#Set an event to happen if the mouse cursor hovers over the load file button
load_file_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_file_button"]))
#Pack the button into the button frame with some padding
load_file_button.pack(side=LEFT, padx=1, pady=2)
#Same procedure for the other buttons
load_folder_button = ttk.Button(button_frame, text="Load Folder", command=lambda:loadFileHelper("folder"), padding="2 2 2 2")
load_folder_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_folder_button"]))
load_folder_button.pack(side=LEFT, padx=1, pady=2)
load_csv_file_button = ttk.Button(button_frame, text="Load Features", command=lambda:loadFileHelper("csv"), padding="2 2 2 2")
load_csv_file_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_csv_file_button"]))
load_csv_file_button.pack(side=LEFT, padx=1, pady=2)
extract_button = ttk.Button(button_frame, text="Extract features", command=extract_flag.set, padding="2 2 2 2", state=DISABLED)
extract_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["extract_button"]))
extract_button.pack(side=LEFT, padx=1, pady=2)
save_button = ttk.Button(button_frame, text="Save CSV", command=save_file_flag.set, padding="2 2 2 2", state=DISABLED)
save_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["save_button"]))
save_button.pack(side=LEFT, padx=1, pady=2)

#MARK: Model Entries
model_frame = ttk.Frame(root, padding="0 2 0 2")
model_frame.pack(anchor=NW)
model_entry_frame = ttk.Frame(model_frame, padding="2 2 2 2")
model_entry_frame.pack()
#Learning rate elements
learning_rate_frame = ttk.Frame(model_entry_frame, padding="2 2 2 2")
learning_rate_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["learning_rate"]))
learning_rate_frame.pack(side=LEFT)
learning_rate_label = ttk.Label(learning_rate_frame, text="Learning Rate").pack()
learning_rate_entry = ttk.Entry(learning_rate_frame, textvariable=learning_rate, width=12, justify=CENTER).pack()
#Epochs elements
epochs_frame = ttk.Frame(model_entry_frame, padding="2 2 2 2")
epochs_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["epochs"]))
epochs_frame.pack(side=LEFT)
epochs_label = ttk.Label(epochs_frame, text="Epochs").pack()
epochs_entry = ttk.Entry(epochs_frame, textvariable=epochs, width=12, justify=CENTER).pack()
#Test size elements
test_size_frame = ttk.Frame(model_entry_frame, padding="2 2 2 2")
test_size_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["test_size"]))
test_size_frame.pack(side=LEFT)
test_size_label = ttk.Label(test_size_frame, text="Test Size").pack()
test_size_entry = ttk.Entry(test_size_frame, textvariable=test_size, width=12, justify=CENTER).pack()
#Batch size elements
batch_size_frame = ttk.Frame(model_entry_frame, padding="2 2 2 2")
batch_size_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["batch_size"]))
batch_size_frame.pack(side=LEFT)
batch_size_label = ttk.Label(batch_size_frame, text="Batch Size").pack()
batch_size_entry = ttk.Entry(batch_size_frame, textvariable=batch_size, width=12, justify=CENTER).pack()
#Random state elements
random_state_frame = ttk.Frame(model_entry_frame, padding="2 2 2 2")
random_state_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["random_state"]))
random_state_frame.pack(side=LEFT)
random_state_label = ttk.Label(random_state_frame, text="Random State").pack()
random_state_entry = ttk.Entry(random_state_frame, textvariable=random_state, width=12, justify=CENTER).pack()

#MARK: Model buttons
model_button_frame = ttk.Frame(model_frame, padding="2 2 2 2")
model_button_frame.pack(anchor=N)
load_model_button = ttk.Button(model_button_frame, text="Load Model", command=load_model_flag.set)
load_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_model_button"]))
load_model_button.pack(side=LEFT)
train_model_button = ttk.Button(model_button_frame, text="Train Model", command=train_model_flag.set, state=DISABLED)
train_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["train_model_button"]))
train_model_button.pack(side=LEFT)
predict_genre_button = ttk.Button(model_button_frame, text="Predict Genre", command=predict_genre_flag.set, state=DISABLED)
predict_genre_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["predict_genre_button"]))
predict_genre_button.pack(side=LEFT)
save_model_button = ttk.Button(model_button_frame, text="Save Model", command=save_model_flag.set, state=DISABLED)
save_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["save_model_button"]))
save_model_button.pack(side=LEFT)

#MARK: Paths
#Scrolling path frame
file_frame = ttk.Frame(root, padding="2 0 2 0")
#Set the hint text on a mouse hover event
file_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["file_path_label"]))
file_frame.pack(padx=2, anchor=W, fill=BOTH, expand=True)
#This label shows the text above the paths
file_label = ttk.Label(file_frame, text="Selected file path(s):")
file_label.pack(side=TOP, anchor=W)
#Create the scrolled frame
file_path_frame = ScrolledFrame(file_frame, width=400, height=0)
#Needs a helper frame to display the contents in
file_path_helper_frame = file_path_frame.display_widget(Frame)
file_path_frame.pack(anchor=W, expand=True, fill=BOTH)
#This label shows the file paths
file_path_label = ttk.Label(file_path_helper_frame, textvariable=file_path)
#Bind scrolling events to the respective windows
file_path_frame.bind_scroll_wheel(file_path_frame)
file_path_frame.bind_scroll_wheel(file_path_label)
file_path_frame.bind_arrow_keys(root)
file_path_label.pack(padx=5, pady=2, expand=True)

#MARK: Hint label
#For labels with dynamic wrap length, justify=CENTER, anchor=N and fill=X needs to be set for it to be centered
hint_label = ttk.Label(root, textvariable=hint_text, justify=CENTER, anchor=N)
hint_label.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["hint_label"]))
hint_label.bind('<Configure>', setWraplength)
hint_label.pack(padx=2, pady=(0,2), fill=X)

#MARK: Progress
#Extra frame for the progress text, hidden as long as no process is running
progress_frame = ttk.Frame(root, padding="0 2 0 2")
progress_bar = ttk.Progressbar(progress_frame, style="TProgressbar")
progress_bar.pack(fill=X, padx=5)
#Extra frame to bundle the text labels together
progress_text_frame = ttk.Frame(progress_frame)
progress_text_frame.pack()
progress_text_label = ttk.Label(progress_text_frame, textvariable=progress_text, justify=CENTER)
progress_text_label.pack(side=LEFT)
progress_number_label = ttk.Label(progress_text_frame, textvariable=progress_number, justify=CENTER)
progress_number_label.pack(side=LEFT)

root.mainloop()

#TODO: Remove this if not needed
# rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# cbc = cb.CatBoostClassifier(verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
# xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# for clf in (rf, cbc, xgb):
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)