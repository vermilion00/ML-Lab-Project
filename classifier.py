#Allow loading music file or pre-extracted features
#Show spectrogram of file to be classified?
#Allow training the model via gui
#Input training parameters, add button to reset to default params
#Choose between different models
#Use askopenfilenames to allow selecting multiple files, then iterate through all of them
#Save extracted features as csv, add button to save to specified location
#Add progress bar or progress text field

# TODO:
# What is harmony and perceptr?
# How is the duration calculated in the dataset?
# Is it important that the calculated values differ slightly from the dataset values
# Make the path area a scrollable area with a max height and set width
# Writing 100 rows of csv is instant, but maybe add progress bar to saveCSV function

from constants import *
import numpy as np
import pandas as pd
import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkscrolledframe import ScrolledFrame
import threading
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import librosa
from librosa import feature
from glob import glob

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

#MARK: Threading events
extract_flag = threading.Event()
load_file_flag = threading.Event()
save_file_flag = threading.Event()

#MARK: Progress func
def updateProgress(progress, total_progress):
    #Process name and 0% is set in function directly
    num = int(progress/total_progress*100)
    progress_number.set(str(num if num < 100 else 100) + '%')

#MARK: Thread 1 handler
#Handles several functions on one thread
#Since all these functions can't run concurrently, they can share a thread
def thread1_handler():
    global file_type
    while True:
        if load_file_flag.is_set():
            load_file_flag.clear()
            loadFiles(file_type)
        elif extract_flag.is_set():
            extract_flag.clear()
            startExtraction()
        elif save_file_flag.is_set():
            save_file_flag.clear()
            saveCSV()
        #Add other functions that run on the same thread as elif
        else:
            #If thread is not needed, sleep for 100ms before polling again
            time.sleep(0.1)

#MARK: Load helper
#A helper function, since I can't call a function and set the flag in a button action
def loadFileHelper(type):
    global file_type
    file_type = type
    load_file_flag.set()

#MARK: Load Files
def loadFiles(type):
    # global file_path
    global paths
    match type:
        case "audio":
            #Which file types should be selectable?
            filetypes = (
                ('Audio files', '*.wav *.mp3 *.flac *.ogg *.mat'),
                ('All files', '*.*')
            )
            #Opens an explorer window and returns the file path and names
            file_paths = fd.askopenfilenames(
                title = 'Select the audio file you wish to classify',
                filetypes = filetypes
            )
            #If multiple files are added, show them separated by newlines
            #Only update file path if selection is made
            if file_paths != "":
                file_path_str = ""
                for i in file_paths:
                    file_path_str += i + "\n"
                #Remove the last newline character
                if len(file_path_str) > 0:
                    file_path_str = file_path_str[:-1]
                file_path.set(file_path_str)
                #Save the tuple globally to make iterating easier
                paths = file_paths
        case "csv":
            #TODO: Make this work
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
                #Reset selected audio file paths if csv is selected
                paths = []
            #TODO: Read contents into memory, so they have the same shape as extracted features
            readCSV(file_path_str)
        case "folder":
            #Unfortunately tkinter can't open a window to select either a file or a folder
            file_path_str = fd.askdirectory(
                title='Select the directory containing the audio files you wish to classify'
            )
            #Check if the file path isn't empty
            if file_path_str != "":
                file_path_str += '/'
                file_path.set(file_path_str)
                file_paths = []
                #Look for all allowed filetypes in the folder and add them to path list
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
                    file_path_str = ""
                    for i in file_paths:
                        file_path_str += i + "\n"
                    #Remove the last newline character
                    if len(file_path_str) > 0:
                        file_path_str = file_path_str[:-1]
                #If file path is empty, look for csv file instead
                else:
                    #Look for the csv file in the directory
                    file_paths += glob(file_path_str + '*.csv')
                    #Only take the first csv, since all tracks should be in one
                    file_path_str = file_paths[0].replace('\\', '/')
                    file_paths = list(file_path_str)
                    #TODO: Read csv
                    readCSV(file_path_str)
                #Set the file path string in the gui
                file_path.set(file_path_str)
                #Save the tuple globally to make iterating easier
                paths = file_paths

#TODO: make this a thing
#MARK: Load CSV
def readCSV(csv_path):
    global result_list
    #Reset the result list
    result_list = []
    #Catch possible error
    try:
        #Open the csv file
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                #Ignore header row
                if line_count != 0:
                    #Each row is a list of all features of that track, so just add it to list
                    result_list.append(row)
                line_count += 1
    except:
        #Error when opening the csv file
        print(READ_CSV_FAILED_MSG)
        mb.showerror(title="Failed to read file", message=READ_CSV_FAILED_MSG)
    #TODO: Remove this
    print(result_list)


#MARK: Save csv
def saveCSV():
    global result_list
    #Open file dialog to choose the directory
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
            except:
                print(OVERWRITE_FAILED_MSG)
                mb.showerror(title="Failed to save file", message=OVERWRITE_FAILED_MSG)
    else:   #If no data is available to save
        print(NO_DATA_MSG)
        mb.showerror(title="No data available", message=NO_DATA_MSG)
        
#MARK: Extract
def startExtraction():
    global paths
    global result_list
    FEATURE_FUNCTION_LIST = [
        feature.chroma_stft,
        feature.spectral_centroid,
        feature.spectral_bandwidth,
        feature.spectral_rolloff,
        #harmony is not a thing, librosa.effects.harmonic?
        #What is perceptr?
    ]

    #Check if files are selected
    if paths != [] and paths != ():
        #Replace the hint label with the progress label
        hint_label.pack_forget()
        progress_frame.pack(side=BOTTOM, padx=2, pady=(0,2))
        progress_text.set("Running Extraction: ")
        progress_number.set("0%")
        #Every path is divided into 7 steps
        total_progress = len(paths) * 7
        progress = 0
        result_list = []
        #Iterate over every selected path and extract the audio features
        for i in paths:
            #load the audio file using librosa
            #y is a time-series-array, sr is the sample rate
            y, sr = librosa.load(i, sr=None)
            progress += 1
            updateProgress(progress, total_progress)
            #Loop through list of functions and add result to list
            feature_list_mean = [np.mean(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
            feature_list_var = [np.var(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
            progress += 1
            updateProgress(progress, total_progress)
            #Calculate the mfccs separately, as it returns a list of 20 results, 
            #for which we need to calculate mean and var separately
            feature_list_mean += [np.mean(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
            feature_list_var += [np.var(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
            progress += 1
            updateProgress(progress, total_progress)
            #Make a feature list, extract the filename from the path, and get length
            feature_list = [i[i.rindex('/')+1:], librosa.get_duration(y=y, sr=sr)]
            #Combine the lists so that mean and var alternate
            feature_list += [feat for pair in zip(feature_list_mean, feature_list_var) for feat in pair]
            progress += 1
            updateProgress(progress, total_progress)
            #Insert the rms, tempo and crossing rate values here, since they don't take the sr as a parameter
            feature_list.insert(4, np.mean(feature.rms(y=y)))
            feature_list.insert(5, np.var(feature.rms(y=y)))
            feature_list.insert(12, np.mean(feature.zero_crossing_rate(y=y)))
            feature_list.insert(13, np.var(feature.zero_crossing_rate(y=y)))
            progress += 1
            updateProgress(progress, total_progress)
            #Use index 0 since function gives back a list with one element
            feature_list.insert(14, feature.tempo(y=y, sr=sr)[0])
            #Since harmony and perceptr can't currently be extracted, set them to 0 to match dataset length
            for i in range(4): feature_list.insert(14, np.float32(0.0))
            progress += 1
            updateProgress(progress, total_progress)
            #Get the label from the filename
            feature_list.append(feature_list[0][:feature_list[0].index('.')])
            #Append feature list of current track to complete result list, incase multiple tracks are selected
            result_list.append(feature_list)
            progress += 1
            updateProgress(progress, total_progress)

        #Hide progress label and show hints again
        time.sleep(0.5)
        progress_frame.pack_forget()
        hint_label.pack(side=BOTTOM, padx=2, pady=(0,2))
        #TODO: Remove this
        # print(result_list)
    #No files are selected
    else:
        mb.showerror(title="No files selected", message=NO_FILES_MSG)
        print(NO_FILES_MSG)

#MARK: Threading
#Put long functions on a different thread so the GUI can update still
thread1 = threading.Thread(target=thread1_handler, daemon=True)
thread1.start()

#MARK: Tkinter
root = Tk()
root.title('Music Genre Classifier')
root.minsize(width=428, height=140)
root.geometry("428x400")
# root.maxsize(width=800, height=500)
# root.resizable(False, False)

file_path = StringVar()
paths = ()
result_list = []
file_type = ""
hint_text = StringVar(value=HINT_TEXT["program_start"])
progress_text = StringVar()
progress_number = StringVar()

#MARK: Buttons
#Make a new frame to group the related buttons together
button_frame = ttk.Frame(root, padding="2 0 2 0")
button_frame.pack(anchor=W)
#Create a button widget
load_file_button = ttk.Button(button_frame, text="Load Audio", command=lambda m="audio": loadFileHelper(m), padding="2 2 2 2")
#Set an event to happen if the mouse cursor hovers over the load file button
load_file_button.bind('<Enter>', lambda a, m="load_file_button": hint_text.set(HINT_TEXT[m]))
#Pack the button into the button frame with some padding
load_file_button.pack(side=LEFT, padx=1, pady=2)
#Same procedure for the other buttons
load_folder_button = ttk.Button(button_frame, text="Load Folder", command=lambda m="folder": loadFileHelper(m), padding="2 2 2 2")
load_folder_button.bind('<Enter>', lambda a, m="load_folder_button": hint_text.set(HINT_TEXT[m]))
load_folder_button.pack(side=LEFT, padx=1, pady=2)
load_csv_file_button = ttk.Button(button_frame, text="Load Features", command=lambda m="csv": loadFileHelper(m), padding="2 2 2 2")
load_csv_file_button.bind('<Enter>', lambda a, m="load_csv_file_button": hint_text.set(HINT_TEXT[m]))
load_csv_file_button.pack(side=LEFT, padx=1, pady=2)
extract_button = ttk.Button(button_frame, text="Extract features", command=extract_flag.set, padding="2 2 2 2")
extract_button.bind('<Enter>', lambda a, m="extract_button": hint_text.set(HINT_TEXT[m]))
extract_button.pack(side=LEFT, padx=1, pady=2)
save_button = ttk.Button(button_frame, text="Save CSV", command=save_file_flag.set, padding="2 2 2 2")
save_button.bind('<Enter>', lambda a, m="save_button": hint_text.set(HINT_TEXT[m]))
save_button.pack(side=LEFT, padx=1, pady=2)

#MARK: Paths
#Scrolling path frame
file_frame = ttk.Frame(root, padding="2 0 2 0")
#Set the hint text on a mouse hover event
file_frame.bind('<Enter>', lambda a, m="file_path_label": hint_text.set(HINT_TEXT[m]))
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
hint_label = ttk.Label(root, textvariable=hint_text)
hint_label.bind('<Enter>', lambda a, m="hint_label": hint_text.set(HINT_TEXT[m]))
hint_label.pack(padx=2, pady=(0,2))

#MARK: Progress
#Extra frame for the progress text, hidden as long as no process is running
progress_frame = ttk.Frame(root, padding="0 2 0 2")
# progress_frame.bind('<Enter>', lambda a, m="progress_label": hint_text.set(HINT_TEXT[m]))
progress_text_label = ttk.Label(progress_frame, textvariable=progress_text)
progress_text_label.pack(side=LEFT)
progress_number_label = ttk.Label(progress_frame, textvariable=progress_number)
progress_number_label.pack(side=LEFT)


root.mainloop()

# rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# cbc = cb.CatBoostClassifier(verbose=0, eval_metric='Accuracy', loss_function='MultiClass')
# xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# for clf in (rf, cbc, xgb):
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)