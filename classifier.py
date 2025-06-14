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
# Add recursive path finder mode (to train the model from gui)

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

#MARK: Progress func
#Handles showing and hiding the progress bar and updating it
def updateProgress(process):
    global progress
    global total_progress
    global paths
    #Process name and 0% is set in function directly
    progress += 1
    #Check if this is the first call for this progress
    if progress == 1:
        match process:
            case "extraction":
                #Set total progress to the amount of steps per path * amount of paths
                total_progress = EXTRACTION_STEPS * len(paths)
                progress_text.set("Running extraction: ")
                #Lock the extraction button for the duration of the process
                extract_button.config(state=DISABLED)
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
        hint_label.pack(side=BOTTOM, padx=2, pady=(0,2))
        #Reset the progress
        progress = 0
        #Do something when process is finished
        match process:
            case "extraction":
                #Unlock save button only after extraction has finished
                save_button.config(state=NORMAL)
                #Unlock extract button again
                extract_button.config(state=NORMAL)

#MARK: Thread 1 handler
#Handles several functions running on one thread
#Since all these functions can't run concurrently, they can share a thread
def thread1_handler():
    global file_type
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
        title = 'Select the audio file you wish to classify',
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
            file_path_str = ""
            #Add newlines add the end of each path for the gui display
            for i in file_paths:
                file_path_str += i + "\n"
            #Remove the last newline character
            if len(file_path_str) > 0:
                file_path_str = file_path_str[:-1]
        #If no audio files are detected but subfolders are available,
        #ask if they should be scanned recursively to add all audio to path
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
                readCSV(file_path_str)
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
                            file_path_str = ""
                            #Add newlines add the end of each path for the gui display
                            for i in file_paths:
                                file_path_str += i + "\n"
                            #Remove the last newline character
                            if len(file_path_str) > 0:
                                file_path_str = file_path_str[:-1]
                        #No files found in subfolders
                        else:
                            mb.showwarning(title="No files found", message="No files have been found in all subfolders.")

                hint_text.set(HINT_TEXT["no_files_found"])
        #Set the file path string in the gui
        file_path.set(file_path_str)

#MARK: Read CSV
#Reads the features in a csv into memory
def readCSV(csv_path):
    global result_list
    global already_saved
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
            #Since new features have been loaded, reset the saved flag
            already_saved = False
            #Enable the save button, since features are now available to save
            save_button.config(state=NORMAL)
            #Disable the extract button, since old audio files have now been unloaded
            extract_button.config(state=DISABLED)
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
        
#MARK: Extract
#Extracts the features from the loaded audio files
def startExtraction():
    global paths
    global result_list
    global already_extracted
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
            y, sr = librosa.load(i, sr=None)
            #Update the progress text and bar (needs to be called each progress step)
            updateProgress("extraction")
            #Loop through list of functions and add result to list
            feature_list_mean = [np.mean(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
            feature_list_var = [np.var(func(y=y, sr=sr)) for func in FEATURE_FUNCTION_LIST]
            updateProgress("extraction")
            #Calculate the mfccs separately, as it returns a list of 20 results, 
            #for which we need to calculate mean and var separately
            feature_list_mean += [np.mean(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
            feature_list_var += [np.var(mfcc) for mfcc in feature.mfcc(y=y, sr=sr)]
            updateProgress("extraction")
            #Make a feature list, extract the filename from the path, and get length
            feature_list = [i[i.rindex('/')+1:], librosa.get_duration(y=y, sr=sr)]
            #Combine the lists so that mean and var alternate
            feature_list += [feat for pair in zip(feature_list_mean, feature_list_var) for feat in pair]
            updateProgress("extraction")
            #Insert the rms, tempo and crossing rate values here, since they don't take the sr as a parameter
            feature_list.insert(4, np.mean(feature.rms(y=y)))
            feature_list.insert(5, np.var(feature.rms(y=y)))
            feature_list.insert(12, np.mean(feature.zero_crossing_rate(y=y)))
            feature_list.insert(13, np.var(feature.zero_crossing_rate(y=y)))
            updateProgress("extraction")
            #Use index 0 since function gives back a list with one element
            feature_list.insert(14, feature.tempo(y=y, sr=sr)[0])
            #Since harmony and perceptr can't currently be extracted, set them to 0 to match dataset length
            for i in range(4): feature_list.insert(14, np.float32(0.0))
            updateProgress("extraction")
            #Get the label from the filename
            feature_list.append(feature_list[0][:feature_list[0].index('.')])
            #Append feature list of current track to complete result list, incase multiple tracks are selected
            result_list.append(feature_list)
            updateProgress("extraction")
            #Set the extracted flag
            already_extracted = True
    #No files are selected
    else:
        mb.showerror(title="No files selected", message=NO_FILES_MSG)
        print(NO_FILES_MSG)

#MARK: Threading
#Put long functions on a different thread so the GUI can update still
thread1 = threading.Thread(target=thread1_handler, daemon=True)
thread1.start()

#MARK: Tkinter config
#The base state of every gui element is configured here
root = Tk()
root.title('Music Genre Classifier')
root.minsize(width=428, height=145)
root.geometry("428x300")

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
hint_label = ttk.Label(root, textvariable=hint_text)
hint_label.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["hint_label"]))
hint_label.pack(padx=2, pady=(0,2))

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