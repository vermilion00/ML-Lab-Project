# TODO:
# How is the duration calculated in the dataset?
# Is it important that the calculated values differ slightly from the dataset values?
# Writing 100 rows of csv is instant, but maybe add progress bar to saveCSV function
# Rewrite functions to not need global variables
# Play around with ttk styling
# Glob files in single lookup instead of looping over each ending
# Fix being able to scroll up past first path for some reason
# Add option to show waveplots/spectrograms
# Add button to save new default parameters
#
# Fixes to bug:
# - Calls to their functions take too long
# Program crashes if the first three progress += 1 calls happen too fast
# Speed up startup somehow
# Separate first x songs from each class out for testing (first 2 of each?)
# Add message showing which ones belong there
# Switch to requests lib for download progress callback?
# Split models in classifier class, check it respective model is loaded before prediction
# When changing the model, pack/unpack respective buttons with their own defaults
# When model settings are shown, increase window height accordingly

from constants import *
import classifier
import numpy as np
import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkscrolledframe import ScrolledFrame
import threading
import csv
import librosa
from librosa import feature, effects, beat
from glob import glob
from os import path, remove
from sys import argv
from urllib.request import urlopen
from io import BytesIO
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
from math import floor

#MARK: Threading events
extract_flag = threading.Event()
load_file_flag = threading.Event()
save_file_flag = threading.Event()
train_model_flag = threading.Event()
predict_genre_flag = threading.Event()
save_model_flag = threading.Event()
load_model_flag = threading.Event()
load_url_flag = threading.Event()

#MARK: Progress func
#Handles showing and hiding the progress bar as well as updating it
def updateProgress(process, progress):
    global total_progress
    global new_paths
    global model_available
    global extraction_progress
    global download_progress
    #Check if this is the first call for this progress
    #Since thread 2 crashes if the first call to this function happens with progress > 1, also check for 2
    if progress == 1 or progress == 2:
        match process:
            case "extraction":
                #Set total progress to the amount of steps per path * amount of paths
                total_progress = EXTRACTION_STEPS * len(new_paths)
                progress_text.set("Running extraction: ")
                #Update button states
                updateUI("extraction_started")
            case "training":
                #Set total progress to the amount of steps per path * amount of paths
                total_progress = epochs.get()
                progress_text.set("Training model: ")
                updateUI("training_started")
            case "download":
                total_progress = 100
                progress_text.set("Downloading File: ")
                updateUI("download_started")
        #Replace the hint label with the progress label and bar
        hint_label.pack_forget()
        progress_frame.pack(side=BOTTOM, padx=2, pady=(0,2), fill=X)
        progress_number.set("0%")

    num = int(progress/total_progress*100)
    #Process is still ongoing
    if num < 100:
        progress_number.set(f'{num}%')
        progress_bar['value'] = num
    #Process is done
    else:
        #Limit the max progress value to 100
        progress_bar["value"] = 100
        #Hide the progress number
        progress_number.set("")
        #Do something when process is finished
        match process:
            case "extraction":
                #Show a message
                progress_text.set("Extraction Complete")
                updateUI("extraction_finished")
                #Reset the extraction progress counter
                extraction_progress = 0
                #Sleep for a short time keep the progress completed messages up
                time.sleep(1)
            case "training":
                progress_text.set("Training Complete")
                updateUI("training_finished")
                #Reset the classifier progress counter
                classifier.progress = 0
                #Sleep for a short time keep the progress completed messages up
                time.sleep(1)
            case "download":
                progress_text.set("Download Complete")
                updateUI("download_finished")
                download_progress = 0
        #Hide progress label and bar
        progress_frame.pack_forget()
        #Set the hint label to say that the process is done
        hint_text.set(HINT_TEXT[process])
        #Show the hint label again
        hint_label.pack(padx=2, pady=(0,2), fill=X)

#MARK: Thread 1
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
        elif load_url_flag.is_set():
            load_url_flag.clear()
            loadURL()
        elif save_file_flag.is_set():
            save_file_flag.clear()
            saveCSV()
        elif train_model_flag.is_set():
            train_model_flag.clear()
            trainModelHelper(result_list)
        elif predict_genre_flag.is_set():
            predict_genre_flag.clear()
            #Show the predicted genres in their label
            updateUI("prediction_started")
            match model_type.get():
                case "Neural Model":
                    predicted_genres.set(c.predictGenre(result_list, "neural", used_slow_features))
                case "Cat Boost":
                    predicted_genres.set(c.predictGenre(result_list, "cat", used_slow_features))
            updateUI("prediction_finished")
        elif save_model_flag.is_set():
            save_model_flag.clear()
            saveModel()
        elif load_model_flag.is_set():
            load_model_flag.clear()
            loadModelHelper()
        #Add other functions that run on the same thread as elif
        else:
            #If thread is not needed, sleep for 100ms before polling again
            time.sleep(0.1)

#MARK: Thread 2
#Not my favourite way of updating progress but I don't know a better way to show
#training progress when that code is in a separate file
def thread2_handler():
    global extraction_progress
    global download_progress
    local_extraction_progress = 0
    local_training_progress = 0
    local_download_progress = 0
    extraction_progress = 0
    download_progress = 0
    while True:
        if local_extraction_progress != extraction_progress:
            local_extraction_progress = extraction_progress
            updateProgress("extraction", extraction_progress)
        elif local_download_progress != download_progress:
            local_download_progress = download_progress
            updateProgress("download", download_progress)
        elif local_training_progress != classifier.progress:
            local_training_progress = classifier.progress
            updateProgress("training", classifier.progress)
        else:
            #Since the thread crashes if the first progress incrementation isn't caught, use a shorter time here
            time.sleep(0.05)

#MARK: Load helper
#A helper function, since I can't call a function and set the flag in a button action
def loadFileHelper(type):
    global file_type
    file_type = type
    load_file_flag.set()

#MARK: Load audio
def loadAudio():
    global paths
    global new_paths
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
        #In case the extraction is cancelled, save the old path to restore it
        old_path = file_path.get()
        file_path_str = ""
        #Add newlines at the end of each path for the gui display
        for i in file_paths:
            file_path_str += i + "\n"
        #Remove the last newline character
        if len(file_path_str) > 0:
            file_path_str = file_path_str[:-1]
        
        #If the checkbox isn't marked or the filepath is empty, replace the text
        if append_path.get() == False or file_path.get() == "No files selected":
            new_paths = list(file_paths)
            paths = file_paths
            file_path.set(file_path_str)
        #If the checkbox is marked, append to the path
        else:
            new_paths = list(file_paths)
            #Save the tuple globally to make iterating easier
            paths += file_paths
            file_path.set(file_path.get() + '\n' + file_path_str)
        #Since new data is available, reset extracted flag
        already_extracted = False
        #Extract files after loading them
        if startExtraction() == "cancelled":
            #If the extraction is cancelled, restore the old path
            file_path.set(old_path)
        else:
            updateUI("loaded_audio")

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
         #If the checkbox isn't marked or the filepath is empty, replace the text
        if append_path.get() == False or file_path.get() == "No files selected":
            #Reset paths and file text
            paths = []
            file_path.set(file_path_str)
        else:
            #If the checkbox is marked, append to the path
            file_path.set(file_path.get() + '\n' + file_path_str)
        readCSV(file_path_str)

#MARK: Load Folder
def loadFolder():
    global paths
    global new_paths
    global already_extracted
    #Unfortunately tkinter can't open a window to select either a file or a folder
    file_path_str = fd.askdirectory(
        title='Select the directory containing the audio files you wish to classify'
    )
    #Check if the file path isn't empty
    if file_path_str != "":
        file_path_str += '/'
        file_paths = []
        #Save the old path to restore it if extraction is cancelled
        old_path = file_path.get()
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
            #Convert the file path list to a string
            file_path_str = ""
            for i in file_paths:
                #Add newlines add the end of each path for the gui display
                file_path_str += i + "\n"
            #Remove the last newline character
            if len(file_path_str) > 0:
                file_path_str = file_path_str[:-1]
            #If the checkbox isn't marked or the filepath is empty, replace the text
            if append_path.get() == False or file_path.get() == "No files selected":
                new_paths = list(file_paths)
                paths = file_paths
                file_path.set(file_path_str)
            #If the checkbox is marked, append to the path
            else:
                new_paths = list(file_paths)
                #Save the tuple globally to make iterating easier
                paths += file_paths
                file_path.set(file_path.get() + '\n' + file_path_str)
            #Since new data is available, reset extracted flag
            already_extracted = False
            #Extract files after loading them
            if startExtraction() == "cancelled":
                #If the extraction is cancelled, restore the old path
                file_path.set(old_path)
            else:
                #Update GUI accordingly
                updateUI("loaded_audio")
    
        #If file path is empty, look for csv file instead
        else:
            #Look for the csv file in the directory
            file_paths += glob(file_path_str + '*.csv')
            #Check if a csv file has been found
            if file_paths != []:
                #Only take the first csv, since all tracks should be in one
                file_path_str = file_paths[0].replace('\\', '/')
                file_paths = list(file_path_str)
                if append_path.get() == False or file_path.get() == "No files selected":
                    #Reset path
                    paths = []
                    #Set the file path string in the gui
                    file_path.set(file_path_str)
                else:
                    #Append new file to file text
                    file_path.set(file_path.get() + '\n' + file_path_str)
                #Read the contents of the csv file into memory
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
                            #Convert the file path list to a string
                            file_path_str = ""
                            for i in file_paths:
                                #Add newlines add the end of each path for the gui display
                                file_path_str += i + "\n"
                            #Remove the last newline character
                            if len(file_path_str) > 0:
                                file_path_str = file_path_str[:-1]
                            #If the checkbox isn't marked or the filepath is empty, replace the text
                            if append_path.get() == False or file_path.get() == "No files selected":
                                new_paths = list(file_paths)
                                paths = file_paths
                                file_path.set(file_path_str)
                            #If the checkbox is marked, append to the path
                            else:
                                new_paths = list(file_paths)
                                #Save the tuple globally to make iterating easier
                                paths += file_paths
                                file_path.set(file_path.get() + '\n' + file_path_str)
                            #Since new data is available, reset extracted flag
                            already_extracted = False
                            #Extract files after loading them
                            if startExtraction() == "cancelled":
                                #If the extraction is cancelled, restore the old path
                                file_path.set(old_path)
                            else:
                                #Update GUI accordingly
                                updateUI("loaded_audio")
                        #No files found in subfolders
                        else:
                            mb.showinfo(title="No files found", message="No files have been found in any subfolders.")

                hint_text.set(HINT_TEXT["no_files_found"])

#MARK: Read CSV
#Reads the features in a csv file into memory
def readCSV(csv_path):
    global result_list
    global already_saved
    global model_already_trained
    global used_slow_features
    #Reset the result list
    if append_path.get() == False or file_path.get() == "No files selected":
        result_list = []
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
            #Update the button state accordingly
            updateUI("opened_csv")
            used_slow_features = use_slow_features.get()
    except Exception as e:
        #Error when opening the csv file
        print(READ_CSV_FAILED_MSG)
        mb.showerror(title="Failed to read file", message=READ_CSV_FAILED_MSG + f'\n{e}')

#MARK: Save CSV
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
            except Exception as e:
                print(OVERWRITE_FAILED_MSG)
                mb.showerror(title="Failed to save file", message=OVERWRITE_FAILED_MSG + f'\n{e}')
    else:   #If no data is available to save
        print(NO_DATA_MSG)
        mb.showerror(title="No data available", message=NO_DATA_MSG)

#MARK: Save Model
def saveModel():
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
    filename = fd.asksaveasfilename(
        filetypes=filetype,
        defaultextension=filetype,
        title=title,
        initialfile=default_name,
        initialdir=current_dir
    )
    #If filename is empty, the save prompt has been canceled
    if filename != "":
        #Catch errors when writing to file
        try:
            #Set the saved flag
            model_already_saved = c.saveModel(filename)
            hint_text.set("Saved model")
        except Exception as e:
            print(OVERWRITE_FAILED_MSG)
            mb.showerror(title="Failed to save file", message=OVERWRITE_FAILED_MSG + f'\n{e}')

#MARK: Load Model
def loadModelHelper():
    global model_already_saved
    global model_available
    global result_list
    filetypes = (
        ('Keras model file', '*.keras'),
        ('All files', '*.*')
    )
    #Only allow selecting one file here
    file_path_str = fd.askopenfilename(
        title = 'Select the keras model file you wish to load',
        filetypes = filetypes,
        initialdir=current_dir
    )
    #Check if a file has been selected
    if file_path_str != "":
        try:
            #Load the selected model
            c.loadModel(file_path_str)
            #Reset the Model Saved flag
            model_already_saved = False
            #Set the model_available flag
            model_available = True
            updateUI("loaded_model")
            model_loaded_text.set(f"Model status:\nLoaded model \"{file_path_str}\"")
            hint_text.set(f"Loaded model \"{file_path_str}\"")
        except Exception as e:
            print("Failed to load model")
            mb.showerror(title="Failed to load model or scaler", message=MODEL_LOADING_FAILED_MSG + f'\n{e}')


#A helper function to increment progress counter during list execution
def incrementProgressHelper():
    global extraction_progress
    extraction_progress += 1

#MARK: Extract
#Extracts the features from the loaded audio files
def startExtraction(source="file", segment=None):
    global new_paths
    global result_list
    global already_extracted
    global already_saved
    global model_already_trained
    global extraction_progress
    global used_slow_features
    #Check if audio files are selected
    if new_paths != []:
        #If the slow feature option is mismatched between the model and the current setting,
        #ask if it should be changed before extracting
        if use_slow_features.get() != c.used_slow_features:
            if c.used_slow_features == True:
                answer = mb.askyesnocancel(title="Options Mismatch", message=OPTION_MISMATCH_1_MSG)
            else:
                answer = mb.askyesnocancel(title="Options Mismatch", message=OPTION_MISMATCH_2_MSG)
            #If the answer is yes, change the setting to match the model
            if answer == True:
                use_slow_features.set(c.used_slow_features)
            #If the prompt was cancelled, abort the function call
            elif answer == None:
                return "cancelled"
            #If the answer is no, just continue
        #Only clear already extracted data if append path isn't checkeds
        if append_path.get() == False:
            result_list = []
        #Iterate over every selected path and extract the audio features
        for i in new_paths:
            try:
                #Download the url target first if the source isn't a file
                if source != "file":
                    y, sr = loadFileFromURL(i, segment)
                else:
                    #Try the easy implementation
                    try:
                        #Load audio from a file
                        y, sr = librosa.load(i)
                    except:
                        #If librosa can't open the file, try resampling it to wav first (needs ffmpeg)
                        try:
                            #Resample to wav
                            wav = BytesIO
                            AudioSegment.from_file(i).export(wav, 'wav')
                            y, sr = librosa.load(wav, sr=None)
                        except Exception as e:
                            print(f"Failed to load file.\n{e}")
                            mb.showerror(title="Failed to load file.", message=f"The file {i} failed to load. Make sure the format is one of [.wav, .mp3, .flac, .ogg]. For extended file support, e.g. YouTube, make sure that ffmpeg is installed correctly.\n{e}")

                extraction_progress += 1
                #Put (almost) all functions in a list for easy processing
                if use_slow_features.get() == True:
                    feature_function_list = [
                        feature.chroma_stft(y=y, sr=sr),
                        feature.rms(y=y),
                        feature.spectral_centroid(y=y, sr=sr),
                        feature.spectral_bandwidth(y=y, sr=sr),
                        feature.spectral_rolloff(y=y, sr=sr),
                        feature.zero_crossing_rate(y=y),
                        #Add helper function to increment progress counter during list execution
                        incrementProgressHelper(),
                        #Use an unpacking operator since this returns two values we need
                        *effects.hpss(y=y),
                        incrementProgressHelper(),
                        #Use an unpacking operator since this returns a list of lists
                        *feature.mfcc(y=y, sr=sr)
                    ]
                #Don't extract hpss
                else:
                    feature_function_list = [
                        feature.chroma_stft(y=y, sr=sr),
                        feature.rms(y=y),
                        feature.spectral_centroid(y=y, sr=sr),
                        feature.spectral_bandwidth(y=y, sr=sr),
                        feature.spectral_rolloff(y=y, sr=sr),
                        feature.zero_crossing_rate(y=y),
                        #Add helper function to increment progress counter during list execution
                        incrementProgressHelper(),
                        #Use an unpacking operator since this returns two values we need
                        0,0,
                        incrementProgressHelper(),
                        #Use an unpacking operator since this returns a list of lists
                        *feature.mfcc(y=y, sr=sr)
                    ]
                extraction_progress += 1
                #Make a feature list, extract the filename from the path, and get length
                #If the name doesn't follow naming procedure, use default name
                try:
                    name = i[i.rindex('/')+1:] 
                except:
                    name = "missing_name"
                feature_list = [name, librosa.get_duration(y=y, sr=sr)]
                #Delete the progress helper function indexes
                del feature_function_list[6]
                #This one is at index 8 since hpss unpacks into two indexes
                del feature_function_list[8]
                #Iterate through all functions in the array and get the mean and var
                for feat in feature_function_list:
                    feature_list.append(np.mean(feat))
                    feature_list.append(np.var(feat))
                extraction_progress += 1
                #Insert the tempo, since the return format is different and no mean or var is needed
                #Unpack and save the first value, since the rest is not needed
                if use_slow_features.get() == True:
                    feature_list.insert(18, *beat.beat_track(y=y, sr=sr)[0])
                else:
                    feature_list.insert(18, 0)
                extraction_progress += 1
                #Get the label from the filename
                #If the label doesn't follow naming procedure, use default label
                try:
                    label = feature_list[0][:feature_list[0].index('.')]
                except:
                    label = "missing_label"
                feature_list.append(label)
                #Append feature list of current track to complete result list, incase multiple tracks are selected
                result_list.append(feature_list)
                extraction_progress += 1
                #Set the extracted flag
                already_extracted = True
                #Reset the already saved flag
                already_saved = False
                #Since new data is available that the model can be trained on, clear the flag
                model_already_trained = False
                #Update the slow features state
                used_slow_features = use_slow_features.get()

            #A file in the list could not be opened
            except Exception as e:
                #If the first file can't be opened, add progress slowly so progress bar can load
                if extraction_progress == 0:
                    extraction_progress = 1
                    time.sleep(0.1)
                    extraction_progress = EXTRACTION_STEPS
                else:
                    #Add the progress, since the updateProgress function expects 7 more calls to happen
                    extraction_progress += EXTRACTION_STEPS
                print(f"A file at {i} could not be opened.")
                mb.showerror(title="Invalid file", message=f"The file \"{i[i.rindex('/')+1:]}\" at path \"{i[:i.rindex('/')+1]}\" could not be opened. The extraction will continue without it.\n{e}")
    #No files are selected
    else:
        mb.showerror(title="No files selected", message=NO_FILES_MSG)
        print(NO_FILES_MSG)

#MARK: Update Hook
def ytUpdateHook(download):
    global download_progress
    percent = download['_percent_str']
    for i, char in enumerate(percent):
        if char == '%':
            value = percent[i-5:i+1].replace('%', '')
            break
    #Save the progress as an int
    download_progress = floor(float(value.replace(' ', '')))

#MARK: Load File URL
def loadFileFromURL(url, segment):
    global total_progress
    #Set the total progress here to avoid crashing the program when loading small files
    total_progress = 100
    #Load audio from url
    #If url is a youtube link, use yt-dlp
    # if ".youtu.be" in url or ".youtube." in url:
    try:
        if segment == None:
            yt_options = {
                "format": "bestaudio",
                "outtmpl": "temp.wav",
                "progress_hooks": [ytUpdateHook]
            }
        #If a segment is specified, download only that portion
        else:
            yt_options = {
                "format": "bestaudio",
                "outtmpl": "temp.wav",
                "download_ranges": download_range_func(None, [(segment[0], segment[1])]),
                "force_keyframes_at_cuts": True,
                "progress_hooks": [ytUpdateHook]
            }
        try:
            #TODO: Try saving to BytesIO directly - Need to find out how to reencode the file from bytesIO
            with YoutubeDL(yt_options) as audio:
                audio.download([url])
                hint_text.set("Download complete, loading File")
            y, sr = librosa.load('temp.wav', sr=None)
            #Remove the temp file after loading it
            remove('temp.wav')
        except Exception as e:
            print(f"Failed to load YouTube URL.\n{e}")
            mb.showerror("Failed to load YouTube URL", message=f"Failed to load YouTube URL. Either the URL is invalid, or ffmpeg is not installed correctly and added to PATH.\n{e}")
    #Try using this easy implementation
    except:
    # else:
        try:
            file = BytesIO(urlopen(url).read())
            hint_text.set("Download complete, loading File")
        except Exception as e:
            print(f"URL failed to load.\n{e}")
            mb.showerror(title="Failed to load URL", message=f"The URL failed to load.\n{e}")
        #Load the features from a bytes object
        try:
            y, sr = librosa.load(file)
        except:
            #If librosa can't open the file, try resampling it to wav first (needs ffmpeg)
            try:
                #Resample to wav
                wav = BytesIO
                with urlopen(url) as request:
                    request.seek = lambda *args: None
                    AudioSegment.from_file(request).export(wav, "wav")
                wav.seek(0)
                hint_text.set("Download complete, loading File")
                y, sr = librosa.load(wav, sr=None)
            except Exception as e:
                print(f"Failed to load file.\n{e}")
                mb.showerror(title="Failed to load file.", message=f"The file {url} failed to load. Make sure the format is one of [.wav, .mp3, .flac, .ogg]. For extended file support, e.g. YouTube, make sure that ffmpeg is installed correctly.\n{e}")
    return y, sr

#MARK: Train Model
def trainModelHelper(result_list):
    global model_already_saved
    global model_already_trained
    global model_available
    global used_slow_features
    params_changed = False
    try:
        #Check if the parameters have changed
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
    #Invalid input
    except Exception as e:
        print(INVALID_INPUT_MSG)
        #Show a warning box if an input field contains invalid input
        mb.showerror(title="Invalid input", message=INVALID_INPUT_MSG + f'\n{e}')
        #Abort the function call
        return
    #Prep the data
    c.prepareData(result_list, use_slow_features=used_slow_features)
    match model_type.get():
        case "Neural Model":
            #Build the model
            c.buildModel()
            #Train the model
            test_accuracy, test_loss = c.trainModel()
        case "Cat Boost":
            #Cat boost doesn't build the model
            test_accuracy, test_loss = c.trainModelCat(depth=6, task_type="CPU")
    #In case the training is stopped early, add progress to avoid issues
    classifier.progress = epochs.get()
    #Set the new button state
    updateUI("training_finished")
    #Update labels
    model_loaded_text.set(f"Model status:\nTraining complete - Model loaded\nTest Accuracy: {test_accuracy*100:.2f}%, Test Loss: {test_loss:.4f}")
    hint_text.set("Model training complete")
    #Reset the saved model flag, since it's been trained again
    model_already_saved = False
    #Set the model available flag
    model_available = True
    #Reset the model already trained flag
    model_already_trained = False

#MARK: Update UI
def updateUI(key:str):
    global model_available
    global already_extracted
    match key:
        case "extraction_started":
            #Lock the extraction button for the duration of the process
            # extract_button.config(state=DISABLED)
            #Lock the load buttons
            load_file_button.config(state=DISABLED)
            load_csv_file_button.config(state=DISABLED)
            load_folder_button.config(state=DISABLED)
            #Lock the save button
            save_button.config(state=DISABLED)
            #Lock the model buttons for the duration of the process
            train_model_button.config(state=DISABLED)
            predict_genre_button.config(state=DISABLED)
        case "extraction_finished":
            #Ulock the load buttons
            load_file_button.config(state=NORMAL)
            load_csv_file_button.config(state=NORMAL)
            load_folder_button.config(state=NORMAL)
            #Unlock save button only after extraction has finished
            save_button.config(state=NORMAL)
            #Unlock extract button again
            # extract_button.config(state=NORMAL)
            #Unlock the model buttons
            train_model_button.config(state=NORMAL)
            #Only unlock the prediction button if a model is loaded
            if model_available:
                predict_genre_button.config(state=NORMAL)
            #Hide the predicted genre frame, since new paths have been loaded
            genre_frame.pack_forget()
        case "training_started":
            #Lock the model buttons
            load_model_button.config(state=DISABLED)
            train_model_button.config(state=DISABLED)
            predict_genre_button.config(state=DISABLED)
            save_model_button.config(state=DISABLED)
        case "training_finished":
            #Unlock the model buttons after the process
            load_model_button.config(state=NORMAL)
            train_model_button.config(state=NORMAL)
            predict_genre_button.config(state=NORMAL)
            save_model_button.config(state=NORMAL)
            #Unlock the show matrices button
            show_matrices_button.config(state=NORMAL)
            #Lock the predict button until new data is loaded
            predict_genre_button.config(state=DISABLED)
        case "prediction_started":
            #Lock the model buttons
            load_model_button.config(state=DISABLED)
            train_model_button.config(state=DISABLED)
            predict_genre_button.config(state=DISABLED)
            save_model_button.config(state=DISABLED)
        case "prediction_finished":
            #Unlock the model buttons after the process
            load_model_button.config(state=NORMAL)
            train_model_button.config(state=NORMAL)
            predict_genre_button.config(state=NORMAL)
            save_model_button.config(state=NORMAL)
            #Show the genre frame
            # genre_frame.pack(after=file_path_label, anchor=W)
            genre_frame.pack(anchor=W)
            #Update the hint text
            hint_text.set(HINT_TEXT["prediction"])
        case "download_started":
            progress_text.set("Downloading file")
            #Replace the hint label with the progress label
            hint_label.pack_forget()
            # progress_bar.config()
            progress_frame.pack(side=BOTTOM, padx=2, pady=(0,2), fill=X)
        case "download_finished":
            hint_text.set("Download finished")
            #Replace the progress label with the hint label
            progress_frame.pack_forget()
            hint_label.pack(padx=2, pady=(0,2), fill=X)
        #TODO: Check if this case exists
        case "loaded_audio":
            #Enable the extract button, since audio paths are now available
            # extract_button.config(state=NORMAL)
            #Disable the save button, to avoid confusion, since the new audio files haven't been extracted
            save_button.config(state=DISABLED)
            #Lock the model buttons, to avoid confusion, since the new audio files haven't been extracted
            train_model_button.config(state=DISABLED)
            predict_genre_button.config(state=DISABLED)
            #Hide the predicted genre frame, since new paths have been loaded
            genre_frame.pack_forget()
        case "opened_csv":
            #Enable the save button, since features are now available to save
            save_button.config(state=NORMAL)
            #Disable the extract button, since old audio files have now been unloaded
            # extract_button.config(state=DISABLED)
            #Unlock the model buttons, since data is now available
            train_model_button.config(state=NORMAL)
            #Only unlock the predict genre button if a model is available
            if model_available:
                predict_genre_button.config(state=NORMAL)
            #Hide the predicted genre frame, since new paths have been loaded
            genre_frame.pack_forget()
        case "loaded_model":
            save_model_button.config(state=NORMAL)
            #Unlock the show matrices button
            show_matrices_button.config(state=NORMAL)
            #If the selected files have been extracted, unlock predict genre button
            if already_extracted:
                predict_genre_button.config(state=NORMAL)
        case "show_options":
            #Hide the show advanced button
            show_advanced_button.pack_forget()
            #Show the hide advanced button
            hide_advanced_button.pack(after=predict_genre_button, side=LEFT, padx=1, pady=2)
            #Show the advanced options
            model_frame.pack(anchor=N)
            #Increase the window height by the options height, which is 107
            height = root.winfo_height()
            root.geometry(f"{root.winfo_width()}x{height+107}")
        case "hide_options":
            #Hide the advanced options    
            model_frame.pack_forget()
            #Hide the hide advanced button
            hide_advanced_button.pack_forget()
            #Show the show advanced button
            show_advanced_button.pack(after=predict_genre_button, side=LEFT, padx=1, pady=2)
            #Decrease the window height by the options height
            height = root.winfo_height()
            root.geometry(f"{root.winfo_width()}x{height-107}")

#MARK: Wraplength
#Set the wraplength of the label based on the window size
def setWraplength(event, label=None):
    match label:
        #Special case for this label since the contents of the file frame don't change in size
        case "model_loaded_label":
            #Subtract 30 from the width for due to the scrollbar
            model_loaded_label.configure(wraplength=event.width-30)
        #All other elements
        case _:
            #Subtract 6 from the width for padding
            event.widget.configure(wraplength=event.width-6)

#MARK: Load URL
def loadURL():
    #TODO: Bring this in line with other functions
    global new_paths
    global paths
    #Check if the entry field contains text
    if url.get() != "":
        #Check if a length is denoted by e.g. [2,10]
        split_url = url.get().split(' ', maxsplit=1)
        #If the checkbox isn't marked or the filepath is empty, replace the text
        if append_path.get() == False or file_path.get() == "No files selected":
            # paths = file_paths
            file_path.set(url.get())
        #If the checkbox is marked, append to the path
        else:
            #Save the tuple globally to make iterating easier
            # paths += file_paths
            file_path.set(file_path.get() + '\n' + url.get())
        #Add url to List so that the extraction function will work properly
        new_paths = []
        new_paths.append(split_url[0])
        #If the entry only contains the url, just call the function
        if len(split_url) == 1:
            startExtraction(new_paths)
        else:
            #If the entry contains a segment, parse the segment first
            try:
                #Remove spaces
                split_url[1] = split_url[1].replace(' ', '')
                #Split the string by the comma
                segment = split_url[1].split(',')
                #Remove the leading and trailing brackets
                segment[0] = float(segment[0][1:])
                segment[1] = float(segment[1][:-1])
                startExtraction(new_paths, segment)
            #Malformed segment
            except:
                print(INVALID_SEGMENT_MSG)
                mb.showerror(title="Invalid Segment", message=INVALID_SEGMENT_MSG)

def updateSelection(event):
    match model_type.get():
        case "Neural Model":
            learning_rate.set(DEFAULT_NEURAL["learning_rate"])
            epochs.set(DEFAULT_NEURAL["epochs"])
            # epochs_label.config(text="Epochs")
        case "Cat Boost":
            learning_rate.set(DEFAULT_CAT["learning_rate"])
            epochs.set(DEFAULT_CAT["iterations"])
            # epochs_label.config(text="Iterations")

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
root.minsize(width=400, height=224)
root.geometry("430x340")

c = classifier.Classifier(patience=1000)
file_path = StringVar(value="No files selected")
file_paths = []
new_paths = []
paths = []
result_list = []
file_type = ""
#Initialise the hint text with the startup message
model_loaded_text = StringVar(value="Model status: No model loaded")
hint_text = StringVar(value=HINT_TEXT["program_start"])
progress_text = StringVar()
progress_number = StringVar()
predicted_genres = StringVar()
url = StringVar()
#Variables for the dropdown menu
model_type = StringVar(value="Neural Model")
model_types = ["Neural Model", "Cat Boost"]
append_path = BooleanVar(value=False)
use_slow_features = BooleanVar(value=True)
used_slow_features = use_slow_features.get()
#Progress bar progress
extraction_progress = 0
download_progress = 0
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
current_dir = path.dirname(path.abspath(argv[0]))

#MARK: Buttons
button_helper_frame = ttk.Frame(root, padding="0 2 0 2")
button_helper_frame.pack(anchor=N)
#Make a new frame to group the related buttons together
button_row_1_frame = ttk.Frame(button_helper_frame, padding="2 0 2 0")
button_row_1_frame.pack(anchor=N, side=TOP)
#Create a checkbox
checkbox = ttk.Checkbutton(button_row_1_frame, variable=append_path, text="Append")
checkbox.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["checkbox"]))
checkbox.pack(side=LEFT)
#Create a button widget
load_file_button = ttk.Button(button_row_1_frame, text="Load Audio", command=lambda:loadFileHelper("audio"), padding="2 2 2 2")
#Set an event to happen if the mouse cursor hovers over the load file button
load_file_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_file_button"]))
#Pack the button into the button frame with some padding
load_file_button.pack(side=LEFT, padx=1, pady=2)
#Same procedure for the other buttons
load_folder_button = ttk.Button(button_row_1_frame, text="Load Folder", command=lambda:loadFileHelper("folder"), padding="2 2 2 2")
load_folder_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_folder_button"]))
load_folder_button.pack(side=LEFT, padx=1, pady=2)
predict_genre_button = ttk.Button(button_row_1_frame, text="Predict Genre", command=predict_genre_flag.set, state=DISABLED)
predict_genre_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["predict_genre_button"]))
predict_genre_button.pack(side=LEFT, padx=1, pady=2)
show_advanced_button = ttk.Button(button_row_1_frame, width=13, text="Show Options", command=lambda: updateUI("show_options"))
show_advanced_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["show_options_button"]))
show_advanced_button.pack(side=LEFT, padx=1, pady=2)
#Hide button remains unpacked for now
hide_advanced_button = ttk.Button(button_row_1_frame, width=13, text="Hide Options", command=lambda: updateUI("hide_options"))
hide_advanced_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["hide_options_button"]))

#MARK: URL Widgets
url_frame = ttk.Frame(button_helper_frame, padding="5 0 5 0")
url_frame.pack(fill=X)
url_entry = ttk.Entry(url_frame, textvariable=url, justify=CENTER)
url_entry.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["url_entry"]))
url_entry.pack(side=LEFT, fill=X, expand=True)
load_url_button = ttk.Button(url_frame, text="Load URL", command=load_url_flag.set)
load_url_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_url_button"]))
load_url_button.pack(side=LEFT)

#MARK: Model buttons
model_frame = ttk.Frame(button_helper_frame, padding="0 2 0 2")
button_row_3_frame = ttk.Frame(model_frame, padding="2 2 2 2")
button_row_3_frame.pack(anchor=N, side=TOP)
load_model_button = ttk.Button(button_row_3_frame, text="Load Model", command=load_model_flag.set)
load_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_model_button"]))
load_model_button.pack(side=LEFT)
train_model_button = ttk.Button(button_row_3_frame, text="Train Model", command=train_model_flag.set, state=DISABLED)
train_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["train_model_button"]))
train_model_button.pack(side=LEFT)
save_model_button = ttk.Button(button_row_3_frame, text="Save Model", command=save_model_flag.set, state=DISABLED)
save_model_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["save_model_button"]))
save_model_button.pack(side=LEFT)
load_csv_file_button = ttk.Button(button_row_3_frame, text="Load CSV", command=lambda:loadFileHelper("csv"), padding="2 2 2 2")
load_csv_file_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["load_csv_file_button"]))
load_csv_file_button.pack(side=LEFT)
save_button = ttk.Button(button_row_3_frame, text="Save CSV", command=save_file_flag.set, padding="2 2 2 2", state=DISABLED)
save_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["save_button"]))
save_button.pack(side=LEFT)

#MARK: Model Entries
model_entry_frame = ttk.Frame(model_frame)
model_entry_frame.pack()
#Learning rate elements
learning_rate_frame = ttk.Frame(model_entry_frame)
learning_rate_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["learning_rate"]))
learning_rate_frame.pack(side=LEFT)
learning_rate_label = ttk.Label(learning_rate_frame, text="Learning Rate").pack()
learning_rate_entry = ttk.Entry(learning_rate_frame, textvariable=learning_rate, width=12, justify=CENTER).pack()
#Epochs elements
epochs_frame = ttk.Frame(model_entry_frame)
epochs_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["epochs"]))
epochs_frame.pack(side=LEFT)
epochs_label = ttk.Label(epochs_frame, text="Epochs").pack()
epochs_entry = ttk.Entry(epochs_frame, textvariable=epochs, width=12, justify=CENTER).pack()
#Test size elements
test_size_frame = ttk.Frame(model_entry_frame)
test_size_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["test_size"]))
test_size_frame.pack(side=LEFT)
test_size_label = ttk.Label(test_size_frame, text="Test Size").pack()
test_size_entry = ttk.Entry(test_size_frame, textvariable=test_size, width=12, justify=CENTER).pack()
#Batch size elements
batch_size_frame = ttk.Frame(model_entry_frame)
batch_size_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["batch_size"]))
batch_size_frame.pack(side=LEFT)
batch_size_label = ttk.Label(batch_size_frame, text="Batch Size").pack()
batch_size_entry = ttk.Entry(batch_size_frame, textvariable=batch_size, width=12, justify=CENTER).pack()
#Random state elements
random_state_frame = ttk.Frame(model_entry_frame)
random_state_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["random_state"]))
random_state_frame.pack(side=LEFT)
random_state_label = ttk.Label(random_state_frame, text="Random State").pack()
random_state_entry = ttk.Entry(random_state_frame, textvariable=random_state, width=12, justify=CENTER).pack()

#MARK: Model options
options_frame = ttk.Frame(model_frame, padding="2 5 2 2")
options_frame.pack()
dropdown = ttk.Combobox(options_frame,
                        values=model_types,
                        textvariable=model_type,
                        justify=CENTER,
                        # validatecommand=updateSelection,
                        state='readonly')
#Pre-select the default value
dropdown.set(model_types[0])
dropdown.bind('<<ComboboxSelected>>', updateSelection)
dropdown.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["dropdown_box"]))
dropdown.pack(side=LEFT, padx=2)
#Call show matrix function from main thread
show_matrices_button = ttk.Button(options_frame, text="Show Matrices", command=c.showMatrix, state=DISABLED)
show_matrices_button.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["show_matrix_button"]))
show_matrices_button.pack(side=LEFT, padx=2)
slow_features_box = ttk.Checkbutton(options_frame, variable=use_slow_features, text="Use Slow Features")
slow_features_box.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["slow_features_box"]))
slow_features_box.pack(side=LEFT, padx=2)

#MARK: Scrolling Frame
#Scrolling center frame
file_frame = ttk.Frame(root, padding="2 0 2 0")
#Set the hint text on a mouse hover event
file_frame.bind('<Enter>', lambda a: hint_text.set(HINT_TEXT["file_path_label"]))
file_frame.pack(padx=5, anchor=W, fill=BOTH, expand=True)
#Create the scrolled frame
file_path_frame_helper = ScrolledFrame(file_frame, width=400, height=0)
#Needs a helper frame to display the contents in
file_path_frame = file_path_frame_helper.display_widget(Frame)
file_path_frame_helper.pack(anchor=W, expand=True, fill=BOTH)
model_loaded_label = ttk.Label(file_path_frame, textvariable=model_loaded_text, justify=LEFT)
file_frame.bind('<Configure>', lambda a: setWraplength(a, label="model_loaded_label"))
model_loaded_label.pack(padx=3, fill=X, expand=True)
#This label shows the text above the paths
file_text_label = ttk.Label(file_path_frame, text="Selected files:")
file_text_label.pack(padx=3, pady=(2,0), anchor=W)
#This label shows the file paths
file_path_label = ttk.Label(file_path_frame, textvariable=file_path)
file_path_label.pack(expand=True, anchor=W, padx=3)
#This frame shows the predicted genres
genre_frame = ttk.Frame(file_path_frame)
genre_text_label = ttk.Label(genre_frame, text="Predicted Genres:")
genre_text_label.pack(padx=3, pady=(2,0), anchor=W)
genre_label = ttk.Label(genre_frame, textvariable=predicted_genres)
genre_label.pack(padx=3, anchor=W)
#Bind scrolling events to the respective windows
#If scrolling should only be possible inside the frame, it needs to be bound to each label inside
# file_path_frame_helper.bind_scroll_wheel(file_path_frame_helper)
# file_path_frame_helper.bind_scroll_wheel(file_path_label)
file_path_frame_helper.bind_scroll_wheel(root)
file_path_frame_helper.bind_arrow_keys(root)

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

#Load a model at startup if it is saved in the starting directory
try:
    #If multiple models are in the starting directory, load the first one found
    model_path = glob("*.keras")[0]
    c.loadModel(model_path)
    use_slow_features.set(c.used_slow_features)
    hint_text.set(f"Loaded model \"{model_path}\" from the default directory.")
    print(f"Loaded model \"{model_path}\" from the default directory.")
    #Update the model loaded text
    model_loaded_text.set(f"Model status:\nLoaded model \"{model_path}\" from the default directory.\nTest Accuracy: {c.test_acc*100:.2f}%, Test Loss: {c.test_loss:.4f}")
    #Set the model available flag
    updateUI("loaded_model")
    model_available = True
#No model found/malformed file
except:
    #Update the root window so the size change works
    root.update()
    #Show model options if no model is loaded at startup
    updateUI("show_options")
    model_loaded_text.set("Model status:\n" + NO_MODEL_MSG)
    print("No model loaded")

root.mainloop()