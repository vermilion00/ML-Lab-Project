SAVE_DIR = ".temp_classifier"
SAVE_FILE = SAVE_DIR + "/" + "features.csv"
FORBIDDEN_CHARS = r'[\\/:*?"<>|]' #Not allowed in windows file names

HINT_TEXT = {
    "program_start": "If you have a model file placed in the default directory, it will be loaded at program start. To train a model, either load and extract features from audio files or select a pre-extracted feature file, then click \"Train model\". To classify audio, you need to have a model loaded. Select the audio files and extract their features before predicting their genre.",
    "load_file_button": "Click to select the audio file(s) you wish to import.",
    "load_csv_file_button": "Click to select a pre-extracted feature list.",
    "load_folder_button": "Click to select a folder containing the audio files you wish to import.",
    "extract_button": "Click to begin extracting features from the selected files. Only available when audio files are selected.",
    "save_button": "Click to save the extracted features to a specified location.",
    "file_path_label": "Shows the selected file path.",
    "progress_label": "Shows the progress on the currently running process.",
    "hint_label": "Shows helpful hints as well as process progress.",
    "read_csv": "Loaded features from CSV file.",
    "no_files_found": "No audio or csv files found in directory.",
    "extraction": "Extraction completed.",
    "training": "Training completed",
    "learning_rate": "Input the learning rate for the model as a float.",
    "epochs": "Input the epoch amount for the model as an Int.",
    "batch_size": "Input the batch size for the model as an Int.",
    "random_state": "Input the random state for the model as an Int.",
    "test_size": "Input the test size for the model as a float.",
    "train_model_button": "Click to train the model with the loaded feature set using the parameters set above. This also scales the features and splits the feature set. The process will take a while.",
    "predict_genre_button": "Click to predict the genre of the loaded feature set(s). Currently non-functional.",
    "save_model_button": "Click to save the current model to a file. Only available when a model is loaded.",
    "load_model_button": "Click to load a trained model from a .keras file."
}

HEADER = (
    'filename',
    'length',
    'chroma_stft_mean',
    'chroma_stft_var',
    'rms_mean',
    'rms_var',
    'spectral_centroid_mean',
    'spectral_centroid_var',
    'spectral_bandwidth_mean',
    'spectral_bandwidth_var',
    'rolloff_mean',
    'rolloff_var',
    'zero_crossing_rate_mean',
    'zero_crossing_rate_var',
    'harmony_mean',
    'harmony_var',
    'perceptr_mean',
    'perceptr_var',
    'tempo',
    'mfcc1_mean',
    'mfcc1_var',
    'mfcc2_mean',
    'mfcc2_var',
    'mfcc3_mean',
    'mfcc3_var',
    'mfcc4_mean',
    'mfcc4_var',
    'mfcc5_mean',
    'mfcc5_var',
    'mfcc6_mean',
    'mfcc6_var',
    'mfcc7_mean',
    'mfcc7_var',
    'mfcc8_mean',
    'mfcc8_var',
    'mfcc9_mean',
    'mfcc9_var',
    'mfcc10_mean',
    'mfcc10_var',
    'mfcc11_mean',
    'mfcc11_var',
    'mfcc12_mean',
    'mfcc12_var',
    'mfcc13_mean',
    'mfcc13_var',
    'mfcc14_mean',
    'mfcc14_var',
    'mfcc15_mean',
    'mfcc15_var',
    'mfcc16_mean',
    'mfcc16_var',
    'mfcc17_mean',
    'mfcc17_var',
    'mfcc18_mean',
    'mfcc18_var',
    'mfcc19_mean',
    'mfcc19_var',
    'mfcc20_mean',
    'mfcc20_var',
    'label'
)

OVERWRITE_FAILED_MSG = "Failed to save the file, as the file you're trying to overwrite is likely opened already. Close the file before trying again!"
NO_DATA_MSG = "No data to save, extract some files first!"
NO_FILES_MSG = "No files selected, select some files to extract the features from first!"
READ_CSV_FAILED_MSG = "Failed to read CSV file, make sure it's in the correct format!"
MALFORMED_CSV_MSG = "CSV is malformed, make sure the features are laid out correctly!"
ALREADY_EXTRACTED_MSG = "The currently loaded audio files have already been extracted. Do you really want to extract their features again?"
ALREADY_SAVED_MSG = "The currently loaded features have already been saved. Would you like to save them again?"
SEARCH_SUBFOLDERS_MSG = "The selected folder contains no files to load, but subfolders have been detected. Would you like to recursively search all subfolders for audio files?"
INVALID_INPUT_MSG = "One or more input fields contain invalid inputs! Take a look at the hint regarding each input field to see what field accepts what input, and use a dot as a decimal separator for floating point numbers."
MODEL_ALREADY_SAVED_MSG = "The loaded model has already been saved in it's current state. Would you like to save it again?"
MODEL_LOADING_FAILED_MSG = "An error has occured when loading the model. Make sure you have selected the correct file."
MODEL_ALREADY_TRAINED_MSG = "The current model has already been trained with these parameters and features. Would you like to train it again? You can change the parameters using the input fields above."

EXTRACTION_STEPS = 5