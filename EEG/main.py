from eeg_preprocessing import *
from train_model import *
import os
import multiprocessing

data_df = data(root = os.getcwd()+'/EEG/data/raw/')

labels_to_drop = ['21', '22', '23'] # drop motor execution data

# iterates over every participant and trains every model (3) for each one individually
for index, row in data_df.iterrows():

    # NFEEG
    save_path_X = row['save_path_X_NFEEG']
    save_path_Y = row['save_path_Y_NFEEG'] 
    
    # multiprocessing for GPU usage // otherwise GPU would run out of memory 
    p = multiprocessing.Process(target = train_nfeeg, args = (save_path_X, save_path_Y, labels_to_drop, 0.2, 42, 8, 5, 0.0001))
    p.start()
    p.join()

    # EEGNet
    save_path_X = row['save_path_X_EEGNet']
    save_path_Y = row['save_path_Y_EEGNet']

    # multiprocessing for GPU usage // otherwise GPU would run out of memory 
    p = multiprocessing.Process(target = train_eegnet, args = (save_path_X, save_path_Y, labels_to_drop, 0.2, 42, 16, 5, 0.0001))
    p.start()
    p.join()

    # FBCNet
    save_path_X = row['save_path_X_FBCNet']
    save_path_Y = row['save_path_Y_FBCNet']

    # multiprocessing for GPU usage // otherwise GPU would run out of memory 
    p = multiprocessing.Process(target = train_fbcnet, args = (save_path_X, save_path_Y, labels_to_drop, 0.2, 42, 16, 5, 0.0001))
    p.start()
    p.join()