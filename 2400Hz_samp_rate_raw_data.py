# 2400 Hz sampling rate raw data
# %%
import numpy as np
import os
import pandas as pd
import EfficentDataProcess as EDP
from scipy.signal import lfilter, lfilter_zi, savgol_filter
import functions_time_points_x_68 as fut

import math
import random

# %%

harmonics_remover_b     = np.zeros((49,), dtype=np.float32)
harmonics_remover_b[0]  = 0.86327126
harmonics_remover_b[-1] = -0.86327126

harmonics_remover_a     = np.zeros((49,), dtype=np.float32)
harmonics_remover_a[0]  = 1
harmonics_remover_a[-1] = -0.72654253


FSs = [2400] #Sampling frequencies
decimation_factor = 24
win_len = 200 #ms
step = 50 #ms meaning windows will have 75% overlap
n_epochs = 50
batch_len = 5 #seconds 
batch_len_idx = int(batch_len/(step/1000))
# %%

# define the filter class. Both the harmonics removal filter and the envelope extraction filter are instances of this class.
class Filter():
    '''
    A custom filter class that filters data and keeps track of the zi as filtering happens
    '''
    def __init__(self, filter_num_coeffs ,filter_denom_coeffs, data_total_channels : int ):
        self.b  = filter_num_coeffs         # filter numerator coefficients
        self.a  = filter_denom_coeffs       # filter denominator coefficients
        # Initial conditions for the filter delays
        # Since the data is multi-dimensional, the initial condition array should be constructed for each one of the data channels
        # The idea: first create zi for one channel and then repeat it to all channels. Final zi is of shape (number_of_filter_delays, number_of_data_channels) 
        self.zi = np.tile(lfilter_zi( self.b, self.a ), (data_total_channels, 1)).T                         
    
    def filt(self, chunk_data, zi):
        '''
        Always assume filtering in 'increasing row' direction
        '''
        return lfilter(self.b, self.a, chunk_data, zi=zi, axis=0)
# %%
# reordered data columns
reordered_data_columns = ['HA-2015.08.05_channel{}'.format(i) for i in range(1,65)]
reordered_data_columns.extend(["ref_q_w", "ref_q_x", "ref_q_y", "ref_q_z", 'subject_id', 'condition'])
reordered_label_columns = ['thumb_op','thumb_fl3','index_fl2', 'middle_fl2', 'ring_fl2', 'little_fl2', "flexion", "pronation", 'subject_id', 'condition']

# %%
def prepare_data(portiion_emg_path_array, portion_wrist_path_array, portion_finger_path_array):

    for path_idx, (emgPath, wristPath, fingerPath) in enumerate(zip(portiion_emg_path_array, portion_wrist_path_array, portion_finger_path_array)):

        # filters created with pre-defined parameters
        envelope_extract_filter   =   Filter( np.array([0.3586042 , 0.21028036, 0.20656252, 0.16322745, 0.0981327], dtype=np.float32), 
                                      np.array([1], dtype=np.float32), data_total_channels=64 )
        harmonics_removal_filter  =   Filter(harmonics_remover_b, harmonics_remover_a, data_total_channels=64)

        '''
        Load the aligned data. Data is already resampled at 2400Hz

        '''
        emgData = pd.read_csv(emgPath)
        fingerData = pd.read_csv(fingerPath)
        wristData = pd.read_csv(wristPath)

        '''
        Filter 50Hz harmonics and obtain envelope
        '''
        ###      remove harmonics       ###
        emgData.loc[:, "HA-2015.08.05_channel1":"HA-2015.08.05_channel64"], harmonics_removal_filter.zi = harmonics_removal_filter.filt(emgData.loc[:,"HA-2015.08.05_channel1":"HA-2015.08.05_channel64"], zi=harmonics_removal_filter.zi)
        ###     low-pass to get envelope    ###
        emgData.loc[:, "HA-2015.08.05_channel1":"HA-2015.08.05_channel64"], envelope_extract_filter.zi  = envelope_extract_filter.filt(np.absolute(emgData.loc[:,"HA-2015.08.05_channel1":"HA-2015.08.05_channel64"]), zi=envelope_extract_filter.zi)

        ## filter finger angles
        fingerData.loc[:, 'thumb_op':'little_fl2'] = savgol_filter( fingerData.loc[:, 'thumb_op':'little_fl2'], 11, 3, axis=0)
        '''
        Merge some quaternion components to data and add flexion and pronation to new columns in labels.
        After the merge, completeData will have 68 columns: 64 EMG channels + 4 reference quaternion components
        '''
        completeData = emgData.join( wristData.loc[:, 'ref_q_w':'ref_q_z'] )
        # shift subject_id and condition to the last columns
        completeData = completeData.reindex(columns=reordered_data_columns)

        # fake flexion and pronation
        completeLabel = fingerData
        completeLabel['flexion'] = 0.5
        completeLabel['pronation'] = 0.5
        completeLabel = completeLabel.reindex(columns=reordered_label_columns)


        del emgData
        del fingerData
        del wristData

        #Cut data
        print('Create three sets: train, validation and test')
        # Tables is a list of input dataframes. In our case, the list has only ONE dataframe; this dataframe has a column named "set"
        # Labels is a single dataframe of labels corresponding to the input dataframe. Labels also has a column named "set"
        # "set" column contains flags indicating whether a row of data/label is for train, test or validation
        Tables, Labels = EDP.hh_cut_data_on_percentage([completeData], completeLabel, val_percentage = 1/6, test_percentage = 1/6)

        '''
        Blanking (stimulation, blanking itself, integration back to the signal of the blanked parts)
        '''
        # we need to blank only "test" part, so we make a subset of Tables and also remove a column "set"
        # Tables is pandas df -> transform into numpy array 
        columns_to_remove = ["condition", "set", "subject_id"]
        set_column_position = Tables[0].columns.get_loc("set")
        subject_id_column_position = Tables[0].columns.get_loc("subject_id")
        condition_column_position = Tables[0].columns.get_loc("condition")
        
        Tables_test = Tables[0][Tables[0]["set"] == "test"].copy()
        Tables_test = Tables_test.drop(columns = columns_to_remove)
        Tables_test_np = Tables_test.to_numpy()
        
        # stimulation imitation
            # stimulation_segemnts - a list of 2D arrays, each representing a random segment (stimulation) for all channels
            # time_frames - a list of lists, where start and end time indexes are placed for each stimulation segment
        stimulation_segments, time_frames = fut.extract_non_overlapping_segments(Tables_test_np)
        
        # blanking
            # blanked_with0_stimulation_segments - list with 2D numpy arrays with blanked signal, zeros are preserved in the arrays
            # blanked_no0_stimulation_segments - list with 2D numpy arrays with blanked signal, zeros are NOT preserved in the arrays
            # _ - lists with lists, each small list represents the collelction of blanking time frames for one segment
            # blanked_time_frames_ind_in_segments - lists with lists, each small list represents the collelction of blanking time frames (in the forms of indexes, continuos numbering) for one segment
        blanked_with0_stim_segs, blanked_no0_stim_segs, _, blanked_time_frames_ind_in_segments = fut.blank_emg_segments(stimulation_segments = stimulation_segments, time_frames = time_frames, pulse_width_us = 160, stimulation_freq_Hz = 10, add_blank_us = 0, sampling_rate_Hz = 2400)
        
        # integration of blanked segments back into the data
            # completeData_with0_np - 2D numpy array representing blanked EMG data with preserved zeros
            # completeData_no0_np - 2D numpy array representing blanked EMG data with NOT preserved zeros
            # _ - alternatively made completeData_no0_np
        completeData_with0_np, completeData_no0_np, _ = fut.blanked_segments_integration(initial_emg_data = Tables_test_np, stimulation_segments_with0 = blanked_with0_stim_segs, stimulation_segments_no0 = blanked_no0_stim_segs, time_frames = time_frames)

        '''
        Downsample data to 100Hz
        '''
        
        # iloc is only for Pandas
            # recreation of pandas df with "test" column
        completeData_with0 = pd.DataFrame(completeData_with0_np, columns = Tables_test.columns)
        columns_names = completeData_with0.columns
        num_rows_completeData_with0 = len(completeData_with0[columns_names[0]]) 
        completeData_with0.insert(subject_id_column_position, "subject_id", ["test"] * num_rows_completeData_with0)
        completeData_with0.insert(condition_column_position, "condition", ["test"] * num_rows_completeData_with0)
        completeData_with0.insert(set_column_position, "set", ["test"] * num_rows_completeData_with0)
        
        completeData_no0 = pd.DataFrame(completeData_no0_np, columns = Tables_test.columns)
        num_rows_completeData_no0 = len(completeData_no0[columns_names[0]]) 
        completeData_no0.insert(subject_id_column_position, "subject_id", ["test"] * num_rows_completeData_no0)
        completeData_no0.insert(condition_column_position, "condition", ["test"] * num_rows_completeData_no0)
        completeData_no0.insert(set_column_position, "set", ["test"] * num_rows_completeData_no0)
            # decimation of test Pandas df
        Tables_completeData_with0  = completeData_with0.iloc[::decimation_factor, : ] # test
        Tables_completeData_no0 = completeData_no0.iloc[::decimation_factor, : ] # test
            # decimation of validation + train Pandas df
        Tables_train_val = Tables[0][Tables[0]["set"].isin(["train", "validation"])].copy()
        Tables_train_val = Tables_train_val.iloc[::decimation_factor, : ] # val and train
        
        # we also discussed an option where we include zeros in training, but here we do not do this for now
        
            # decimation of Labels for train + validation
        Labels_train_val = Labels[Labels["set"].isin(["train", "validation"])].copy()
        Labels_train_val = Labels_train_val.iloc[::decimation_factor, : ]
        
            # decimation of Labels_with0_test 
        Labels_with0_test = Labels[Labels["set"].isin(["test"])].copy()
        Labels_with0_test = Labels_with0_test.iloc[::decimation_factor, : ]
        
        Labels_with0_test_no_set = Labels_with0_test.drop(columns = columns_to_remove)
        Labels_test_np = Labels_with0_test_no_set.to_numpy()
            # removing rows of blanked regions from the Labels 
        Labels_no0_test_np = fut.removing_blanked_regions_from_labels(label_array = Labels_test_np, blank_time_frames = blanked_time_frames_ind_in_segments)
            # getting back to pandas df
        Labels_no0_test = pd.DataFrame(Labels_no0_test_np, columns = Labels_with0_test_no_set.columns)
        columns_names_Labels = Labels_no0_test.columns
        num_rows_Labels_no0_test = len(Labels_no0_test[columns_names_Labels[0]]) 
        
        set_column_position_Labels = Labels.columns.get_loc("set")
        subject_id_column_position_Labels = Labels.columns.get_loc("subject_id")
        condition_column_position_Labels = Labels.columns.get_loc("condition")
        
        Labels_no0_test.insert(subject_id_column_position_Labels, "subject_id", ["test"] * num_rows_Labels_no0_test)
        Labels_no0_test.insert(condition_column_position_Labels, "condition", ["test"] * num_rows_Labels_no0_test)
        Labels_no0_test.insert(set_column_position_Labels, "set", ["test"] * num_rows_Labels_no0_test)

                # for now we have:
                
                # Labels_no0_test - decimated to 100 Hz pandas df with Labels for "test", where zeros are not preserved
                # Labels_with0_test - decimated to 100 Hz pandas df with Labels for "test", where zeros are preserved
                # Labels_train_val - decimated to 100 Hz pandas df with Labels for "train" and "validation"
                # Tables_train_val - decimated to 100 Hz pandas df with "train" and "validation"
                # completeData_with0 - 2400 Hz, pandas df, only "test" , rows with zeros are preserved
                # completeData_no0 - 2400 Hz, pandas df, only "test" , rows with zeros are NOT preserved
                # Tables_completeData_with0 - decimated to 100 Hz pandas df, only "test" , rows with zeros are preserved
                # Tables_completeData_no0 - decimated to 100 Hz pandas df, only "test" , rows with zeros are NOT preserved 
                
                
                
        # concatenate downsampled Tables with0
        completeTables_with0 = pd.concat([Tables_completeData_with0,Tables_train_val], axis = 0, ignore_index = True)
        # concatenate downsampled Tables no0
        completeTables_no0 = pd.concat([Tables_completeData_no0,Tables_train_val], axis = 0, ignore_index = True)
        # concatenate downsampled Labels with0
        Labels_with0 = pd.concat([Labels_train_val, Labels_with0_test], axis = 0, ignore_index = True)
        # concatenate downsampled Labels no0
        Labels_no0 = pd.concat([Labels_train_val, Labels_no0_test], axis = 0, ignore_index = True)
        
        print('Create overlapping time windows')
        # Train, Val and Test are all lists of dataframes (overlapping windows). 
        # 2nd param is a list containing the sampling frequency of the input dataframe. Since we downsampled to 100Hz, here we just put [2400/24]
        # 3rd param: window length in float, representing the length of window in seconds
        # 4th param: time in which there is new data, in seconds. E.g. For a 200ms window, putting 0.07s here will give you 140ms of overlapping data and 70ms of new data in the next window  
        #Train,Val,Test = EDP.hh_cut_time_windows(Tables, [FSs[0]/decimation_factor], win_len/1000, step/1000)
        Train,Val,Test_with0 = EDP.hh_cut_time_windows([completeTables_with0], [FSs[0]/decimation_factor], win_len/1000, step/1000)
        _, _,Test_no0 = EDP.hh_cut_time_windows([completeTables_no0], [FSs[0]/decimation_factor], win_len/1000, step/1000)

        print('Align targets on windows')
        # hh_align_with_windows will produce a dataframe of labels
        Train_Labels_with0    = EDP.hh_align_with_windows(Train,Labels_with0)
        Val_Labels_with0      = EDP.hh_align_with_windows(Val,Labels_with0)
        Test_Labels_with0 = EDP.hh_align_with_windows(Test_with0,Labels_with0)
        
        #Train_Labels_no0    = EDP.hh_align_with_windows(Train,Labels_no0)
        #Val_Labels_no0      = EDP.hh_align_with_windows(Val,Labels_no0)
        Test_Labels_no0     = EDP.hh_align_with_windows(Test_no0,Labels_no0)

        min_val = 0
        max_val = 1

        # Rescale targets
        print(f'Rescale labels between {min_val} and {max_val}')
        # assuming thumb_op ... little_fl2 is ordered like this.
        # min-max rescaling of labels to 0~1
        Train_Labels_with0.loc[:, 'thumb_op' : 'little_fl2']  = EDP.h_scale_labels(  Train_Labels_with0.loc[:,  'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        Test_Labels_with0.loc[:, 'thumb_op' : 'little_fl2']   = EDP.h_scale_labels(  Test_Labels_with0.loc[:,   'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        Val_Labels_with0.loc[:, 'thumb_op' : 'little_fl2']    = EDP.h_scale_labels(  Val_Labels_with0.loc[:,    'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        
        Test_Labels_no0.loc[:, 'thumb_op' : 'little_fl2']   = EDP.h_scale_labels(  Test_Labels_no0.loc[:,   'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        #Train_Labels_no0.loc[:, 'thumb_op' : 'little_fl2']  = EDP.h_scale_labels(  Train_Labels_no0.loc[:,  'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        #Val_Labels_no0.loc[:, 'thumb_op' : 'little_fl2']    = EDP.h_scale_labels(  Val_Labels_no0.loc[:,    'thumb_op' : 'little_fl2'], min_val, max_val, reversed=True)
        

        print('expand dim of input to 3D for CNN')
        #reshape input as CNN input needs to be 3d
        for i in range(len(Train[0])):
            Train[0][i] =  np.expand_dims(Train[0][i].values, axis = -1)

        for i in range(len(Val[0])):
            Val[0][i] =  np.expand_dims(Val[0][i].values, axis = -1)
            
        for i in range(len(Test_with0[0])):
            Test_with0[0][i] = np.expand_dims(Test_with0[0][i].values, axis = -1)
            
        for i in range(len(Test_no0[0])):
            Test_no0[0][i] = np.expand_dims(Test_no0[0][i].values, axis = -1)


        # write numpy binary files to disk
        dataDir = os.path.dirname( emgPath )
        cut_data_dir = os.path.join( dataDir, "2_cut_data" )
        if not os.path.exists(cut_data_dir):
            os.mkdir(cut_data_dir)

        print('Save chunked data as .npy')
        for i, idx in enumerate(range(0, len(Train[0]), batch_len_idx)):
            np.save(cut_data_dir + f'/Train_{i}', Train[0][idx:idx+batch_len_idx])
            np.save(cut_data_dir + f'/Train_Labels_{i}', Train_Labels_with0.values[idx:idx+batch_len_idx])

        for i,idx in enumerate(range(0,len(Val[0]),batch_len_idx)):
            np.save(cut_data_dir + f'/Val_{i}', Val[0][idx:idx+batch_len_idx])
            np.save(cut_data_dir + f'/Val_Labels_{i}', Val_Labels_with0.values[idx:idx+batch_len_idx])

        #for i,idx in enumerate(range(0,len(Test[0]),batch_len_idx)):
            #np.save(cut_data_dir + f'/Test_{i}', Test[0][idx:idx+batch_len_idx])
            #np.save(cut_data_dir + f'/Test_Labels_{i}', Test_Labels.values[idx:idx+batch_len_idx])
            
        for i,idx in enumerate(range(0,len(Test_no0[0]),batch_len_idx)):
            np.save(cut_data_dir + f'/Test_no0_{i}', Test_no0[0][idx:idx+batch_len_idx])
            np.save(cut_data_dir + f'/Test_Labels_no0_{i}', Test_Labels_no0.values[idx:idx+batch_len_idx])
            
        for i,idx in enumerate(range(0,len(Test_with0[0]),batch_len_idx)):
            np.save(cut_data_dir + f'/Test_with0_{i}', Test_with0[0][idx:idx+batch_len_idx])
            np.save(cut_data_dir + f'/Test_Labels_with0_{i}', Test_Labels_with0.values[idx:idx+batch_len_idx])

        print(f"Data saved in '{cut_data_dir}'")


# %%
if __name__ == '__main__':

    # TODO: Define root folder that contains data folders for each recording
    #dataRoot = r"D:\EMG_TENS_LIB\PreparedData"
    #dataRoot = r"/Users/Lenovo/Desktop/franklin/aiden_data/2400HzRaw/20221007_FingerWrist01_2400Hz"
    dataRoot = r"/Users/Lenovo/Desktop/franklin/aiden_data/2400HzRaw/no_hidden_files"

    # array with the paths of the Tables.csv and the Labels.csv files
    emg_path_array = []
    wrist_path_array = []
    finger_path_array = []

    current_dirs = []

    # make sure dataRoot only contains folders of aligned data
    # Each folder of aligned data should contain only EMG.csv, Finger.csv and Wrist.csv
    for recordingDir in os.listdir(dataRoot):
        recordingDirFullPath = os.path.join( dataRoot, recordingDir )
        print(f"recording full path: {recordingDirFullPath}")
        for fileName in os.listdir(recordingDirFullPath):
            if fileName == "EMG.csv":
                emg_path_array.append(os.path.join(recordingDirFullPath, fileName).replace(os.sep, '/'))
            elif fileName == "Finger.csv":
                finger_path_array.append(os.path.join( recordingDirFullPath, fileName ).replace(os.sep, '/'))
            elif fileName == "Wrist.csv":
                wrist_path_array.append(os.path.join( recordingDirFullPath, fileName ).replace(os.sep, '/'))

    # Extract length of the databases and create Val and Test datasets
    print('EMG Path:', emg_path_array, sep='\n')
    print('Wrist Path:', wrist_path_array, sep='\n')
    print('finger Path:', finger_path_array, sep='\n')

    if (len(emg_path_array) != len(finger_path_array)) or (len(emg_path_array) != len(wrist_path_array)):
        raise ValueError('wrong number of files retrieved')

    emg_path_array.sort()
    finger_path_array.sort()
    wrist_path_array.sort()
    print("start data processing\n")

    # process data
    prepare_data(emg_path_array, wrist_path_array, finger_path_array)

    print("process complete")
    
# %%
    # import the model
import tensorflow as tf
print(tf.__version__)

model_path = '/Users/Lenovo/Desktop/franklin/model/aiden_pretrained/ad_smaller_model'
loaded_model = tf.keras.models.load_model(model_path)
loaded_model.summary() 


# %%
# work with the model 
    # for one file
    

# %%
    # for concatenated files