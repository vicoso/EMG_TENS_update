# functions for data time_points x 68 (channels)
import random
import numpy as np
import math
import os


# storage of functions


def remove_zero_rows(array):
    #import numpy as np
    # required packs: numpy as np
    # array - 2D numpy array time_points x 68 (channels)
    # removes rows filled with only zeros from a given matrix
    # return matrix without rows, which were filled with only zeros
    non_zero_rows = np.any(array != 0, axis=1)
    
    result_array = array[non_zero_rows, :]
    return result_array


def blank_emg_segments(stimulation_segments: list, time_frames: list, pulse_width_us: int, stimulation_freq_Hz: int, add_blank_us: int, sampling_rate_Hz: int):
    # required packs: math, numpy as np, functions as fu, random
    
    # Tb - period of blanking
    # Tf - period of stimulation
    # 2 * pulse_width < Tb < Tf
    # 1/(2 * pulse_width) > 1/Tb > 1/Tf
    # stimulation segments - list of numpy arrays 2D EMG data time points x channels
    # time_frames - list of lists, where start and end time indexes are placed for each small list - time_frame
    # pulse_width - pulse_width in microseconds
    # stimulation_freq - stimulation frequency (in Hz)
    # add_blank - additional time for blanking (microseconds)
    # sampling_rate - in Hz
    
    # blanks (puts the signal to 0) the emg_segments
    # return:
    # blanked_with0_stimulation_segments and blanked_no0_stimulation_segments
    # blanked_with0_stimulation_segments - list with 2D numpy arrays with blanked signal, zeros are left in the arrays
    # blanked_no0_stimulation_segments - list with 2D numpy arrays with blanked signal, zeros are NOT left in the arrays
    # blanked_time_frames_in_segments - lists with lists, each small list represents the collelction of blanking time frames for one segment
    # blanked_time_frames_ind_in_segments - lists with lists, each small list represents the collelction of blanking time frames for one segment in the forms of indexes
    
    #blank_freq = random.uniform(1/(2*pulse_width*10**(-6)), stimulation_freq) # in Hz
    #random.seed(5)
    
    #import math
    #import numpy as np
    #import functions as fu
    #import random
    
    
    blank_dur = (2*pulse_width_us + add_blank_us)*10**(-6) # in seconds
    #blank_dur_i = math.ceil(blank_dur * sampling_rate_Hz)
    blank_dur_i = blank_dur * sampling_rate_Hz
    
    Ts = 1/stimulation_freq_Hz # period of stimulation in sec
    #Ts_i =  int(Ts * sampling_rate_Hz) # period of stimulation in indexes
    Ts_i =  Ts * sampling_rate_Hz
    
    blanked_with0_stimulation_segments = []
    blanked_no0_stimulation_segments = []
    K = 0
    blanked_time_frames_in_segment = [] # list of blanking time frames for one stimulation segment
    blanked_time_frames_in_segments = [] # list with lists of blanking time frames for one stimulation segment
    
    for segment_counter, stimulation_segment in enumerate(stimulation_segments):
        start_time_index = 0
        end_time_index = stimulation_segment.shape[0] - 1
        #start_index_of_blanking = int(start_time_index + K * Ts_i)
        #end_index_of_blanking = int(start_time_index + K*Ts_i + blank_dur_i)
        start_index_of_blanking = start_time_index + K * Ts_i
        end_index_of_blanking = start_time_index + K*Ts_i + blank_dur_i
    
        while end_index_of_blanking <= end_time_index:
            #start_index_of_blanking = int(start_time_index + K * Ts_i)
            start_index_of_blanking = start_time_index + K * Ts_i
            rows_to_insert_zeros = slice(math.ceil(start_index_of_blanking), math.ceil(end_index_of_blanking) + 1)
            #rows_to_insert_zeros = slice(start_index_of_blanking, end_index_of_blanking + 1)
            stimulation_segments[segment_counter][rows_to_insert_zeros, :] = 0 #modification in place (I think so at least)
            #blank_time_frame = [start_index_of_blanking, end_index_of_blanking + 1]
            blank_time_frame = [math.ceil(start_index_of_blanking), math.ceil(end_index_of_blanking)]
            blanked_time_frames_in_segment.append(blank_time_frame)
            K +=1
            #end_index_of_blanking = int(start_time_index + blank_dur_i + K*Ts_i)
            start_time_index = K*Ts_i
            end_index_of_blanking = start_time_index + blank_dur_i + K*Ts_i
            
            if end_index_of_blanking > end_time_index:
                blanked_with0_stimulation_segments.append(stimulation_segments[segment_counter]) # stimulation_segments[segment_counter] - it is one numpy array
                #segment_with_removed0 = fu.remove_zero_rows(stimulation_segments[segment_counter])
                non_zero_rows = np.any(stimulation_segments[segment_counter] != 0, axis = 1)
                segment_with_removed0 = stimulation_segments[segment_counter][non_zero_rows, :]
                blanked_no0_stimulation_segments.append(segment_with_removed0)
                blanked_time_frames_in_segments.append(blanked_time_frames_in_segment)
                K = 0
                blanked_time_frames_in_segment = []
    
    blank_time_frames_ind_in_segment = []
    blanked_time_frames_ind_in_segments = []
    
    for i, time_frame in enumerate(time_frames):
        # making a list of indexes from a time_frame
        elements_in_between = list(range(time_frame[0] + 1, time_frame[-1]))
        expanded_time_frame = [time_frame[0]] + elements_in_between + [time_frame[-1]]
        
        #num_of_blank_events_in_segment = len(blanked_time_frames_in_segments[i])
        # len(time_frames) == len(blanked_time_frames_in_segments) ?
        for blank_event in blanked_time_frames_in_segments[i]:
            start_ind_blank = expanded_time_frame[blank_event[0]]
            end_ind_blank = expanded_time_frame[blank_event[1]]
            blank_event_frame_ind = [start_ind_blank, end_ind_blank]
            blank_time_frames_ind_in_segment.append(blank_event_frame_ind)
            
        blanked_time_frames_ind_in_segments.append(blank_time_frames_ind_in_segment)
        blank_time_frames_ind_in_segment = []    
            
    return blanked_with0_stimulation_segments, blanked_no0_stimulation_segments, blanked_time_frames_in_segments, blanked_time_frames_ind_in_segments

    
def concatenate_and_save(folder_path, output_folder): # left untouched
    # required packs: os, numpy as np
    # folder_path - the folder, where a one stores the files to concatanate
    # output_folder - the folder where the concatanated data is saved
    
    # iterates through .npy files in a given folder
    # concatanates them and stores in the output_folder
    
    # saves the concataned data in the output_folder
    
    #import os
    #import numpy as np
    
    all_data = []
    folder_name = os.path.basename(folder_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            all_data.append(data)

    concatenated_data = np.concatenate(all_data, axis=0) # along rows

    output_filename = f"{folder_name}_concatenated_data.npy"
    output_file = os.path.join(output_folder, output_filename)
    np.save(output_file, concatenated_data)
    
    
def extract_non_overlapping_segments(emg_data):
    # required packs: random, numpy as np
    
    #import random
    #import numpy as np
    
    # generates non-overlapping random duration segments of 2D EMG data time_points X channels
    # Parameters:
    # emg_data: 2D array representing EMG data (time points x channels)
    
    # returns: emg_segments - a list of 2D arrays, each representing a random segment (stimulation) for all channels,
    # time_frames - a list of lists, where start and end time indexes are placed for each segment
    
    num_time_points, num_channels = emg_data.shape
    last_time_index = num_time_points - 1
    time_frames = [] # the list of time frames
    emg_segments = [] # the list of emg segments
    num_of_start_points = random.randint(1, 200)

    # generating num_of_start_points unique integers with a minimum difference of 3
    start_points = set() # using set we avoid duplicates in start points
    while len(start_points) < num_of_start_points:
        new_start_point = random.randint(0, last_time_index - 1)
        # Check if the new_integer satisfies the conditions
        if all(abs(new_start_point - existing_start_point) >= 4 for existing_start_point in start_points):
            start_points.add(new_start_point)

    list_of_start_points = sorted(list(start_points)) # list of num_time_points starting points
    enum_start_points = list(enumerate(list_of_start_points))

    for i, start_point in enumerate(list_of_start_points):
        if i < (len(list_of_start_points) - 1):
            end_point = random.randint(start_point + 1, enum_start_points[i + 1][1] - 1) #randint includes and left and right
        else:
            end_point = random.randint(start_point + 1, last_time_index)
            
        time_frame = [start_point, end_point]
        emg_segment = emg_data[start_point : end_point + 1, :]
        emg_segments.append(emg_segment)
        time_frames.append(time_frame)
        
    return emg_segments, time_frames

def blanked_segments_integration(initial_emg_data, stimulation_segments_with0: list, stimulation_segments_no0: list, time_frames: list):
    
    #import numpy as np
    # required packs: numpy as np
    # creates blanked EMG data (with 0 and without 0) from initial EMG data
    
    # initial_emg_data - 2D EMG data
    # stimulation_segments_with0 - list with 2D arrays representing blanked stimulation segments with preserved 0
    # stimulation_segments_no0 - list with 2D arrays representing blanked stimulation segments with NOT preserved 0
    # time_frames - list of lists with two elements in each - start and end point of the stimulation

    # returns:
    # concatenated_data_with0 - 2D numpy array representing blanked EMG data with preserved zeros
    # concatenated_data_no0 - 2D numpy array representing blanked EMG data with NOT preserved zeros

    
    start_ind_init_data = 0
    to_concatenate_data_with0_list = []
    to_concatenate_data_no0_list = []
    last_index_of_init_data = initial_emg_data.shape[0] - 1
    
    for i, time_frame in enumerate(time_frames):
        start_time_index = time_frame[0] # start_time_index of stimulation
        end_time_index = time_frame[1] # end_tine_index of stimulation
        emg_seg_to_keep = initial_emg_data[start_ind_init_data : start_time_index, :] # emg_seg_to_keep - the emg segment, wehre no stimulation happened
        to_concatenate_data_with0_list.append(emg_seg_to_keep)
        to_concatenate_data_no0_list.append(emg_seg_to_keep)
        
        to_concatenate_data_with0_list.append(stimulation_segments_with0[i])
        to_concatenate_data_no0_list.append(stimulation_segments_no0[i])
        
        start_ind_init_data = end_time_index + 1
        
    concatenated_data_with0_inter = np.concatenate(to_concatenate_data_with0_list, axis = 0) # intermediate result
    concatenated_data_no0_inter = np.concatenate(to_concatenate_data_no0_list, axis = 0)    # intermediate result
    
    if initial_emg_data.shape[0] > concatenated_data_with0_inter.shape[0]:
       to_concatenate_data_with0_list.append(initial_emg_data[time_frame[1] + 1 : last_index_of_init_data + 1, :]) 
       concatenated_data_with0_final = np.concatenate(to_concatenate_data_with0_list, axis = 0)
       
       to_concatenate_data_no0_list.append(initial_emg_data[time_frame[1] + 1 : last_index_of_init_data + 1, :])
       concatenated_data_no0_final = np.concatenate(to_concatenate_data_no0_list, axis = 0) 
       
    else:
        concatenated_data_with0_final = concatenated_data_with0_inter
        concatenated_data_no0_final = concatenated_data_no0_inter
    
    # removing zeros from concatenated_data_with0_final to create alt_concatenated_data_no0_final 
    copy_concatenated_data_with0_final = np.copy(concatenated_data_with0_final)
    alt_non_zero_rows = np.any(copy_concatenated_data_with0_final != 0, axis = 1)
    alt_concatenated_data_no0_final = copy_concatenated_data_with0_final[alt_non_zero_rows, :]
            
    return concatenated_data_with0_final, concatenated_data_no0_final, alt_concatenated_data_no0_final
        
    
        
def window_data_2(emg_data, overlap_ms: int, sampling_rate_Hz: int, points_per_win: int):
    # emg_data - 2D EMG array, which needs to be windowed
    # overlap_ms - the size of the overlap between windows in miliseconds 
    # sampling_rate_Hz - sampling rate in Hz
    # points_per_win - the sizw of window in data (time) points
    
    # takes 2D array and turns into 3D (number_of_windows, points_per_window, channels)
    # returns windowed array 3D
    
    #import numpy as np
    #import math
    
    num_of_indexes, num_channels = emg_data.shape
    print("concatenated data shape = ", emg_data.shape)
        #overlap in data points - I am not sure in this calculation
    overlap_points = math.floor((overlap_ms / 1000)* sampling_rate_Hz)
    print("overlap points = ", overlap_points)

    num_windows = math.ceil((num_of_indexes - points_per_win) / (points_per_win - overlap_points))

    print("number of windows = ", num_windows)

    list_of_windows = []
    for j in range(num_windows):
        window = emg_data[j*(points_per_win - overlap_points):points_per_win + j*(points_per_win - overlap_points), :]
        list_of_windows.append(window)

    print("length of the list with windows = ", len(list_of_windows))

    prewindowed_array = np.stack(list_of_windows)
    windowed_array = np.transpose(prewindowed_array, (0, 1, 2))    
    print("shape of the windowed array = ", windowed_array.shape)
    
    return windowed_array

def dewindow_via_indexes_2(emg_data, overlap_ms: int, sampling_rate_Hz: int):
    # required packs: math, numpy as np
    # emg_data - 3D EMG data, (500, 20, 68) 500 windows, 20 points, 68 channels
    # overlap_ms - the overlapping window in ms
    # sampling_rate in Hz
    
    # from 3D array makes 2D array (68, v), which representa deqindowed EMG data
    # returns 2D array - dewindowed EMG data
    
    #import math
    #import numpy as np
    
    overlap_points = math.floor((overlap_ms / 1000)* sampling_rate_Hz) 
    #overlap_points = (overlap_ms / 1000)* sampling_rate_Hz
    # reshaping from (500, 20, 68) to (10 000, 68)
    print("overlap_points = ", overlap_points)
    num_windows, points_per_win, num_channles = emg_data.shape
    reshaped_emg_data = emg_data.reshape(-1, emg_data.shape[2])
        
    shape_of_old_emg = emg_data.shape
    shape_of_new_emg = reshaped_emg_data.shape # 0 - channels, 1 - indexes (data_points)
    print([shape_of_old_emg, shape_of_new_emg])
        #print(shape_of_new_emg)
        
    to_concatenate_data_list = []
    initial_piece = reshaped_emg_data[0:20, :]
    to_concatenate_data_list.append(initial_piece)
    #num_of_start_points = 

    for i in range(1, num_windows + 1):
        piece_to_add = reshaped_emg_data[points_per_win + (i-1) * (points_per_win - overlap_points):points_per_win + i * (points_per_win - overlap_points), :]
        to_concatenate_data_list.append(piece_to_add)

    concatenated_data = np.concatenate(to_concatenate_data_list, axis = 1)
    
    return concatenated_data

def removing_stimulations_from_labels(label_array, time_frames: list): # left untouched
    
    #import numpy as np
    # removes the stimulation time frames fron the labels
    # label_array - 2D EMG array , rows - time points, columns - objects (fingers and wrists)
    # time_frames - a list with lists, where there are two numbers in each - start_ind, end_ind of stimulation
    
    # returns a 2D array of the label with removed rows, which are connected to removed stimulation regions
    time_frames_array = np.array(time_frames)
    
    for start_ind, end_ind in time_frames_array:
        rows_to_insert_zeros = slice(start_ind, end_ind + 1)
        label_array[rows_to_insert_zeros, :] = 0
    
    non_zero_array_mask = np.any(label_array != 0, axis = 1)
    shortened_array = label_array[non_zero_array_mask]
    # well, wrong function since time frames != blanking
    return shortened_array

def removing_blanked_regions_from_labels(label_array, blank_time_frames: list): # left untouched
    #import numpy as np
    # removes the stimulation time frames fron the labels
    # label_array - 2D EMG array , rows - time points, columns - objects (fingers and wrists)
    # blank_time_frames - list with lists, each small list represents the collelction of blanking time frames for one segment
    
    # returns a 2D array of the label with removed rows, which are connected to removed blanked regions
    for blank_time_frame in blank_time_frames:
        #num_of_blank_event_in_seg = len(blank_segment)
        #for i in num_of_blank_event_in_seg:
        time_frames_array = np.array(blank_time_frame)
    
        for start_ind, end_ind in time_frames_array:
            print("start_ind", start_ind)
            print("end_ind", end_ind)
            
            rows_to_insert_zeros = slice(start_ind, end_ind + 1)
            label_array[rows_to_insert_zeros, :] = 0
    
    non_zero_array_mask = np.any(label_array != 0, axis = 1)
    shortened_array = label_array[non_zero_array_mask, :]
    
    return shortened_array