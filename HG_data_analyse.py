import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import streamlit as st
import math
import os
import random
    
def fft_filter(data, d = 150, band = [0,200]):
    '''
    fourier transform filter
    data (array like): data to filter
    d (int): frequency of data
    band (2 element sequence): the band of frequencies to keep (all others will be deleted)
    
    returns: filtered data
    '''
    
    modes = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=(1/d))
    
    band = ((np.abs(freqs) >= band[0]) & (np.abs(freqs) <= band[1]))
    filtered_modes = np.copy(modes)
    filtered_modes[~band] = 0
    data = np.fft.irfft(filtered_modes)
    
    return data


def detect_peaks(data, timestamps, distance = 1):
    '''
    creates a data frame containing all peaks and timestamps
    
    data (array like): data containing peaks
    timestamps (pandas series): timestamps matching data
    distance (int): minimum distance between individual peaks
    
    returns: dataframe with peak timestamps and vallues
    '''
    thresh = max(data)*0.4
    peak_index, _ = find_peaks(data, distance = distance, height = thresh)
    
    peaks = pd.DataFrame(timestamps.iloc[peak_index])
    peaks['value'] = data.iloc[peak_index]
    
    return peaks


def calculate_bpm(timestamps, smooth = 1):    
    '''
    calculates the progressive bpm of a dataset
    
    timestamps(pandas series): timestamps of R peaks from ECG data in miliseconds
    smooth (int): size of rolling average winsdow
    
    returns bpm(array): bpm per timestamp
    '''
    
    time_diff = timestamps.diff()
    bpm = (1000 / time_diff) * 60
    bpm = bpm.rolling(smooth, min_periods = 1).mean()
        
    return bpm


def calculate_HRV(timestamps):
    '''
    calculates heart rate variability with 
    with Root mean square of succesive differences (RMSSD)
    
    timestamps (pandas series): timestamps of R peaks from ECG data in miliseconds
    
    return(int): heart rate variability
    ''' 
    
    time_diff = timestamps.diff()
    HRV = np.sqrt(np.mean(np.square( time_diff.diff() )))
    
    return HRV


def get_filenames(path, appendix):
    '''
    makes a list of all ecg.csv filenames that have to be analyzed
   
    path (str): path to the folder with the csv files
    appendix (str): file names appendix to filter on
    
    returns (dataframe): DF containing all filenames 
    '''
    
    #name format- year month day T Hour min .... sec Z_ movesenseID _ecg_stream.csv
    #name format- year month day T Hour min .... sec Z_ movesenseID _heartRate_stream.csv
    
    fnames = pd.DataFrame(os.listdir(path), columns = ['file name'])
    
    fnames['filt'] = fnames['file name'].str[-len(appendix):] == appendix
    fnames = pd.DataFrame(fnames['file name'].iloc[fnames['filt']], columns = ['file name'])
    fnames['file name'] = fnames['file name']
    
    return fnames 

@st.cache_data
def analyze_ecg_data(path, appendix, d=1, band=[0,200], distance=1, smooth=1):
    '''
    reads given ecg.csv files and returns array of average bpm and HRV for every ecg reading
    
    path (str): path to the folder with the csv files
    appendix (str): file names appendix to filter on
    d (int): frequency of ecg data
    band (2 element sequence): the band of frequencies to keep (all others will be deleted)
    distance (int): minimum distance between individual R peaks
    smooth (int): size of rolling average winsdow for smoothing bpm curve

    returns: dataframe of average bpm and HRV for every ecg reading with ID as index 
    '''

    fnames = get_filenames(path, appendix)

    results = pd.DataFrame(columns = ['avg_bpm', 'HRV'])
    for n, fname in enumerate(fnames['file name']):
        
        #reading in data
        data_ecg = pd.read_csv(path + fname)
        data_ecg['timestamp'] -= data_ecg['timestamp'].iloc[0]
        data_ecg['sample'] -= data_ecg['sample'].mean()
            
        data_ecg['filtered'] = fft_filter(data_ecg['sample'], d = d, band = band) #data cleanup
        peaks = detect_peaks(data_ecg['filtered'], data_ecg['timestamp'], distance = distance) #peak detection
        peaks['bpm'] = calculate_bpm(peaks['timestamp'], smooth = smooth) #peak procesing
      
        avg_bpm = peaks['bpm'].mean()
        HRV = calculate_HRV(peaks['timestamp'])
        
        #storing results
        ID = fname[:13]
        res = pd.DataFrame([[avg_bpm, HRV]], columns = ['avg_bpm', 'HRV'], index = [ID])
        results = pd.concat([results, res]) 
        
        #plotting (for debuging and visualisation)
        '''

        start = 500
        stop = 10000
        
        nxy = math.ceil(math.sqrt(len(fnames)))
        plt.subplot(nxy, nxy, n+1)
        plt.xticks([])
        plt.yticks([])
        #plt.ylabel(XXX, fontsize = 6)
        plt.title(f'{fname[:13]}, n:{n}', fontsize = 6)
        
        
        plt.plot(data_ecg['timestamp'].iloc[start:stop], data_ecg['sample'].iloc[start:stop]) 
        #plt.plot(data_ecg['timestamp'].iloc[start:stop], data_ecg['filtered'].iloc[start:stop])
        #plt.plot(peaks['timestamp'].loc[start:stop], peaks['filtered'].loc[start:stop], linestyle = '', marker = 'x')
        #plt.plot(peaks['timestamp'].loc[start:stop], peaks['bpm'].loc[start:stop]*6)
        #plt.plot(peaks['timestamp'].loc[start:stop], peaks['bpm_smooth'].loc[start:stop]*6)
        #plt.axhline(height, color='r', linestyle='-')

        plt.show()
        '''
    return results

@st.cache_data
def import_HG_data(filename):
    '''
    reads hartstikke gezond data from csv file and sets ID as index
    ID format: year month day T Hour min
    
    filename(str): name of csv file 
    
    returns: HG data in a dataframe
    
    '''
    
    data = pd.read_csv(filename, delimiter = ';')   

    year = data['Datum'].str[-4:]
    month = '0' + data['Datum'].str[3]
    day = data['Datum'].str[:2] 
    hour = data['Tijd'].str[:2]
    minute = data['Tijd'].str[-2:]

    data['ID'] = year + month + day + 'T' + hour + minute
    data = data.set_index('ID')
    
    return data
