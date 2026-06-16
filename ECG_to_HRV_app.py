import pandas as pd
import streamlit as st
import numpy as np
import HG_data_analyse as da
import plotly.graph_objects as go
import matplotlib.pyplot as plt



data = pd.DataFrame(columns = ['ECG', 'time'])
distance = 25
st.title('data analyse app')
uploaded_file = st.file_uploader("Choose a file", type = 'csv')


if st.button('read file'):
    data = pd.read_csv(uploaded_file)
    data = data[['time', 'ECG']]
   
    data['time'] -= data['time'].iloc[0]
    data['ECG'] -= data['ECG'].mean()
    
    #makes sure dataframe has even number of rows
    #to prevent issues with filtering function for some reason
    #also delete first 20 values to prevent noisy signal
    if len(data) % 2 == 0:   
        data = data.iloc[20:]
    else:
        data = data.iloc[21:]
    
    
    data['filtered'] = da.fft_filter(data['ECG'], d = 1000, band = [40,200])
    peaks = da.detect_peaks(data['filtered'], data['time'], distance = distance)
    peaks['bpm'] = da.calculate_bpm(peaks['time'], smooth = 5)
    
    avg_bpm = peaks['bpm'].mean()
    HRV = da.calculate_HRV(peaks['time'])
    avg_bpm = round(avg_bpm, 1)
    HRV = round(HRV, 1)

    plt.plot(data['time'], data['ECG']) 
    plt.show()
    
    st.write('Results')
    col1, col2 = st.columns(2)
    col1.metric('average bmp:', f'{avg_bpm} bpm', border = True)
    col2.metric('HRV:', f'{HRV} ms', border = True)
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['time']/1000, y = data['filtered'],
                                 mode = 'lines',
                                 name = 'ECG'))
    fig.add_trace(go.Scatter(x = peaks['time']/1000, y = peaks['value'],
                                 mode = 'markers',
                                 name = 'peaks',
                                 marker_color = 'red'))
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    col1.write('Raw data:')
    col1.write(data)
    col2.write('Raw peak data:')
    col2.write(peaks)
        