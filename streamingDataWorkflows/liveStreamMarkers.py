"""
Functionality to simulate the streaming of labelled marker data from motive by feeding each frame 
"""


# import python specific libraries
import os
import sys
import numpy as np
import pandas as pd
import atexit
import time

sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')


import lib_streamAndRenderDataWorkflows.Client.NatNetClient as NatNetClient
import lib_streamAndRenderDataWorkflows.Client.DataDescriptions as DataDescriptions
import lib_streamAndRenderDataWorkflows.Client.MoCapData as MoCapData
import lib_streamAndRenderDataWorkflows.Client.PythonSample as PythonSample

print("Program started")
# add base dir to system path


# import my libraries
from lib_streamAndRenderDataWorkflows import streamData, VisualiseLiveData

# set what type of data to get e.g. Bone, Bone Marker
typeData = "Bone Marker"

# feed in location of csv data to extract dataframe 
try:        
    dataLocation = "Data/Rishita-jumping jacks 2023-10-18.csv"
    simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation,includeCols='Bone Marker')
except FileNotFoundError: # if file is run from location of file this is needed
    try:
        dataLocation = "Rishita-jumping jacks 2023-10-18.csv"
        simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone Marker')
    except:
        try:
            dataLocation = "../Data/Rishita-jumping jacks 2023-10-18.csv"
            simulatedDF = streamData.extractDataFrameFromCSV(dataLocation = dataLocation, includeCols='Bone Marker')
        except:
            raise Exception('File not found')


# initialise shared memory
shared_Block,sharedArray = streamData.defineSharedMemory(sharedMemoryName= 'Test Rigid Body', dataType= "Bone Marker", noDataTypes= 25)

print("Starting to dump data into shared memory")
#dump latest data into shared memory


streamData.fetchLiveData(sharedArray, shared_Block, simulate=False)
    
print("Program ended successfully")
shared_Block.close()