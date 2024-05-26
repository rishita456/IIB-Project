"""
Functionality to simulate the streaming of rigid body data from motive by feeding each frame 
"""
# import python specific libraries
import os
import sys
import numpy as np
import pandas as pd
import atexit
import time
import datetime

# add base dir to system path
sys.path.insert(0,'/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/')
sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')

# import my libraries
import lib_streamAndRenderDataWorkflows.Client.NatNetClient as NatNetClient
import lib_streamAndRenderDataWorkflows.Client.DataDescriptions as DataDescriptions
import lib_streamAndRenderDataWorkflows.Client.MoCapData as MoCapData
import lib_streamAndRenderDataWorkflows.Client.PythonSample as PythonSample
from lib_streamAndRenderDataWorkflows import streamData, VisualiseLiveData
print("Program started")

# set what type of data to get e.g. Bone, Bone Marker
typeData = "Bone"

# initialise shared memory
shared_Block,sharedArray = streamData.defineSharedMemory(sharedMemoryName= 'Test Rigid Body', dataType= "Bone", noDataTypes= 51,bodyType='skeleton')
print('start fetching data - ', datetime.datetime.now())
# this function fetches data from motive and dumps data in the shared memory
streamData.fetchLiveData(sharedArray, shared_Block, simulate=False)

print('dumped data into shared memory - ', datetime.datetime.now())
    
print("Program ended successfully")
shared_Block.close()