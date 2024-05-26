"""
Contains tests for workflows involved in streaming data from motive and related to shared memory
"""
# import standard python libraries
import sys
import pytest
import warnings
import pytest
import os

# add Root Directory to system path to import created packages
try:
    sys.path.insert(0,'/Users/rishitabanerjee/Desktop/BrainMachineInterfaces/')
except ModuleNotFoundError:
    try: 
        sys.path.insert(0,'/Users/ashwin/Documents/Y4 project Brain Human Interfaces/General 4th year Github repo/BrainMachineInterfaces')
    except:
        pass

# import created packages
from lib_streamAndRenderDataWorkflows.streamData import *



def testExtractDataFrameFromCSV():
    with pytest.warns(UserWarning):
        warnings.warn("DtypeWarning", UserWarning) # added as currently a warning about datatypes exist when importing csv

        try:
            dataLocation = "Data/charlie_suit_and_wand_demo.csv"
            df = extractDataFrameFromCSV(dataLocation= dataLocation, includeCols='Bone')
        except FileNotFoundError: # execute lines below if the file is being called from the directory it lies in
            try:
                dataLocation = "../Data/charlie_suit_and_wand_demo.csv"
                df = extractDataFrameFromCSV(dataLocation= dataLocation,includeCols='Bone')
                print('')
            except FileNotFoundError:
                raise Exception("Unusual working directory discovered, current directory is: {}".format(os.getcwd()))

        assert df.shape == (1801,359)
        print(df)

def testFetchLiveData():
#    with pytest.raises(Exception,match = "Simulated Dataframe data not provided but the fetch live data simulator is called"):
#        fetchLiveData(s,,sharedMemoryLocation=None,simulate=True)
    pass
    

def testDefineSharedMemory():
    sharedBlock,sharedArray = defineSharedMemory(sharedMemoryName= 'Motive Dump', dataType= "Bone Marker", noDataTypes= 25)
    atexit.register(sharedBlock.close)
    assert sharedArray.shape == (25,3)
    print("Executed")
    

if __name__ == "__main__":
    testExtractDataFrameFromCSV()
    testFetchLiveData()
    testDefineSharedMemory()