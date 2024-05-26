import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, summation, exp

from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.ndimage import convolve
from scipy.stats import zscore
from scipy.fft import fft


# data = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep_12_run1.npy')
time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep6/run1/time.npy")
position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep6/run1/position.npy")
theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep6/run1/angle.npy")


def split_data(time, theta, n, tseps):

    segments = []

    for i in range(n):

        if i == 0:

    
            post_calibration_index =  np.where(time > tseps[i])
            start_index = post_calibration_index[0][0]
            calibration_index = post_calibration_index[0][0]

            settle_time = 0
            index = 0
    
            calibration_times = time[0:calibration_index]
            calibration_thetas = theta[0:calibration_index]


            for t in calibration_times:
                slice = calibration_thetas[index:-1]
                if np.all(abs((slice)) < 5):
                    settle_time = t
                    print(settle_time)
                    break
                index = index + 1


            settle_index = np.where(time>settle_time)
            print(settle_index)
            print(calibration_index)
            segment0 = (0,post_calibration_index[0][0])
            segments.append(segment0)


        

        else:

            post_segment_i_index = np.where(time > tseps[i])

            end_index = post_segment_i_index[0][0]
            segment_i = (start_index,end_index)

            start_index = post_segment_i_index[0][0]
            segments.append(segment_i)

    segments.append((start_index,len(time)))

    return segments

segments = split_data(time, theta, 2, [30, 36])

print(segments)



    



