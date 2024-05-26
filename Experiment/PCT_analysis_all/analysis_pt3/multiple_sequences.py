import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
import config
from pykalman import KalmanFilter
from scipy.signal import butter,filtfilt
import plotly.graph_objects as go


# training_gains = [2, 5, 7, 10, 12, 15, 17, 20, 22]
# trial = 1
# P_id = 2

# fracs = []

# for gain in training_gains:
#     time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#     position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#     theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

#     # detrend position data

#     x = np.reshape(time, (len(time), 1))
#     model = LinearRegression()
#     pos = position[0:-2]-750
#     model.fit(x, pos)
#     # calculate trend
#     trend = model.predict(x)
#     detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

#     time = time[1::2]
#     detrended_pos = detrended_pos[1::2]



#     sequence = []
#     steps = []
#     start = detrended_pos[0]

#     for i in range(len(detrended_pos)-1):
#         steps.append(detrended_pos[i+1]-detrended_pos[i])
#     print(steps[1:10])
#     steps = np.array(steps)
#     for i in range(len(steps)):
#         if steps[i]>0:
#             sequence.append(1)

#         else:
#             sequence.append(0)

#     sequence = np.array(sequence)
#     counts, _ = np.histogram(sequence)
#     fracs.append(counts[1]/(counts[0]+counts[1]))

    


# plt.hist(sequence)
# plt.show()


training_gains1 =config.TRIAL_GAINS_4[1][:]
training_gains2 =config.TRIAL_GAINS_4[2][0:8]
fracs = []
mses = []
v1 = []
P_id = 4
final_transition = np.zeros((2,2))

i = 1
for gain in training_gains1:

    slice = training_gains1[0:i]

    trial = slice.count(gain)
    training_gain = gain
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/angle.npy')

    # detrend position data

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position[0:-2]-750
    model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    time = time[1::2]
    detrended_pos = detrended_pos[1::2]
    theta = theta[1::2]

    settle_time = 0
    index = 0
    mse = 0
    calibration_index =  np.where(time < 30)
    time = time[0:calibration_index[-1][-1]]
    theta = theta[0:calibration_index[-1][-1]]
    position = detrended_pos[0:calibration_index[-1][-1]]

    sequence = []
    steps = []
    start = detrended_pos[0]


    for i in range(len(position)-1):
        steps.append(position[i+1]-position[i])
    steps = np.array(steps)
    for i in range(len(steps)):
        if steps[i]>0:
            sequence.append(1)

        else:
            sequence.append(0)
    t_00 = 0
    t_01 = 0
    t_10 = 0
    t_11 = 0
    sequence = np.array(sequence)
    for i in range(len(sequence)-1):
        if sequence[i] == 0:
            if sequence[i+1] == 0:
                t_00 = t_00 + 1
            else:
                t_01 == t_01 + 1
        else:
            if sequence[i+1] == 0:
                t_10 = t_10 + 1
            else:
                t_11 == t_11 + 1

    


    
    


    mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
    mse = np.sqrt(mse)
    counts, _ = np.histogram(sequence)
    transition_matrix = np.array([[t_00, t_01],[t_10, t_11]])

    fracs.append(counts[-1]/(counts[0]+counts[-1]))
    v1.append(sequence[0])
    mses.append(mse)
    index = i + 1
    
    final_transition = final_transition + transition_matrix

i = 1
for gain in training_gains2:
    slice = training_gains2[0:i]
    trial = slice.count(gain)
    training_gain = gain
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')
    if len(time) == 0:
        index = index + 1
        continue
     # detrend position data

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position[0:-2]-750
    model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    time = time[1::2]
    detrended_pos = detrended_pos[1::2]
    theta = theta[1::2]
   

    settle_time = 0
    index = 0
    mse = 0
    calibration_index =  np.where(time < 30)
    time = time[0:calibration_index[-1][-1]]
    theta = theta[0:calibration_index[-1][-1]]
    position = detrended_pos[0:calibration_index[-1][-1]]

    sequence = []
    steps = []
    start = detrended_pos[0]

    for i in range(len(position)-1):
        steps.append(position[i+1]-position[i])
    steps = np.array(steps)
    for i in range(len(steps)):
        if steps[i]>0:
            sequence.append(1)

        else:
            sequence.append(0)

    

    t_00 = 0
    t_01 = 0
    t_10 = 0
    t_11 = 0
    sequence = np.array(sequence)
    for i in range(len(sequence)-1):
        if sequence[i] == 0:
            if sequence[i+1] == 0:
                t_00 = t_00 + 1
            elif sequence[i+1]==1:
                t_01 = t_01 + 1
        elif sequence[i]==1:
            if sequence[i+1] == 0:
                t_10 = t_10 + 1
            elif sequence[i+1]==1:
                t_11 = t_11 + 1

    
   
    
    


    mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
    mse = np.sqrt(mse)
    counts, _ = np.histogram(sequence)
    transition_matrix = np.array([[t_00, t_01],[t_10, t_11]])

   

    fracs.append(counts[-1]/(counts[0]+counts[-1]))
    v1.append(sequence[0])
    mses.append(mse)
    index = i + 1
    
    final_transition = final_transition + transition_matrix

sums = np.sum(final_transition, axis=1)

final_transition = np.array([[final_transition[0,:]/sums[0]], [final_transition[1,:]/sums[1]]])

print(final_transition)
plt.hist(fracs)
plt.show()
plt.hist(v1)
plt.show()

plt.scatter(mses, fracs)
plt.show()


# kalman smoother