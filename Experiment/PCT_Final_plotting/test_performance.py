import pygame
import math
import config
import matplotlib.pyplot as plt
import numpy as np
# from  simulate import simulate
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from gameObjectsV2 import Pendulum, Cart
from constant_timescale_alltrials import get_dist
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import math
from constant_params_single_trial import get_step_estimates_const
from sklearn.metrics import r2_score 

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff=6.7, fs=120, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
   



gain = 5
trial = 1
P_id = 9

# time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
# position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
# theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/angle.npy')
calibration_index =  np.where(time < 15)
calibration_index2 =  np.where(time < 30)
# time = time[0:calibration_index[-1][-1]]
# theta = theta[0:calibration_index[-1][-1]]
# position = position[0:calibration_index[-1][-1]]
time = time[calibration_index[-1][-1]:calibration_index2[-1][-1]]
theta = theta[calibration_index[-1][-1]:calibration_index2[-1][-1]]
position = position[calibration_index[-1][-1]:calibration_index2[-1][-1]]
time_og = time
position_og = position
theta_og = theta

x = np.reshape(time_og, (len(time_og), 1))
model = LinearRegression()

try:
    # pos = position[0:-2]-750
    pos = position-750
    model.fit(x, pos)
except ValueError:
    pos = position[0:-2]-750
    # pos = position-750
    model.fit(x, pos)
# calculate trend
trend = model.predict(x)
detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]
# plt.plot(pos)
# plt.plot(detrended_pos)
# plt.show()
# print(len(theta))
theta_steps = []
steps = []

for i in range(len(detrended_pos)-1):
    steps.append(detrended_pos[i+1]-detrended_pos[i])
    theta_steps.append(theta[i+1]-theta[i])
steps = np.array(steps)
steps_og = steps



steps = steps_og[0::2]

if np.count_nonzero(steps) < len(steps_og)/2:
    steps = steps_og[1::2]

steps = np.repeat(steps, 2)
steps = np.insert(steps,0, steps[0])
steps = np.insert(steps,0, steps[0])
steps_og2 = steps


detrended_pos = np.array(detrended_pos)
theta_steps = np.array(theta_steps)
theta_steps = np.insert(theta_steps, 0,theta[0])
theta_steps_og = theta_steps


# Filter the data
y_og = butter_lowpass_filter(steps_og2)
plt.plot(time[100:-100],y_og[100:-102], label = '$a_t$')
# lengths = [50, 100, 200, 400, 800, 1600, 3000]
lengths = [200, 225, 250, 275, 300, 325, 350, 375, 400]
losses = []
# lengths = [200]
for length in lengths:
    lenz = (len(theta)//length)*length

    opt_kp, opt_kd, filter_noise_samples, fit_noise_samples = get_dist(len(theta)//length)
    opt_kp = np.repeat(opt_kp, length)
    opt_kd = np.repeat(opt_kd, length)

    # opt_kp = np.repeat([-0.05], lenz)
    # opt_kd = np.repeat([0.2], lenz)

    step_est = (opt_kp[100:])*((gain)**-1)*theta[100:lenz] + opt_kd[100:]*((gain)**-1)*theta_steps[100:lenz]

    # y_og = y_og[2:lenz+2]
    loss = ((np.linalg.norm(y_og[100:lenz] - step_est*gain)))/math.sqrt(lenz-100)
    losses.append(loss)
    plt.plot(time[100:lenz],gain*step_est, label = '$\hat{a_t}$' + str(length))

   
plt.legend()
plt.title('Participant 9 - performance of model on test trial T1')
plt.xlabel('Time')
plt.ylabel('Pixels')
textstr = '     '.join(["$\gamma$ = " + str((gain))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
plt.text(0.5, 0.83, textstr, transform=plt.gcf().transFigure, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

plt.show()


plt.plot(lengths, losses)
plt.show()


gain = 5
trial = 1
P_id = 9
performances = []

time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
for size in lengths:
    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1,size= np.array([size]))
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    performances.append(performance[0])

Kps, Kds, dist, final_estimate, steps = get_step_estimates_const(time_og, position_og, gain*theta_og, gain, 1)

const_perf = (np.linalg.norm(np.abs(steps - final_estimate)))/math.sqrt(len(steps))

plt.axhline(const_perf, label = 'const_param', c = 'green', linestyle='--')

plt.plot(np.array(lengths)*0.004, performances, label = 'Training trial performance', c = 'orange')
plt.plot(np.array(lengths)*0.004, losses, label = 'Test trial performance', c = 'red')
plt.ylabel('Loss')
plt.xlabel('Optimization window size (s)')
    
textstr = '     '.join(["$\gamma$ = " + str((gain))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
plt.text(0.5, 0.83, textstr, transform=plt.gcf().transFigure, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

plt.show()


# plt.plot(time, filtered_steps, label = 'filtered_steps')

# plt.title('Filtered step estimates')
# plt.ylabel('Step size')
# plt.xlabel('Time')
# plt.legend()
# plt.show()


