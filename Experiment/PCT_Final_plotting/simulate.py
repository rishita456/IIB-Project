
import numpy as np
import config
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import config
import math
import obspy
from obspy.signal.detrend import polynomial
from scipy.signal import butter,filtfilt
from constant_timescale_alltrials import get_dist
from sklearn.metrics import r2_score 

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff=4, fs=120, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def simulate(step_estimates, start, dt, time, gain, order = 6):


    pos_estimates = np.zeros(len(step_estimates)+1)
    steps2 = []
    pos_estimates[0] = start[0]
    
    for i in range(len(step_estimates)):
        pos_estimates[i+1] = pos_estimates[i]+step_estimates[i]

    cart_vel = step_estimates/dt
    for i in range(len(step_estimates)-1):
        steps2.append(cart_vel[i+1]-cart_vel[i])

    steps2 = np.array(steps2)
    cart_acc = steps2/(dt)
    m = np.mean(pos_estimates)
    pos_estimates = pos_estimates - np.mean(pos_estimates)
    detrended_pos = pos_estimates
    # x = np.reshape(time, (len(time), 1))
    # model = LinearRegression()

    # try:
    #     # pos = position[0:-2]-750
    #     model.fit(x, pos_estimates-750)
    # except ValueError:
    #     pos = pos_estimates[0:-1]-750
    #     # pos = position-750
    #     model.fit(x, pos)
    # # calculate trend
    # trend = model.predict(x)
    # detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    # theta_est = np.convolve(pos_estimates, g_t, 'full')
    theta_est = np.convolve(detrended_pos, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]/gain
    # theta_est = polynomial(theta_est, order=order, plot=False)
    # theta_est = theta_est


   
    return np.array(detrended_pos)+m, theta_est, cart_acc
gain = 1

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_9/test_trials/85sec_trial/angle.npy')
calibration_index =  np.where(time < 15)
calibration_index2 =  np.where(time < 30)
time = time[0:calibration_index[-1][-1]]
theta = theta[0:calibration_index[-1][-1]]
position = position[0:calibration_index[-1][-1]]


x = np.reshape(time, (len(time), 1))
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
plt.plot(pos)
plt.plot(detrended_pos)
plt.show()
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



opt_kp, opt_kd, filter_noise_samples, fit_noise_samples = get_dist(len(theta))

step_est = (opt_kp)*((gain)**-1)*theta + opt_kd*((gain)**-1)*theta_steps

plt.plot(steps_og2)
# pred = butter_lowpass_filter(butter_lowpass_filter(step_est, 6.7)+filter_noise_samples, 6.7)
plt.plot(butter_lowpass_filter(step_est, 6.7))
plt.title(str(r2_score(steps_og2[2:], butter_lowpass_filter(step_est, 6.7))))
plt.show()