import pygame
import math
import config
import matplotlib.pyplot as plt
import numpy as np
from  simulate import simulate
from estimate_steps import get_step_estimates


gain = 5
trial = 2
P_id = 5

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

steps, filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_time, size, perf, const_time = get_step_estimates(time, position, theta, gain,1)

filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))

time = time[0:len(step_est)]
theta = theta[0:len(step_est)]
position = position[0:len(step_est)+2]

plt.plot(time, filtered_steps)
plt.plot(time, step_est)
plt.show()

plt.plot(time, filtered_steps + filter_noise)
plt.plot(time, step_est + filter_noise + fit_noise)
plt.show()
dt = 0.017

start = (position[1], theta[1])
plt.plot(time, position[1:-1], label = 'original')
for i in range(10):
    filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
    fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))
    pos_es, ang_es, cartacc = simulate(step_est + filter_noise + fit_noise, start, dt, time)
    plt.plot(time, pos_es[0:-1], '--', alpha = 0.5)
plt.legend()
plt.show()

plt.plot(time, theta, label = 'original')
for i in range(10):
    filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
    fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))
    pos_es, ang_es, cartacc = simulate(step_est + filter_noise + fit_noise, start, dt, time)
    plt.plot(time, ang_es[0:-1], '--', alpha = 0.5)
plt.legend()
plt.show()


