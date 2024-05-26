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

fig, axs = plt.subplots(2)
fig.suptitle('Estimates')
axs[0].plot(time, position[1:-1], label = 'original position')
axs[1].plot(time, theta, label = 'original angle')
for i in range(5):

    filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_timescales, size, perf, const_time  = get_step_estimates(time, position, theta, gain)

    # plt.plot(opt_kp)
    # plt.show()

    # plt.plot(opt_kd)
    # plt.show()

    # plt.plot(opt_timescales)
    # plt.show()


    start = (position[1], theta[1])
    dt = 0.017
    

    filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
    fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))
    pos_es, ang_es, cartacc = simulate(step_est + filter_noise + fit_noise, start, dt, time)

    pos_es = np.insert(pos_es, 0, pos_es[0])
    pos_es = np.insert(pos_es, len(pos_es), pos_es[len(pos_es)-1])

    theta = ang_es
    position = pos_es

    time = time[0:len(step_est)+1]

    axs[0].plot(time, position[1:-1], '--', alpha = 0.5)
    axs[1].plot(time, ang_es, '--', alpha = 0.5)


axs[0].legend()
axs[1].legend()
plt.show()


gain = 5
trial = 2
P_id = 5

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_timescales, size, perf, const_time   = get_step_estimates(time, position, theta, gain)

filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))
dt = 0.017

fig, axs = plt.subplots(2)

start = (position[1], theta[1])
axs[0].plot(time, position[1:-1], label = 'original position')
axs[1].plot(time, theta, label = 'original angle')
time = time[0:len(step_est)+1]
for i in range(10):
    filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
    fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))
    pos_es, ang_es, cartacc = simulate(step_est + filter_noise + fit_noise, start, dt, time)
    axs[0].plot(time, pos_es, '--', alpha = 0.5)
    axs[1].plot(time, ang_es, '--', alpha = 0.5)
axs[0].legend()
axs[1].legend()
plt.show()







    # plt.plot(time, pos_es, '--', alpha = 0.5)
    # plt.legend()
    # plt.show()



