import pygame
import math
import config
import matplotlib.pyplot as plt
import numpy as np
from  simulate import simulate
from estimate_steps import get_step_estimates

from sklearn.neighbors import KernelDensity


gain = 5
trial = 1
P_id = 5

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_timescales = get_step_estimates(time, position, theta)

gains = [2, 5, 7, 10, 12, 15, 17, 20]
trials = [1, 2]
filter_mus = []
filter_sigmas = []
fit_mus = []
fit_sigmas = []
kp_means1 = []
kp_means2 = []
kd_means1 = []
kd_means2 = []
opt_t1 = []
opt_t2 = []
opt_kps = np.zeros((20,4000))
opt_kds = np.zeros((20,4000))
opt_times = np.zeros((20,4000))
index = 0
for g in gains:
    for t in trials:
        trial = t
        gain = g

        time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
        position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
        theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

        filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_timescales, size, perf, const_time  = get_step_estimates(time, position, theta, gain)
        filter_mus.append(filter_noise_mu)
        filter_sigmas.append(filter_noise_sigma)
        fit_mus.append(fit_noise_mu)
        fit_sigmas.append(fit_noise_sigma)
        opt_kps[index, :] = opt_kp[0:4000]
        opt_kds[index, :] = opt_kd[0:4000]
        opt_times[index, :] = opt_timescales[0:4000]

        if trial == 1:
            kp_means1.append(np.mean(opt_kp))
            kd_means1.append(np.mean(opt_kd))
            opt_t1.append(np.mean(opt_timescales))

        if trial == 2:
            kp_means2.append(np.mean(opt_kp))
            kd_means2.append(np.mean(opt_kd))
            opt_t2.append(np.mean(opt_timescales))

        index = index + 1



# plt.plot(filter_mus)
# plt.show()


# plt.plot(filter_sigmas)
# plt.show()


# plt.plot(fit_mus)
# plt.show()


# plt.plot(fit_sigmas)
# plt.show()
   
# plt.plot(gains, kp_means1)
# plt.plot(gains, kp_means2)
# plt.show()

# plt.plot(gains, kd_means1)
# plt.plot(gains, kd_means2)
# plt.show()


# plt.plot(gains, opt_t1)
# plt.plot(gains, opt_t2)
# plt.show()

# calculate markov dynamics for phase 1

s = []
s_prime = []
for i in range(20):
    s.append(opt_kps[i, 0:(4000-1)])
    s_prime.append(opt_kps[i, 1:4000])

s = np.array(s).flatten()
s_prime = np.array(s_prime).flatten()

# Reshape data for KDE input (each sample is a pair of (s, s'))
data = np.column_stack((s, s_prime))

# Fit KDE to estimate joint distribution
kde = KernelDensity(bandwidth='scott', kernel='gaussian')
kde.fit(data)

# Define grid of (s, s') values for plotting
# s_values = np.linspace(np.min(opt_kps), np.max(opt_kps), 100)
# s_prime_values = np.linspace(np.min(opt_kps), np.max(opt_kps), 100)
s_values = np.linspace(-1,1, 100)
s_prime_values = np.linspace(-1,1, 100)
S_s, S_prime = np.meshgrid(s_values, s_prime_values)
grid_points = np.column_stack((S_s.ravel(), S_prime.ravel()))

# Evaluate KDE on the grid points to get estimated joint density
log_density = kde.score_samples(grid_points)
joint_density = np.exp(log_density).reshape(S_s.shape)
marginal = np.sum(joint_density, axis = 0)
marginal = marginal/np.sum(marginal)



plt.plot(s_values, marginal)
plt.show()

# Plot the estimated joint distribution
plt.figure(figsize=(8, 6))
plt.contour(S_s, S_prime, joint_density, cmap='viridis')
plt.colorbar(label='Estimated Density')
plt.xlabel('s')
plt.ylabel("s'")
plt.title('Estimated Joint Distribution using KDE')
plt.show()

conditional = joint_density
for i in range(len(marginal)):
    conditional[:, i] = conditional[:, i]/marginal[i]

conditional = conditional/np.sum(conditional)


# Plot the estimated joint distribution
plt.figure(figsize=(8, 6))
plt.contour(S_s, S_prime, conditional, cmap='viridis')
plt.colorbar(label='Estimated Density')
plt.xlabel('s')
plt.ylabel("s'")
plt.title('Estimated Conditional Distribution using KDE')
plt.show()

        

        











