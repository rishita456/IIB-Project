
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples

gain = 2
trial = 2
P_id = 5

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time, position, theta, gain, 1)
filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
time = time[0:len(steps)]
plt.plot(time, steps)
plt.plot(time, filtered_steps)
plt.show()
plt.plot(time, filtered_steps)
plt.plot(time, step_est)
plt.show()

plt.plot(filter_residual_values, filter_noise_kde, label = 'filter noise pdf')
plt.plot(fit_residual_values, fit_noise_kde, label = 'fit noise pdf')
plt.legend()
plt.show()


plt.plot(time, filtered_steps + filter_noise_samples)
plt.plot(time, step_est + filter_noise_samples + fit_noise_samples, alpha = 0.5)
plt.legend()
plt.show()

plt.plot(size, performance)
plt.axhline(pct_performance)
plt.show()
