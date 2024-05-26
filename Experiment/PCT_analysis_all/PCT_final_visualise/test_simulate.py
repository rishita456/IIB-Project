import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from simulate import simulate


gain = 10
trial = 2
P_id = 9

time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
# time_og = np.pad(time_og, (100,0), 'edge' )
# theta_og = np.pad(theta_og, (100,0), 'edge')
# position_og = np.pad(position_og, (100,0), 'edge')


steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1)
time = time_og[0:len(steps)]
theta = theta_og[0:len(steps)]
position = position_og[0:len(steps)+2]
order =30

dt = 0.017

start = (position[1], theta[1])
pos_es, ang_es, cartacc = simulate(steps, start, dt, time, order)

plt.plot(time_og, position_og[1:-1], label = 'original')
plt.plot(time, pos_es[0:-1], label = 'simulate')
plt.legend()
plt.show()

plt.plot(time_og, theta_og, label = 'original')
plt.plot(time, ang_es[0:-1], label = 'simulate')
plt.legend()
plt.show()

time = time_og[0:len(step_est)]
theta = theta_og[0:len(step_est)]
position = position_og[0:len(step_est)+2]

filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
plt.plot(filter_residual_values, filter_noise_kde, label = 'filter')
plt.plot(fit_residual_values, fit_noise_kde)
plt.legend()
plt.show()


pos_es, ang_es, cartacc = simulate(step_est + fit_noise_samples + filter_noise_samples, start, dt, time, order)
plt.plot(time_og, position_og[1:-1], label = 'original')
plt.plot(time, pos_es[0:-1], label = 'simulate')
plt.legend()
plt.show()

plt.plot(time_og, theta_og, label = 'original')
plt.plot(time, ang_es[0:-1], label = 'simulate')
plt.legend()
plt.show()


dt = 0.017

start = (position[1], theta[1])
if pct == 0:
    fig, axs = plt.subplots(2)
    axs[0].plot(time_og, position_og[1:-1], label = 'original')
    axs[1].plot(time_og, theta_og, label = 'original')
    for i in range(20):
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))

        pos_es, ang_es, cartacc = simulate(step_est + fit_noise_samples + filter_noise_samples, start, dt, time,order)
        
        axs[0].plot(time, pos_es[0:-1], alpha = 0.5)
        axs[1].plot(time, ang_es[0:-1], alpha = 0.5)

    axs[0].legend()
    axs[1].legend()
    plt.show()