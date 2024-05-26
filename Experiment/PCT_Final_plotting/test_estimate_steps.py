
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from sklearn.metrics import r2_score

gain = 5
trial = 1
P_id = 4

trials = [1,2]
gains = [2, 5, 7, 10]

time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, 1.05, size=np.array([200]))
filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
time = time_og[0:len(steps)]

fig, axs = plt.subplots(1)
axs.plot(time,steps, label = '$u_t$', alpha = 0.5)
axs.plot(time,filtered_steps, label = '$u\'_t$')
axs.legend()
fig.text(0.5, 0.04, 'time(s)', ha='center')
fig.text(0.04, 0.5, 'Centered screen position', va='center', rotation='vertical')
plt.title('Step 1: Filter position data')
plt.show()

# fft 
freq = np.fft.fftfreq(len(steps), time[2] - time[0])  # Frequency bins
fft_result = np.fft.fft(steps)
fft_result2 = np.fft.fft(filtered_steps)


# Plot FFT
# plt.figure(figsize=(10, 6))
fig, axs = plt.subplots(1)
axs.plot(freq, np.abs(fft_result), label = 'fft of $u_t$')
axs.plot(freq, np.abs(fft_result2), label = 'fft of $u\'_t$')

axs.legend()


plt.title('FFT')
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.grid(True)
plt.show()



# plt.plot(time, steps)
# plt.plot(time, filtered_steps)
# plt.show()
plt.plot(time, filtered_steps, label = '$a_t$')
plt.plot(time, step_est, label = '$\hat{a_t}$')
plt.title('Participant 9 - performance of model on training data')
textstr = '     '.join(["$\gamma$ = " + str((gain))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
plt.text(0.5, 0.83, textstr, transform=plt.gcf().transFigure, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
plt.ylabel('Pixels')
plt.xlabel('Time')
plt.legend()
plt.show()

plt.plot(filter_residual_values, filter_noise_kde, label = '$P(r_{filt}|{\gamma}=5, f_{cutoff} = 1)$', c = 'violet')
plt.plot(fit_residual_values, fit_noise_kde, label = '$P(r_{fit}|{\gamma}=5, f_{cutoff} = 1)$', c = 'blue')
plt.ylabel('Probability density')
plt.xlabel('Value')
plt.legend()
plt.show()


plt.plot(time, steps, label = 'original step data' )
plt.plot(time, step_est + filter_noise_samples, alpha = 0.5, label = 'reconstructed step data')
plt.legend()
plt.show()

plt.plot(0.004*size, performance, label = 'loss function over constant timescales', c = 'blue')
plt.axhline(pct_performance, label = 'loss function with optimum timescales', c = 'red')
plt.legend()
plt.show()


cutoff = 0.2
fs = 120
order = 5

fig, axs = plt.subplots(1)
for gain in gains:
    for trial in trials:

        time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
        position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
        theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
        

        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, 7)
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
        time = time_og[0:len(steps)]

        axs.plot(fit_residual_values, fit_noise_kde, label = 'gain' + str(gain) + ' trial = ' + str(trial))
        axs.set_title('Filter residuals pdf')
        axs.legend()
        


plt.show()


