
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples


gain = 2
trial = 2
P_id = 5

trials = [1,2]
gains = [2, 5, 7, 10]


cutoff = 0.2
fs = 120
order = 5
cutoffs = np.arange(1,20,2)

fig, axs = plt.subplots(1)
for gain in gains:
    for trial in trials:

        time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
        position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
        theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
        

        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff, order)
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
        time = time_og[0:len(steps)]

        axs.plot(filter_residual_values, filter_noise_kde, label = 'gain' + str(gain) + ' trial = ' + str(trial))
        axs.set_title('Filter residuals pdf')
        axs.legend()

plt.show()
# y = butter_lowpass_filter(steps, cutoff, fs, order)
# y_k = kalman(steps, np.std(steps))

# fig, axs = plt.subplots(1)
# axs.plot(time,steps, label = 'raw step data', alpha = 0.5)
# axs.plot(time,y, label = 'Butterworth filter')
# axs.legend()
# fig.text(0.5, 0.04, 'time(s)', ha='center')
# fig.text(0.04, 0.5, 'step size', va='center', rotation='vertical')
# plt.show()

# # fft 
# freq = np.fft.fftfreq(len(steps), time[2] - time[0])  # Frequency bins
# fft_result = np.fft.fft(steps)
# fft_result2 = np.fft.fft(y)
# fft_result3 = np.fft.fft(y_k)

# # Plot FFT
# # plt.figure(figsize=(10, 6))
# fig, axs = plt.subplots(2)
# axs[0].plot(freq, np.abs(fft_result), label = 'fft of step position data')
# axs[0].plot(freq, np.abs(fft_result2), label = 'fft of butterworth filtered step position data')
# axs[1].plot(freq, np.abs(fft_result), label = 'fft of step position data')
# axs[1].plot(freq, np.abs(fft_result3), label = 'fft of kalman filtered step position data')
# axs[0].legend()
# axs[1].legend()

# plt.title('FFT of Time Series')
# fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
# fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
# plt.grid(True)
# plt.show()



# plt.plot(time, steps)
# plt.plot(time, filtered_steps)
# plt.show()
# plt.plot(time, filtered_steps)
# plt.plot(time, step_est)
# plt.show()

# plt.plot(filter_residual_values, filter_noise_kde, label = 'filter noise pdf')
# plt.plot(fit_residual_values, fit_noise_kde, label = 'fit noise pdf')
# plt.legend()
# plt.show()


# plt.plot(time, filtered_steps + filter_noise_samples)
# plt.plot(time, step_est + filter_noise_samples + fit_noise_samples, alpha = 0.5)
# plt.legend()
# plt.show()

# plt.plot(size, performance)
# plt.axhline(pct_performance)
# plt.show()
