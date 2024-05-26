
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from scipy.stats import entropy


gain = 5
trial = 2
P_id = 3


cutoff = 7
fs = 120
order = 5

# cutoffs = np.arange(1,15,1)
kl_divs = []

# for gain in gains:
#     for trial in trials:


time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')


steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
time = time_og[0:len(steps)]
plt.plot(filter_residual_values, filter_noise_kde, label = '$P(r_{filt}|{\gamma}, f_{cutoff})$', c = 'cornflowerblue')
plt.plot(fit_residual_values, fit_noise_kde, label = '$P(r_{fit}|{\gamma}, f_{cutoff})$', c = 'indigo')
plt.ylabel('Probability density')
plt.xlabel('Value')
plt.legend()
plt.show()

# fft 
freq = np.fft.fftfreq(len(steps), time[2] - time[0])  # Frequency bins
fft_result = np.fft.fft(steps)
fft_result2 = np.fft.fft(filtered_steps)


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
# plt.semilogx(freq[:len(fft_result2)//2], 2.0/len(fft_result2) * np.abs(fft_result2[:len(fft_result2)//2]), label = 'filtered')
plt.semilogx(freq[:len(fft_result)//2],np.log10(2.0/len(fft_result2) * np.abs(fft_result[:len(fft_result)//2])), label = 'original')
plt.semilogx(freq[:len(fft_result2)//2],np.log10(2.0/len(fft_result2) * np.abs(fft_result2[:len(fft_result2)//2])), label = 'filtered')
plt.title('Magnitude Response (Bode Plot)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

# Plot the phase response
plt.subplot(2, 1, 2)

plt.semilogx(freq[:len(fft_result)//2], np.angle(fft_result[:len(fft_result)//2]), label = 'original')
plt.semilogx(freq[:len(fft_result2)//2], np.angle(fft_result2[:len(fft_result2)//2]), label = 'filtered')
plt.title('Phase Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.tight_layout()
plt.show()



# Plot FFT
fig, axs1 = plt.subplots(1)
axs1.plot(freq, np.abs(fft_result), label = 'fft of step position data')
axs1.plot(freq, np.abs(fft_result2), label = 'fft of butterworth filtered step position data')
axs1.legend()


plt.title('Cutoff = ' + str(cutoff))
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.grid(True)
plt.show()


fig, axs = plt.subplots(1)
axs.plot(time,steps, label = 'raw step data', alpha = 0.5)
axs.plot(time,filtered_steps, label = 'Filtered step data')
axs.legend()
fig.text(0.5, 0.04, 'time(s)', ha='center')
fig.text(0.04, 0.5, 'step size', va='center', rotation='vertical')
plt.show()

