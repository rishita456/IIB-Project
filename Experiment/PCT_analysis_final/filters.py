import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import curve_fit
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from lightning.classification import FistaClassifier
from sklearn.model_selection import GridSearchCV

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def kalman(noisy_data, sigma):
    # Define Kalman filter parameters
    initial_state_mean = noisy_data[0]  # Initial guess for the state
    initial_state_covariance = 1  # Initial guess for the error covariance
    transition_matrix = [1]  # Transition matrix for a simple 1D model
    observation_matrix = [1]  # Observation matrix for a simple 1D model
    observation_covariance = 1  # Variance of the measurement noise
    transition_covariance = sigma  # Variance of the process (noise in the system)

    # Create Kalman filter
    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    # Apply Kalman filter smoothing
    (filtered_state_means, _) = kf.smooth(noisy_data)
    print(np.shape(filtered_state_means))

    return filtered_state_means



gain = 5
trial = 1
P_id = 4

time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')


steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
time = time_og[0:len(steps)]

# plt.plot(time_og, position_og[1:-1])
# plt.show()

# plt.plot(time, steps)
# plt.show()

# Butterworth Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

fs = 120      # sample rate, Hz
cutoff = 2     # desired cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 10      # high order due to slow roll off

# Filter the data
y = butter_lowpass_filter(steps, cutoff, fs, order)
y_k = kalman(steps, np.std(steps))
print(np.std(steps))

fig, axs = plt.subplots(2)
axs[0].plot(time,steps, label = 'raw step data', alpha = 0.5)
axs[0].plot(time,y, label = 'Butterworth filter')
axs[1].plot(time,steps, label = 'raw step data', alpha = 0.5)
axs[1].plot(time,y_k, label = 'Kalman filter')

axs[0].legend()
axs[1].legend()
fig.text(0.5, 0.04, 'time(s)', ha='center')
fig.text(0.04, 0.5, 'step size', va='center', rotation='vertical')
plt.show()

# fft 
freq = np.fft.fftfreq(len(steps), time[2] - time[0])  # Frequency bins
fft_result = np.fft.fft(steps)
fft_result2 = np.fft.fft(y)
fft_result3 = np.fft.fft(y_k)

# Plot FFT
# plt.figure(figsize=(10, 6))
fig, axs = plt.subplots(2)
axs[0].plot(freq, np.abs(fft_result), label = 'fft of step position data')
axs[0].plot(freq, np.abs(fft_result2), label = 'fft of butterworth filtered step position data')
axs[1].plot(freq, np.abs(fft_result), label = 'fft of step position data')
axs[1].plot(freq, np.abs(fft_result3), label = 'fft of kalman filtered step position data')
axs[0].legend()
axs[1].legend()

plt.title('FFT of Time Series')
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.grid(True)

plt.show()

residuals = steps[500:4500] - y[500:4500]
residuals_k = steps[500:4500] - y_k[500:4500,0]

# Fit a normal distribution to the residuals - note not always gaussian
(mu, sigma) = norm.fit(residuals)
# Plot histogram of the data
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')
# Plot the probability density function (PDF) of the fitted normal distribution
x = np.linspace(min(residuals), max(residuals), 1000)  # Range of x values
pdf = norm.pdf(x, mu, sigma)  # PDF of the normal distribution
kde_butter = gaussian_kde(residuals)
plt.plot(x, pdf, color='red', linewidth=2, label='Normal Distribution')
density_butter = kde_butter.evaluate(x)
plt.plot(x, density_butter, color='green', linewidth=2, label='KDE')
plt.title('Histogram and Fitted Normal Distribution and KDE estimates for butterworth filtered data')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()




# Fit a normal distribution to the residuals - note not always gaussian
(mu_k, sigma_k) = norm.fit(residuals_k)

# Plot histogram of the data
plt.figure(figsize=(8, 6))
plt.hist(residuals_k, bins=30, density=True, alpha=0.5, label='Histogram')

# Plot the probability density function (PDF) of the fitted normal distribution
x = np.linspace(min(residuals_k), max(residuals_k), 1000)  # Range of x values
pdf = norm.pdf(x, mu_k, sigma_k)  # PDF of the normal distribution
kde_k = gaussian_kde(residuals_k)
density_k = kde_k.evaluate(x)
plt.plot(x, density_k, color='green', linewidth=2, label='KDE')
plt.plot(x, pdf, color='red', linewidth=2, label='Normal Distribution')
plt.title('Histogram and Fitted Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

fig, axs = plt.subplots(2)
noise_gaussian = np.random.normal(mu, sigma, 4000)
noise_kde = kde_butter.resample(4000)
axs[0].plot(time[500:4500],noise_gaussian + y[500:4500], label = 'reconstructed with gaussian noise')
axs[0].plot(time[500:4500],steps[500:4500], label = 'raw step data', alpha = 0.5)
axs[1].plot(time[500:4500],noise_kde[0,:] + y[500:4500], label = 'reconstructed with noise sampled from kde')
axs[1].plot(time[500:4500],steps[500:4500], label = 'raw step data', alpha = 0.5)
fig.text(0.5, 0.04, 'time(s)', ha='center')
fig.text(0.04, 0.5, 'step size', va='center', rotation='vertical')
plt.title('Butterworth Filtered Data')
plt.legend()
plt.show()

fig, axs = plt.subplots(2)
noise_gaussian = np.random.normal(mu, sigma, 4000)
noise_kde = kde_k.resample(4000)

axs[0].plot(time[500:4500],noise_gaussian + y_k[500:4500,0], label = 'reconstructed with gaussian noise')
axs[0].plot(time[500:4500],steps[500:4500], label = 'raw step data', alpha = 0.5)
axs[1].plot(time[500:4500],noise_kde[0,:] + y_k[500:4500,0], label = 'reconstructed with noise sampled from kde')
axs[1].plot(time[500:4500],steps[500:4500], label = 'raw step data', alpha = 0.5)
fig.text(0.5, 0.04, 'time(s)', ha='center')
fig.text(0.04, 0.5, 'step size', va='center', rotation='vertical')
plt.title('Kalman Filtered Data')
plt.legend()
plt.show()





