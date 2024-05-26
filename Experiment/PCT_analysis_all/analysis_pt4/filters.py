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


# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def return_filtered_data(time, position, theta):

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = pos = position[0:-2]-750
    model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    theta_steps = []
    steps = []
    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
        theta_steps.append(theta[i+1]-theta[i])
    steps = np.array(steps)

    detrended_pos = np.array(detrended_pos)
    theta_steps = np.array(theta_steps)
    theta_steps = np.insert(theta_steps, 0,theta[0])

    # Butterworth Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

    fs = 120      # sample rate, Hz
    cutoff = 2      # desired cutoff frequency of the filter, Hz 
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 5       # high order due to slow roll off

    # Filter the data
    y_butter = butter_lowpass_filter(steps, cutoff, fs, order)
    

    return steps, y_butter

def get_noise_pdf(residuals):

    
    (mu, sigma) = norm.fit(residuals)
    
    x = np.linspace(-2, 2, 1000)  # Range of x values
    pdf = norm.pdf(x, mu, sigma)  # PDF of the normal distribution
    kde = gaussian_kde(residuals)
    density = kde.evaluate(x)

    noise_gaussian = np.random.normal(mu, sigma, 4000)
    noise_kde = kde.resample(4000)

    return x, noise_gaussian, pdf, noise_kde, density

# training_gains = [2, 5, 7, 10, 12, 15, 17, 20]
# # training_gains = [5,]
# trials = [1,2]
# P_id = 2

# # Butterworth Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

# fs = 120      # sample rate, Hz
# cutoff = 2      # desired cutoff frequency of the filter, Hz 
# nyq = 0.5 * fs  # Nyquist Frequency
# order = 5       # high order due to slow roll off


# fig, axs = plt.subplots(2)
# trial = 1
# for gain in training_gains:

   
#         traing = gain

#         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

#         steps, y_butter, y_kalman = return_filtered_data(time_og, position_og, theta_og)

#         residuals_butter = steps - y_butter
#         residuals_kalman = steps - y_kalman

#         x, noise_gaussian, pdf, noise_kde, density = get_noise_pdf(residuals_butter)

#         axs[0].plot(x, pdf, linewidth=2, label=str(gain))
#         axs[1].plot(x, density, linewidth=2, label=str(gain))

# axs[0].legend()
# axs[1].legend()
# fig.text(0.5, 0.04, 'value', ha='center')
# fig.text(0.04, 0.5, 'Probability density', va='center', rotation='vertical')
# plt.show()

