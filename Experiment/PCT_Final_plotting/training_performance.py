import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import math
from estimate_steps import get_noise_pdf_and_samples, get_step_estimates
from constant_params_single_trial import get_step_estimates_const

from matplotlib.lines import Line2D



# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_noise_pdf_and_samples(residuals, n, max = 1, min = -1):
    
    x = np.linspace(min, max, n)  # Range of x values for noise
    # x = np.linspace(min(residuals), max(residuals), n)  # Range of x values general
    kde = gaussian_kde(residuals)
    density = kde.evaluate(x)

    noise_samples = kde.resample(n)

    return x, noise_samples[0,:], density

def calc_func_find_Kp_0_part0(K, theta, theta_steps, traing, y, d):


    Kp_0 = K[0]
    Kd_0 = d*K[1]
    pg = traing

    # return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))#/math.sqrt(len(y))
    return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta + Kd_0*((pg)**-1)*theta_steps))#/math.sqrt(len(y))


gain = 7
trial = 2
P_id = 3

gains = [2, 5, 7, 10, 12, 15]

trial = 1
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
i = 0
for gain in gains:
    time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
    position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
    theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')


    step_sizes = [200, 225, 250, 275, 300, 325, 350, 375, 400]
    performances = []

    for size in step_sizes:
        if gain == 15:
            steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 3, size= np.array([size]))
        else:
            steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1,size= np.array([size]))
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
        time = time_og[0:len(steps)]
        performances.append(performance[0])
    #     if size == 200 or size ==250:
    #         plt.plot(time, abs(fit_residuals), label = str(0.004*size))

    # plt.legend()
    # plt.show()

    Kps, Kds, dist, final_estimate, steps = get_step_estimates_const(time_og, position_og, gain*theta_og, gain, 1)

    const_perf = (np.linalg.norm(np.abs(steps - final_estimate)))/math.sqrt(len(steps))

    axs[i//3,i%3].axhline(const_perf, label = 'const_param', c = 'green', linestyle='--')

    if gain == 15:
        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 3, size= np.array([size]))
    else:
        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1,size= np.array([size]))
    # steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    # axs[i//3,i%3].axhline(pct_performance, label = 'opt', c = 'green' , linestyle='--')
    axs[i//3,i%3].plot(np.array(step_sizes)*0.004, performances, label = 'const rate', c = 'orange')
    if i%3 == 0:
        axs[i//3,i%3].set_ylabel('Loss')
    if i//3 == 1:
        axs[i//3,i%3].set_xlabel('Optimization window size (s)')
    
    textstr = '     '.join(["$\gamma$ = " + str((gain))])
    props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
    axs[i//3,i%3].text(0.8, 0.8, textstr, transform=axs[i//3,i%3].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)


    i = i+1

custom_lines = [
                Line2D([0], [0], color='green', lw=1, linestyle='--'),
                Line2D([0], [0], color='orange', lw=1),
                ]

fig.legend(custom_lines, ['Method1', 'Method2'], loc = 'upper right')
fig.suptitle('Participant 9 - Perfomance of fitting methods on training data (Phase 1)')
plt.show()


# plt.plot(time, filtered_steps, label = 'filtered_steps')

# plt.title('Filtered step estimates')
# plt.ylabel('Step size')
# plt.xlabel('Time')
# plt.legend()
# plt.show()




# # plt.plot(time_og, np.abs(final_estimate - filtered_steps) , c = 'red', label = '$\hat{a_t}$, constant params')
# # plt.plot(time_og, steps, alpha = 0.4, c = 'blue', label = '$a_t$')
# plt.plot(time, step_est, c = 'green', label = '$\hat{a_t}$, params varying every 1s')
if gain == 15:
    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 3, size= np.array([size]))
else:
    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1,size= np.array([size]))
# steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1 ,size= np.array([750]))
filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
time = time_og[0:len(steps)]

# plt.plot(time, step_est, c = 'violet', label = '$\hat{a_t}$, params varying every 2s')

# plt.legend()
# plt.xlabel('Time')
# plt.ylabel("Pendulum angle (degrees)")
# plt.title('Participant 3, phase A trial 7')
# plt.show()

 
