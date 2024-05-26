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
from estimate_steps import get_step_estimates
from filters import get_noise_pdf

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def calc_func_find_Kp_0_part0(K):

   
    global theta
    global theta_steps
    global position
    global traing
    global y
    Kp_0 = K[0]
    Kd_0 = K[1]
    pg = traing

    return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))/len(y)


# Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

fs = 120      # sample rate, Hz
cutoff = 5      # desired cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 5       # high order due to slow roll off

Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []
training_gains = [2, 5, 7, 10, 12, 15, 17, 20]
# training_gains = [5,]
trials = [1,2]
P_id = 2
trial_end_indexes = []
trial_end_index = 0

size = np.arange(200,501, 10)
fig, axs = plt.subplots(1)
for gain in training_gains:
    for trial in trials:
   
        traing = gain

        time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
        position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
        theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

        steps, filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_time, size, perf, const_time, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1)
        # filtered_steps_p, step_est_p, filter_noise_mu_p, filter_noise_sigma_p, fit_noise_mu_p, fit_noise_sigma_p, opt_kp_p, opt_kd_p, opt_time_p, size_p, perf_p, const_time_p = get_step_estimates(time_og, position_og, theta_og, gain, 0)
        # plt.plot(size, perf, c = 'red')
        # plt.axhline(const_time, c = 'red')
        # plt.plot(size_p, perf_p, label = 'p', c = 'blue')
        # # plt.axhline(const_time_p, label = 'p', c = 'blue')
        # plt.legend()
        # plt.show()

        residuals = steps - filtered_steps
        x, noise_gaussian, pdf, noise_kde, density = get_noise_pdf(residuals)
        
        # plt.plot(opt_time)
        # plt.show()
        timescales_across_trials.append(opt_time[500:len(time_og)-500])
        Kps_across_trials.append(opt_kp[500:len(time_og)-500])
        Kds_across_trials.append(opt_kd[500:len(time_og)-500])
        trial_end_index = trial_end_index + len(opt_time[500:len(time_og)-500])

        trial_end_indexes.append(trial_end_index)
        if pct == 0:
            axs.plot(x, pdf, linewidth=2, label=str(gain), linestyle = '--')
        else:
            axs.plot(x, pdf, linewidth=2, label=str(gain))

        
axs.legend()

fig.text(0.5, 0.04, 'value', ha='center')
fig.text(0.04, 0.5, 'Probability density', va='center', rotation='vertical')
plt.show()


trial_end_indexes = np.array(trial_end_indexes)
trial_labels = np.repeat(training_gains, 2)
trial_end_indexes = np.insert(trial_end_indexes,0,0)


trial_labels = np.repeat(training_gains, 2)
for i in range(len(trial_labels)):
    trial_labels[i] = str(trial_labels[i])

trial_labels = np.append(trial_labels, 'end')



timescales_across_trials = np.hstack(timescales_across_trials)
Kps_across_trials = np.hstack(Kps_across_trials)
Kds_across_trials = np.hstack(Kds_across_trials)
rate_across_trials = 1/timescales_across_trials

plt.plot(timescales_across_trials)
for index in trial_end_indexes:
    plt.axvline(index, linestyle = '--', c = 'black')

plt.xticks(trial_end_indexes, labels=trial_labels)
plt.xlabel('Consectutive trial gains')
plt.ylabel('Learning timescales')
plt.title('Participant ' + str(P_id))
plt.show()

plt.plot(rate_across_trials)
for index in trial_end_indexes:
    plt.axvline(index, linestyle = '--', c = 'black')

plt.xticks(trial_end_indexes, labels=trial_labels)
plt.xlabel('Consectutive trial gains')
plt.ylabel('Learning rates')
plt.title('Participant ' + str(P_id))
plt.show()


plt.plot(Kps_across_trials)
for index in trial_end_indexes:
    plt.axvline(index, linestyle = '--', c = 'black')

plt.xticks(trial_end_indexes, labels=trial_labels)
plt.xlabel('Consectutive trial gains')
plt.ylabel('Kp')
plt.title('Participant ' + str(P_id))
plt.show()

plt.plot(Kds_across_trials)
for index in trial_end_indexes:
    plt.axvline(index, linestyle = '--', c = 'black')

plt.xticks(trial_end_indexes, labels=trial_labels)
plt.xlabel('Consectutive trial gains')
plt.ylabel('Kd')
plt.title('Participant ' + str(P_id))
plt.show()

