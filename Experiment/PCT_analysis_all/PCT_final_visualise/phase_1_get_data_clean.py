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
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples



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

for gain in training_gains:
    for trial in trials:
   

        time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
        position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
        theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1)
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
        time = time_og[0:len(steps)]


        # plt.plot(filtered_steps, label = 'filt')
        # plt.plot(step_est, label = ' gain amplification est')
        # steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
        # plt.plot(step_est, label = 'no gain amplification est ')
        # plt.legend()
        # plt.show()



                        
        trial_end_index = trial_end_index + len(opt_time)
        trial_end_indexes.append(trial_end_index)
        timescales_across_trials.append(opt_time)
        Kps_across_trials.append(opt_kp)
        Kds_across_trials.append(opt_kd)
        

        


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

