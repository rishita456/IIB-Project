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


def separate_data(training_gains, phase):

    time_data = []
    position_data = []
    theta_data = []

    if phase == 2:
        training_gains = training_gains[1][:]
        index = 0
        for gain in training_gains:
            index = index + 1
            slice = training_gains[0:index]
            trial = slice.count(gain)
            training_gain = gain
            time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/time.npy')
            position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/position.npy')
            theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/angle.npy')
            
            calibration_index =  np.where(time_og < 30)

            try:
                time1 = time_og[0:calibration_index[-1][-1]]
                theta1 = theta_og[0:calibration_index[-1][-1]]
                position1 = position_og[0:calibration_index[-1][-1]]
            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)

            try:
                time2 = time_og[calibration_index[-1][-1]:]
                theta2 = theta_og[calibration_index[-1][-1]:]
                position2 = position_og[calibration_index[-1][-1]:]
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)

            time_data.append(time2)
            position_data.append(position2)
            theta_data.append(theta2)
            

    if phase == 3:
        training_gains = training_gains[2][0:8]
        index = 0
        for gain in training_gains:
            index = index + 1
            slice = training_gains[0:index]
            trial = slice.count(gain)
            training_gain = gain
            time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
            position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
            theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')
                    
            calibration_index =  np.where(time_og < 30)
            change_index = np.where(time_og < 40)

            
            try:
                time1 = time_og[0:calibration_index[-1][-1]]
                theta1 = theta_og[0:calibration_index[-1][-1]]
                position1 = position_og[0:calibration_index[-1][-1]]
            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)

            try:
                time2 = time_og[calibration_index[-1][-1]:change_index[-1][-1]]
                theta2 = theta_og[calibration_index[-1][-1]:change_index[-1][-1]]
                position2 = position_og[calibration_index[-1][-1]:change_index[-1][-1]]
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)

            time_data.append(time2)
            position_data.append(position2)
            theta_data.append(theta2)

            try:
                time3 = time_og[change_index[-1][-1]:]
                theta3 = theta_og[change_index[-1][-1]:]
                position3 = position_og[change_index[-1][-1]:]

            except IndexError:
                time3 = np.zeros(200)
                theta3 = np.zeros(200)
                position3 = np.zeros(200)

            time_data.append(time3)
            position_data.append(position3)
            theta_data.append(theta3)

    return time_data, position_data, theta_data, training_gains

# Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

fs = 120      # sample rate, Hz
cutoff = 5      # desired cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # high order due to slow roll off

Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []
trials = [1,2]

trial_end_indexes = []
trial_end_index = 0


size = np.arange(200,501, 10)

P_id = 9
training_gains = config.TRIAL_GAINS_9
phase = 2
time_data, position_data, theta_data, training_gains = separate_data(training_gains, phase)


print(len(time_data))
# fig, axs = plt.subplots(1)
# fig2, axs2 = plt.subplots(1)
for i in range(len(time_data)):
    time_og = np.array(time_data[i])
    position_og = np.array(position_data[i])
    theta_og = np.array(theta_data[i])
    if len(time_og) < 510:
        trial_end_indexes.append(trial_end_index)
        continue
    if i%2 == 0:
        gain = 1
    else:
        gain = training_gains[(i//phase)]

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    # plt.plot(size, perf)
    # plt.axhline(const_time)
    # plt.show()
    
    # plt.plot(opt_time)
    # plt.show()
    
    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    trial_end_index = trial_end_index + len(opt_time)

#     trial_end_indexes.append(trial_end_index)

#     if pct == 0:
#         axs.plot(filter_residual_values, filter_noise_kde, linewidth=2, label='gain:'+str(gain) + ' trial:' + str(i) + ' tau:' + str(opt_time[0]), linestyle = '--')
#     else:
#         axs2.plot(fit_residual_values, fit_noise_kde, linewidth=2, label=str(gain), c = 'red', alpha = ((i+1)/(len(time_data)+1)))

        
# axs.legend()
# axs2.legend()

# fig.text(0.5, 0.04, 'value', ha='center')
# fig.text(0.04, 0.5, 'Probability density', va='center', rotation='vertical')
# fig2.text(0.5, 0.04, 'value', ha='center')
# fig2.text(0.04, 0.5, 'Probability density', va='center', rotation='vertical')
# plt.show()






# for gain in training_gains:
#     for trial in trials:
   
#         traing = gain

#         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

#         filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_time = get_step_estimates(time_og, position_og, theta_og, gain)

#         timescales_across_trials.append(opt_time[500:len(time_og)-500])
#         Kps_across_trials.append(opt_kp[500:len(time_og)-500])
#         Kds_across_trials.append(opt_kd[500:len(time_og)-500])
#         trial_end_index = trial_end_index + len(opt_time[500:len(time_og)-500])

#         trial_end_indexes.append(trial_end_index)


trial_end_indexes = np.array(trial_end_indexes)
trial_end_indexes = np.insert(trial_end_indexes,0,0)
trial_end_indexes = trial_end_indexes

trial_labels = np.repeat(training_gains, phase)
for i in range(len(trial_labels)):
    if i%phase == 0:
        trial_labels[i] = str(1)
    elif i%phase == 1:
        trial_labels[i] = str(trial_labels[i])
    elif i%phase == 2:
        trial_labels[i] = str(int(1.5*trial_labels[i]))

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

