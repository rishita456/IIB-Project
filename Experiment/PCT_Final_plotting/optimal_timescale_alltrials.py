import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from scipy.stats import gaussian_kde, entropy
from scipy.optimize import curve_fit
import math
from scipy.stats import skew, kurtosis, pearsonr, ttest_ind
from estimate_steps import get_step_estimates


def fit_linear_regression(gains, data):
    model = LinearRegression()
    gains = gains.reshape((1,-1)).T
    model.fit(gains, data)
    y_pred = model.predict(gains)
    residuals = gains - y_pred
    std_dev = np.std(residuals)
    return y_pred, std_dev

def separate_data(training_gains, phase, P_id):

    time_data = []
    position_data = []
    theta_data = []
    final_tg = []

    if phase == 1:
        training_gains = [2, 5, 7, 10, 12, 15, 17, 20]
        trials = [1,2]

        for gain in training_gains:
            for trial in trials:

                try:

                    time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
                    position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
                    theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

                    time_data.append(time_og)
                    position_data.append(position_og)
                    theta_data.append(theta_og)
                    final_tg.append(gain)
                except FileNotFoundError:
                    pass
        



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
                final_tg.append(1)

            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)
            

            try:
                time2 = time_og[calibration_index[-1][-1]:]
                theta2 = theta_og[calibration_index[-1][-1]:]
                position2 = position_og[calibration_index[-1][-1]:]
                final_tg.append(gain)
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)
                final_tg.append(None)

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
                final_tg.append(1)
            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)
                final_tg.append(None)
                

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)
            

            try:
                time2 = time_og[calibration_index[-1][-1]:change_index[-1][-1]]
                theta2 = theta_og[calibration_index[-1][-1]:change_index[-1][-1]]
                position2 = position_og[calibration_index[-1][-1]:change_index[-1][-1]]
                final_tg.append(gain)
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time2)
            position_data.append(position2)
            theta_data.append(theta2)
            

            try:
                time3 = time_og[change_index[-1][-1]:]
                theta3 = theta_og[change_index[-1][-1]:]
                position3 = position_og[change_index[-1][-1]:]
                final_tg.append(1.5*gain)

            except IndexError:
                time3 = np.zeros(200)
                theta3 = np.zeros(200)
                position3 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time3)
            position_data.append(position3)
            theta_data.append(theta3)
            

    return time_data, position_data, theta_data, final_tg

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




P_id = 7
training_gains = config.TRIAL_GAINS_7
gain_across_trials = []
mean_kp = []
mean_kd = []
std_kp = []
std_kd = []

# Create a figure and subplots
fig, axs = plt.subplots(3, 1,sharey=True, figsize=(12, 8))

phases = [1, 2, 3]
phase = 1

for phase in phases:
    time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
    

# gain = 7
# trial = 2
# P_id = 3

    Kps_across_trials = []
    Kds_across_trials = []

    trial_end_indexes = []
    trial_end_index = 0
# gain = 2
# trial = 1



    for i in range(len(time_data)):
        time_og = np.array(time_data[i])
        position_og = np.array(position_data[i])
        theta_og = np.array(theta_data[i])
        gain = final_tg[i]

        if len(time_og) < 510:
            final_tg[i] = None

            trial_end_indexes.append(trial_end_index)
            continue

    # time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
    # position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
    # theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
        opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp), max=np.max(opt_kp), min = np.min(opt_kp))
        opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd), max=np.max(opt_kd), min = np.min(opt_kd))
        
        trial_end_index = trial_end_index + len(opt_kp)
        # Kps_across_trials.append(opt_kp)
        # Kds_across_trials.append(opt_kd)

        Kps_across_trials.append(np.ones_like(opt_kp)*entropy(opt_kp_dist))
        Kds_across_trials.append(np.ones_like(opt_kd)*entropy(opt_kd_dist))
       
        trial_end_indexes.append(trial_end_index)

        if gain ==1 or gain == 22.5:
            continue


        # mean_kp.append(np.mean(opt_kp))
        # mean_kd.append(np.mean(opt_kd))
        mean_kp.append(entropy(opt_kp))
        mean_kd.append(entropy(opt_kd))
        std_kp.append(np.std(opt_kp))
        std_kd.append(np.std(opt_kd))
        gain_across_trials.append(gain)

    trial_end_indexes = np.array(trial_end_indexes)
    trial_end_indexes = np.insert(trial_end_indexes,0,0)
    trial_end_indexes = trial_end_indexes

    trial_labels = [str(tg) for tg in final_tg]
    


    trial_labels = np.append(trial_labels, 'end')

    Kps_across_trials = np.hstack(Kps_across_trials)
    Kds_across_trials = np.hstack(Kds_across_trials)


    axs[phase-1].plot(Kps_across_trials, c = 'red')
    for index in trial_end_indexes:
        axs[phase-1].axvline(index, linestyle = '--', c = 'black')

    axs[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
    if phase == 3:
        axs[phase-1].set_xlabel('Consectutive trial $\gamma$s')

    axs[phase-1].set_ylabel('$H(P(K_p))$')

    axs[phase-1].set_title('Phase ' + str(phase))
    # plt.xlabel('Consectutive trial gains')
    # plt.ylabel('Kp')
    # plt.title('Participant ' + str(P_id))
    # plt.show()




# Set common title
fig.suptitle('Participant ' + str(P_id) + ' - Entropy of $P(K_p)$ across trials', fontsize=16)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

fig, axs = plt.subplots(1, 2,sharex=True, figsize=(12, 4))

gain_across_trials = np.array(gain_across_trials)
# axs[0].scatter(gain_across_trials, mean_kp, c = 'red')
# # y_pred, std_dev = fit_linear_regression(gain_across_trials, mean_kp)
# # axs[0].plot(gain_across_trials, y_pred, color='black', label='Linear Regression Fit')  # Regression line
# axs[0].set_xlabel('$\gamma$')
# axs[0].set_ylabel('Mean of $K_p$')

# p_val = ttest_ind(gain_across_trials, mean_kp).pvalue
# corr, _ = pearsonr(gain_across_trials, mean_kp)
# textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
# props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
# axs[0].text(0.15, 0.92, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 

axs[0].scatter(gain_across_trials, std_kd, label = '$K_d$', c = 'purple')
y_pred, std_dev = fit_linear_regression(gain_across_trials, std_kd)
axs[0].plot(gain_across_trials, y_pred, color='black')  # Regression line
axs[0].set_xlabel('$\gamma$')
axs[0].set_ylabel('Variance of $K_d$')

p_val = ttest_ind(gain_across_trials, std_kd).pvalue
corr, _ = pearsonr(gain_across_trials, std_kd)
textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
axs[0].text(0.58, 0.92, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 

axs[1].scatter(gain_across_trials, std_kp, c = 'orange', label = '$K_p$')
y_pred, std_dev = fit_linear_regression(gain_across_trials, std_kp)
axs[1].plot(gain_across_trials, y_pred, color='black', label='Linear Regression Fit')  # Regression line
axs[1].set_xlabel('$\gamma$')
axs[1].set_ylabel('Variance of $K_p$')
p_val = ttest_ind(gain_across_trials, std_kp).pvalue
corr, _ = pearsonr(gain_across_trials, std_kp)
textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
axs[1].text(0.58, 0.92, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 


# Define a common legend outside the subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.show()


# fig, axs = plt.subplots(1, 2,sharex=True, figsize=(12, 4))

# axs[0].scatter(gain_across_trials, mean_kd, c = 'blue')
# y_pred, std_dev = fit_linear_regression(gain_across_trials, mean_kd)
# axs[0].plot(gain_across_trials, y_pred, color='black', label='Linear Regression Fit')  # Regression line
# axs[0].set_xlabel('$\gamma$')
# axs[0].set_ylabel('Mean of $K_d$')

# p_val = ttest_ind(gain_across_trials, mean_kd).pvalue
# corr, _ = pearsonr(gain_across_trials, mean_kd)
# textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
# props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
# axs[0].text(0.15, 0.92, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 



# axs[1].scatter(gain_across_trials, std_kd, label = '$K_d$', c = 'purple')
# y_pred, std_dev = fit_linear_regression(gain_across_trials, std_kd)
# axs[1].plot(gain_across_trials, y_pred, color='black')  # Regression line
# axs[1].set_xlabel('$\gamma$')
# axs[1].set_ylabel('Variance of $K_d$')

# p_val = ttest_ind(gain_across_trials, std_kd).pvalue
# corr, _ = pearsonr(gain_across_trials, std_kd)
# textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
# props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
# plt.text(0.58, 0.92, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 


# # Define a common legend outside the subplots
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')


# plt.show()
