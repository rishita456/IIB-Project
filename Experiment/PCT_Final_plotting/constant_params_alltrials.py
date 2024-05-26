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
from scipy.stats import skew, kurtosis, pearsonr, ttest_ind


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


def get_step_estimates(time, position, theta, gain, d, cutoff = 4, order = 5):

    #boolean value of whether optimal fit follows PCT
    pct = 1

    fs = 120      # sample rate, Hz
   
    traing = gain
    # size = np.arange(200,501, 10) # window sizes over which parameters remain constant
    size = [200]
    time_og = time
    position_og = position
    theta_og = theta

    x = np.reshape(time_og, (len(time_og), 1))
    model = LinearRegression()

    try:
        # pos = position[0:-2]-750
        pos = position-750
        model.fit(x, pos)
    except ValueError:
        pos = position[0:-2]-750
        # pos = position-750
        model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    # print(len(theta))
    theta_steps = []
    steps = []

    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
        theta_steps.append(theta[i+1]-theta[i])
    steps = np.array(steps)
    steps_og = steps
 
    steps = steps_og[0::2]
    if np.count_nonzero(steps) < len(steps_og)/2:
        steps = steps_og[1::2]

    steps = np.repeat(steps, 2)
    steps = np.insert(steps,0, steps[0])
    steps = np.insert(steps,0, steps[0])
    steps_og2 = steps





    detrended_pos = np.array(detrended_pos)
    theta_steps = np.array(theta_steps)
    theta_steps = np.insert(theta_steps, 0,theta[0])
    theta_steps_og = theta_steps
    

    # Filter the data
    y_og = butter_lowpass_filter(steps_og2, cutoff, fs, order)

    y_og = y_og[0:len(y_og)-100]
    theta = theta[0:len(y_og)]
    theta_steps = theta_steps[0:len(y_og)]

    
    result = minimize(calc_func_find_Kp_0_part0, (1,1), args=(theta, theta_steps, traing, y_og, d))
    window_estimate = (result.x[0])*((gain)**-1)*theta + result.x[0]*((gain)**-1)*theta_steps
    dist = (np.abs(y_og-window_estimate))
    Kps = (result.x[0]*np.ones(len(window_estimate)))
    Kds = (result.x[1]*np.ones(len(window_estimate)))

    # # iterate over window sizes
    # for j in range(len(size)):
    #     dist = []
    #     est = []
    #     Kps = []
    #     Kds = []

    #     limits = np.arange(0,len(time_og)+1,size[j])
    #     for i in range(len(limits)-1):
            
    #         limit = (limits[i], limits[i+1])
    #         time = time_og[limit[0]:limit[1]]
    #         position = position_og[limit[0]:limit[1]]
    #         theta = theta_og[limit[0]:limit[1]]
    #         theta_steps = theta_steps_og[limit[0]:limit[1]]
    #         y = y_og[limit[0]:limit[1]]
            

    #         result = minimize(calc_func_find_Kp_0_part0, (1,1), args=(theta, theta_steps, traing, y, d))

    #         # print(result.x)
            

    #         window_estimate = (result.x[0])*((gain)**-1)*theta + result.x[0]*((gain)**-1)*theta_steps
    #         est.append(window_estimate)
    #         dist.append(np.abs(y-window_estimate))
    #         Kps.append(result.x[0]*np.ones(len(window_estimate)))
    #         Kds.append(result.x[1]*np.ones(len(window_estimate)))

    #     est_final = np.hstack(np.array(est))
    #     loss_final = np.hstack(np.array(dist))
    #     Kps_final = np.hstack(np.array(Kps))
    #     Kds_final = np.hstack(np.array(Kds))
    #     # print(len(Kps_final))

    #     losses[j, 0:len(loss_final)] = loss_final
    #     Kp[j, 0:len(loss_final)] = Kps_final
    #     Kd[j, 0:len(loss_final)] = Kds_final
    #     estimates[j, 0:len(loss_final)] = est_final

    #     length = len(loss_final)


    

    # # print(length)
    # # print(np.shape(estimates))
    # # print(np.shape(losses))
    # time = time_og[0:length]
    # position = position_og[0:length+2]
    # theta = theta_og[0:length]
    # y_og = y_og[0:length]
    # steps_og2 = steps_og2[0:length]
    # theta_steps_og = theta_steps_og[0:length]

    # filter_residuals = steps_og2 - y_og


    # min_loss_indices = np.argmin(losses, axis=0)[0:length]
    # min_losses = np.zeros(length)
    # optimum_kp = np.zeros(length)
    # optimum_kd = np.zeros(length)
    # optimum_estimate = np.zeros(length)
    # optimum_timescales = 0.0045*size[np.argmin(losses, axis=0)[0:length]]
 
   
    # performance = []
    # for i in range(len(size)):
    #     e = estimates[i,0:length]
    #     performance.append((np.linalg.norm(np.abs(y_og - e)))/math.sqrt(len(y_og)))
     

        
    # performance = np.array(performance)
    # best = np.argmin(performance)

    
    # for i in range(length):
    #     optimum_kp[i] = Kp[min_loss_indices[i], i]
    #     optimum_kd[i] = Kd[min_loss_indices[i], i]
    #     min_losses[i] = losses[min_loss_indices[i], i]
    #     optimum_estimate[i] = estimates[min_loss_indices[i],i]

    

    # final_estimate = optimum_estimate
    # pct_estimate = optimum_estimate

    # # plt.plot(losses[0,:])
    # # plt.plot(losses[len(size)-1,:], label='last')
    # # plt.plot(min_losses, label = 'opt')
    # # plt.plot(np.abs(y_og[0:len(pct_estimate)] - pct_estimate), label =  'pct')
    # # plt.legend()
    # # plt.show()

    # pct_performance = ((np.linalg.norm(y_og[0:len(pct_estimate)] - pct_estimate)))/math.sqrt(len(pct_estimate))
    # # print(pct_performance)
    
    # if performance[best] < pct_performance:
    #     print('no pct')
    #     pct = 0
    #     optimum_timescales = (0.004*size[best])*np.ones_like(optimum_timescales)
    #     optimum_kp = (Kp[best,0:length])
    #     optimum_kd = (Kd[best,0:length])
    #     final_estimate = estimates[best, 0:length]
    # else:
    #     print('yes pct')


    final_estimate = Kps*((traing)**-1)*theta + Kds*((traing)**-1)*theta_steps

    # fit_residuals = y_og[0:len(final_estimate)] - final_estimate

    # # plt.plot(steps, label = 'og')
    # # plt.plot(pct_estimate, label = 'pct')
    # # plt.plot(final_estimate, label = 'final')
    # # plt.legend()
    # # plt.show()


    # return steps_og2, y_og, final_estimate, filter_residuals, fit_residuals, optimum_kp, optimum_kd, optimum_timescales, size, performance, pct_performance, pct
    return Kps, Kds, dist, final_estimate, y_og



P_id = 9
training_gains = config.TRIAL_GAINS_9
gain_across_trials = []
mean_kp = []
mean_kd = []

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

        Kps, Kds, dist, final_estimate, steps = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
        trial_end_index = trial_end_index + len(Kps)
        Kps_across_trials.append(Kps)
        Kds_across_trials.append(Kds)
        trial_end_indexes.append(trial_end_index)

        if gain ==1 or gain == 22.5:
            continue
        mean_kp.append(Kps[10])
        mean_kd.append(Kds[10])
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

    axs[phase-1].set_ylabel('$K_p$')

    axs[phase-1].set_title('Phase ' + str(phase))
    # plt.xlabel('Consectutive trial gains')
    # plt.ylabel('Kp')
    # plt.title('Participant ' + str(P_id))
    # plt.show()




# Set common title
fig.suptitle('Participant ' + str(P_id) + ' - Evolution of $K_p$ across trials', fontsize=16)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
gain_across_trials = np.array(gain_across_trials)
plt.scatter(gain_across_trials, mean_kp, c = 'red', label = '$K_p$')
y_pred, std_dev = fit_linear_regression(gain_across_trials, mean_kp)
plt.plot(gain_across_trials, y_pred, color='black', label='Linear Regression Fit')  # Regression line
plt.xlabel('$\gamma$')
plt.ylabel('$K_p$')
plt.legend()
p_val = ttest_ind(gain_across_trials, mean_kp).pvalue
corr, _ = pearsonr(gain_across_trials, mean_kp)
textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
plt.text(0.3, 0.91, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 
plt.show()


plt.scatter(gain_across_trials, mean_kd, label = '$K_d$', c = 'blue')
y_pred, std_dev = fit_linear_regression(gain_across_trials, mean_kd)
plt.plot(gain_across_trials, y_pred, color='black', label='Linear Regression Fit')  # Regression line
plt.xlabel('$\gamma$')
plt.ylabel('$K_d$')
plt.legend()
p_val = ttest_ind(gain_across_trials, mean_kd).pvalue
corr, _ = pearsonr(gain_across_trials, mean_kd)
textstr = '     '.join(["$p = $" + str((p_val)), "$r= $" + str(round(corr, 3))])
props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
plt.text(0.3, 0.91, textstr, transform=plt.gcf().transFigure, fontsize=10, bbox = props) 
plt.show()

time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_3/test_trials/85sec_trial/time.npy')
position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_3/test_trials/85sec_trial/position.npy')
theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_3/test_trials/85sec_trial/angle.npy')

