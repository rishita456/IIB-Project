import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import config
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from get_data import get_marginal_pdf, get_conditional_pdf
from scipy.stats import skew, kurtosis, pearsonr
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
def fit_linear_regression(gains, data):
    model = LinearRegression()
    gains = gains.reshape((1,-1)).T
    model.fit(gains, data)
    y_pred = model.predict(gains)
    residuals = gains - y_pred
    std_dev = np.std(residuals)
    return y_pred, std_dev

def fit_polynomial_regression(gains, data, degree):

    poly_features = PolynomialFeatures(degree=degree)
    model = LinearRegression()
    gains = gains.reshape((1,-1)).T
    gains_poly = poly_features.fit_transform(gains)
    model.fit(gains_poly,data)
    y_pred = model.predict(gains_poly)
    residuals = gains - y_pred
    std_dev = np.std(residuals)

    return y_pred, std_dev

# def reject_outliers(data, m=2):
#     return data[abs(data - np.mean(data)) < m * np.std(data)]


def autocorrelation(signal):
    """
    Calculate the autocorrelation coefficients of a 1D signal.

    Parameters:
    signal (ndarray): 1D array representing the signal.

    Returns:
    ndarray: Autocorrelation coefficients.
    """
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')  # Compute cross-correlation
    autocorr = autocorr / (np.max(autocorr))  # Normalize
    return autocorr[n-1:]  # Return non-negative lags only



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
        training_gains = training_gains[1][0:8]
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


meanskp_across_trials = []
stdkp_across_trials = []
meanskd_across_trials = []
stdkd_across_trials = []
meanstau_across_trials = []
stdtau_across_trials = []
gain_across_trials = []
skew_tau_across_trials = []
sparsities = []
noise_vars = []
acfs = []
meankp_ones = []
stdkp_ones = []
meankd_ones = []
stdkd_ones = []
meantau_ones = []
stdtau_ones = []



P_id = 9
training_gains = config.TRIAL_GAINS_9
sigma = 200

# Create a figure and subplots
fig, axs = plt.subplots(3, 1,sharey=True, figsize=(12, 8))
fig2, axs2 = plt.subplots(3, 1,sharey=True, figsize=(12, 8))
# fig3, axs3 = plt.subplots(3, 1,sharey=True, figsize=(12, 8))

theta_stacked = np.array([])
Kps_stacked = np.array([])
##---------------------------------------------------------------------PHASE 1-----------------------------------------------------------------------------------------##


phase = 1
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
gain_across_trials.append(final_tg)

Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []

trial_end_indexes = []
trial_end_index = 0


# print(len(time_data))



for i in range(len(time_data)):
    time_og = np.array(time_data[i])
    position_og = np.array(position_data[i])
    theta_og = np.array(theta_data[i])
    gain = final_tg[i]
    if len(time_og) < 510:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1,7)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    
    if pct==0:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue
    
    # obs_val, action_val,  conditional = get_conditional_pdf(theta_og[0:len(steps)], steps)
    # sparsity = len(np.where(conditional<0.01)[0])/40000

    acf = autocorrelation(filtered_steps)
    acfs.append(acf[400])
    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))
    opt_tau_values, opt_tau_samples, opt_tau_dist = get_noise_pdf_and_samples(opt_time, len(opt_time), np.max(opt_time), np.min(opt_time))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    variance_tau = np.trapz(opt_tau_values**2 * opt_tau_dist, opt_tau_values) - (np.trapz(opt_tau_values * opt_tau_dist, opt_tau_values))**2
    std_dev_tau = variance_tau

    variance_noise = np.trapz(filter_residual_values**2 * filter_noise_kde, filter_residual_values) - (np.trapz(filter_residual_values * filter_noise_kde, filter_residual_values))**2
    std_dev_noise = variance_kd

    # mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    # mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]
    # mean_tau = opt_tau_values[np.argmax(opt_tau_dist)]

    mean_kp = np.sum(opt_kp_values * (opt_kp_dist/np.linalg.norm(opt_kp_dist)))
    mean_kd = np.sum(opt_kd_values * (opt_kd_dist/np.linalg.norm(opt_kd_dist)))
    mean_tau = np.sum(opt_tau_values * (opt_tau_dist/np.linalg.norm(opt_tau_dist)))
    # plt.plot(opt_tau_values, opt_tau_dist)
    # plt.show()

    # plt.scatter([1], np.sum(opt_tau_values*(opt_tau_dist/np.linalg.norm(opt_tau_dist))))
    # plt.show()

    trial_end_index = trial_end_index + len(opt_time)

    theta_stacked = np.hstack((theta_stacked,theta_og[0:len(opt_kp)]))
    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)
    meanstau_across_trials.append(mean_tau)
    stdtau_across_trials.append(std_dev_tau)
    skew_tau_across_trials.append(skew(opt_time))
    # sparsities.append(sparsity)
    noise_vars.append(std_dev_noise)

    if gain == 1:
        meankp_ones.append(mean_kp)
        meankd_ones.append(mean_kd)
        stdkp_ones.append(std_dev_kp)
        stdkd_ones.append(std_dev_kd)
        meantau_ones.append(mean_tau)
        stdtau_ones.append(std_dev_tau)



    trial_end_indexes.append(trial_end_index)




trial_end_indexes = np.array(trial_end_indexes)
trial_end_indexes = np.insert(trial_end_indexes,0,0)
trial_end_indexes = trial_end_indexes

trial_labels = [str(tg) for tg in final_tg]


trial_labels = np.append(trial_labels, 'end')

timescales_across_trials = np.hstack(timescales_across_trials)
Kps_across_trials = np.hstack(Kps_across_trials)
Kds_across_trials = np.hstack(Kds_across_trials)
rate_across_trials = 1/timescales_across_trials
Kps_stacked = np.hstack((Kps_stacked, Kps_across_trials))




axs[phase-1].plot(Kps_across_trials, c = 'red', alpha = 0.3)
axs[phase-1].plot(gaussian_filter1d(Kps_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs[phase-1].set_ylabel('$K_p$')
axs[phase-1].set_title('Phase ' + str(phase))


axs2[phase-1].plot(Kds_across_trials, c = 'blue', alpha = 0.3)
axs2[phase-1].plot(gaussian_filter1d(Kds_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs2[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs2[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs2[phase-1].set_ylabel('$K_d$')
axs2[phase-1].set_title('Phase ' + str(phase))

# axs3[phase-1].plot(timescales_across_trials, c = 'green')
# for index in trial_end_indexes:
#     axs3[phase - 1].axvline(index, linestyle = '--', c = 'black')
# axs3[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
# axs3[phase-1].set_ylabel('Optimization window length')
# axs3[phase-1].set_title('Phase ' + str(phase))


# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kp')
# plt.title('Participant ' + str(P_id))
# plt.show()

# plt.plot(Kds_across_trials)
# for index in trial_end_indexes:
#     plt.axvline(index, linestyle = '--', c = 'black')

# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kd')
# plt.title('Participant ' + str(P_id))
# plt.show()

# plt.plot(meanstau_across_trials)
# plt.show()

# plt.plot(stdtau_across_trials)
# plt.show()

# plt.plot(skew_tau_across_trials)
# plt.show()

# plt.plot(kurt_tau_across_trials)
# plt.show()

# plt.scatter(skew_tau_across_trials, sparsities)
# plt.show()

# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kps', Kps_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kds', Kds_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Taus',timescales_across_trials )
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Endindices', trial_end_indexes)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Gains', final_tg)



##---------------------------------------------------------------------PHASE 2-----------------------------------------------------------------------------------------##


phase = 2
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
gain_across_trials.append(final_tg)
Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []


trial_end_indexes = []
trial_end_index = 0


# print(len(time_data))


for i in range(len(time_data)):
    time_og = np.array(time_data[i])
    position_og = np.array(position_data[i])
    theta_og = np.array(theta_data[i])
    
    gain = final_tg[i]
    if len(time_og) < 510:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 6)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]

    
    
    if pct==0:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue

    # obs_val, action_val,  conditional = get_conditional_pdf(theta_og[0:len(steps)], steps)
    # sparsity = len(np.where(conditional<0.01)[0])/40000

    acf = autocorrelation(filtered_steps)
    acfs.append(acf[400])
    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))
    opt_tau_values, opt_tau_samples, opt_tau_dist = get_noise_pdf_and_samples(opt_time, len(opt_time), np.max(opt_time), np.min(opt_time))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    variance_tau = np.trapz(opt_tau_values**2 * opt_tau_dist, opt_tau_values) - (np.trapz(opt_tau_values * opt_tau_dist, opt_tau_values))**2
    std_dev_tau = variance_tau

    variance_noise = np.trapz(filter_residual_values**2 * filter_noise_kde, filter_residual_values) - (np.trapz(filter_residual_values * filter_noise_kde, filter_residual_values))**2
    std_dev_noise = variance_kd

    # mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    # mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]
    # mean_tau = opt_tau_values[np.argmax(opt_tau_dist)]

    mean_kp = np.sum(opt_kp_values * (opt_kp_dist/np.linalg.norm(opt_kp_dist)))
    mean_kd = np.sum(opt_kd_values * (opt_kd_dist/np.linalg.norm(opt_kd_dist)))
    mean_tau = np.sum(opt_tau_values * (opt_tau_dist/np.linalg.norm(opt_tau_dist)))


    trial_end_index = trial_end_index + len(opt_time)

    theta_stacked = np.hstack((theta_stacked,theta_og[0:len(opt_kp)]))
    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)
    meanstau_across_trials.append(mean_tau)
    stdtau_across_trials.append(std_dev_tau)
    skew_tau_across_trials.append(skew(opt_time))
    # sparsities.append(sparsity)
    noise_vars.append(std_dev_noise)

    if gain == 1:
        meankp_ones.append(mean_kp)
        meankd_ones.append(mean_kd)
        stdkp_ones.append(std_dev_kp)
        stdkd_ones.append(std_dev_kd)
        meantau_ones.append(mean_tau)
        stdtau_ones.append(std_dev_tau)


    trial_end_indexes.append(trial_end_index)




trial_end_indexes = np.array(trial_end_indexes)
trial_end_indexes = np.insert(trial_end_indexes,0,0)
trial_end_indexes = trial_end_indexes

trial_labels = [str(tg) for tg in final_tg]


trial_labels = np.append(trial_labels, 'end')

timescales_across_trials = np.hstack(timescales_across_trials)
Kps_across_trials = np.hstack(Kps_across_trials)
Kds_across_trials = np.hstack(Kds_across_trials)
rate_across_trials = 1/timescales_across_trials
Kps_stacked = np.hstack((Kps_stacked, Kps_across_trials))

axs[phase-1].plot(Kps_across_trials, c = 'red', alpha = 0.3)
axs[phase-1].plot(gaussian_filter1d(Kps_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs[phase-1].set_ylabel('$K_p$')
axs[phase-1].set_title('Phase ' + str(phase))


axs2[phase-1].plot(Kds_across_trials, c = 'blue', alpha = 0.3)
axs2[phase-1].plot(gaussian_filter1d(Kds_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs2[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs2[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs2[phase-1].set_ylabel('$K_d$')
axs2[phase-1].set_title('Phase ' + str(phase))


# axs3[phase-1].plot(timescales_across_trials, c = 'green')
# for index in trial_end_indexes:
#     axs3[phase - 1].axvline(index, linestyle = '--', c = 'black')
# axs3[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
# axs3[phase-1].set_ylabel('Optimization window length')
# axs3[phase-1].set_title('Phase ' + str(phase))

# plt.plot(Kps_across_trials)
# for index in trial_end_indexes:
#     plt.axvline(index, linestyle = '--', c = 'black')

# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kp')
# plt.title('Participant ' + str(P_id))
# plt.show()

# plt.plot(Kds_across_trials)
# for index in trial_end_indexes:
#     plt.axvline(index, linestyle = '--', c = 'black')

# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kd')
# plt.title('Participant ' + str(P_id))
# plt.show()

# plt.plot(meanstau_across_trials[len(meanstau_across_trials)-len(time_data):len(meanstau_across_trials)])
# plt.show()

# plt.plot(stdtau_across_trials[len(stdtau_across_trials)-len(time_data):len(stdtau_across_trials)])
# plt.show()

# plt.plot(skew_tau_across_trials[len(skew_tau_across_trials)-len(time_data):len(skew_tau_across_trials)])
# plt.show()

# plt.plot(kurt_tau_across_trials[len(kurt_tau_across_trials)-len(time_data):len(kurt_tau_across_trials)])
# plt.show()

# print(len(kurt_tau_across_trials), len(sparsities))


# plt.scatter(skew_tau_across_trials, sparsities)
# plt.show()

# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kps', Kps_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kds', Kds_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Taus',timescales_across_trials )
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Endindices', trial_end_indexes)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Gains', final_tg)



##---------------------------------------------------------------------PHASE 3-----------------------------------------------------------------------------------------##

# P_id = 9
# training_gains = config.TRIAL_GAINS_9
phase = 3
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
gain_across_trials.append(final_tg)
Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []

trial_end_indexes = []
trial_end_index = 0


# print(len(time_data))


for i in range(len(time_data)):
    time_og = np.array(time_data[i])
    position_og = np.array(position_data[i])
    theta_og = np.array(theta_data[i])
    gain = final_tg[i]
    if len(time_og) < 510:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 6)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]

    

    if pct==0:
        meanskp_across_trials.append(None)
        meanskd_across_trials.append(None)
        stdkp_across_trials.append(None)
        stdkd_across_trials.append(None)
        final_tg[i] = None

        trial_end_indexes.append(trial_end_index)
        continue
    
   
    # obs_val, action_val,  conditional = get_conditional_pdf(theta_og[0:len(steps)], steps)
    # sparsity = len(np.where(conditional<0.01)[0])/40000

    acf = autocorrelation(filtered_steps)
    acfs.append(acf[400])

    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))
    opt_tau_values, opt_tau_samples, opt_tau_dist = get_noise_pdf_and_samples(opt_time, len(opt_time), np.max(opt_time), np.min(opt_time))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    variance_tau = np.trapz(opt_tau_values**2 * opt_tau_dist, opt_tau_values) - (np.trapz(opt_tau_values * opt_tau_dist, opt_tau_values))**2
    std_dev_tau = variance_tau

    variance_noise = np.trapz(filter_residual_values**2 * filter_noise_kde, filter_residual_values) - (np.trapz(filter_residual_values * filter_noise_kde, filter_residual_values))**2
    std_dev_noise = variance_kd

    # mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    # mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]
    # mean_tau = opt_tau_values[np.argmax(opt_tau_dist)]

    mean_kp = np.sum(opt_kp_values * (opt_kp_dist/np.linalg.norm(opt_kp_dist)))
    mean_kd = np.sum(opt_kd_values * (opt_kd_dist/np.linalg.norm(opt_kd_dist)))
    mean_tau = np.sum(opt_tau_values * (opt_tau_dist/np.linalg.norm(opt_tau_dist)))

 

    trial_end_index = trial_end_index + len(opt_time)

    theta_stacked = np.hstack((theta_stacked,theta_og[0:len(opt_kp)]))
    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)
    meanstau_across_trials.append(mean_tau)
    stdtau_across_trials.append(std_dev_tau)
    skew_tau_across_trials.append(skew(opt_time))
    # sparsities.append(sparsity)
    noise_vars.append(std_dev_noise)

    if gain == 1:
        meankp_ones.append(mean_kp)
        meankd_ones.append(mean_kd)
        stdkp_ones.append(std_dev_kp)
        stdkd_ones.append(std_dev_kd)
        meantau_ones.append(mean_tau)
        stdtau_ones.append(std_dev_tau)


    trial_end_indexes.append(trial_end_index)




trial_end_indexes = np.array(trial_end_indexes)
trial_end_indexes = np.insert(trial_end_indexes,0,0)
trial_end_indexes = trial_end_indexes

trial_labels = [str(tg) for tg in final_tg]


trial_labels = np.append(trial_labels, 'end')

timescales_across_trials = np.hstack(timescales_across_trials)
Kps_across_trials = np.hstack(Kps_across_trials)
Kds_across_trials = np.hstack(Kds_across_trials)
rate_across_trials = 1/timescales_across_trials

Kps_stacked = np.hstack((Kps_stacked, Kps_across_trials))



axs[phase-1].plot(Kps_across_trials, c = 'red', alpha = 0.3)
axs[phase-1].plot(gaussian_filter1d(Kps_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs[phase-1].set_ylabel('$K_p$')
axs[phase-1].set_title('Phase ' + str(phase))
axs[phase-1].set_xlabel('Consectutive trial $\gamma$s')

axs2[phase-1].plot(Kds_across_trials, c = 'blue', alpha = 0.3)
axs2[phase-1].plot(gaussian_filter1d(Kds_across_trials,sigma), c = 'green')
for index in trial_end_indexes:
    axs2[phase - 1].axvline(index, linestyle = '--', c = 'black')
axs2[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
axs2[phase-1].set_ylabel('$K_d$')
axs2[phase-1].set_title('Phase ' + str(phase))
axs2[phase-1].set_xlabel('Consectutive trial $\gamma$s')




# axs3[phase-1].plot(timescales_across_trials, c = 'green')
# for index in trial_end_indexes:
#     axs3[phase - 1].axvline(index, linestyle = '--', c = 'black')
# axs3[phase-1].set_xticks(trial_end_indexes, labels=trial_labels)
# axs3[phase-1].set_ylabel('Optimization window length')
# axs3[phase-1].set_title('Phase ' + str(phase))
# axs3[phase-1].set_xlabel('Consectutive trial $\gamma$s')


# plt.plot(Kps_across_trials)
# for index in trial_end_indexes:
#     plt.axvline(index, linestyle = '--', c = 'black')

# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kp')
# plt.title('Participant ' + str(P_id))
# plt.show()

# plt.plot(Kds_across_trials)
# for index in trial_end_indexes:
#     plt.axvline(index, linestyle = '--', c = 'black')

# plt.xticks(trial_end_indexes, labels=trial_labels)
# plt.xlabel('Consectutive trial gains')
# plt.ylabel('Kd')
# plt.title('Participant ' + str(P_id))
# plt.show()


# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kps', Kps_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kds', Kds_across_trials)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Taus',timescales_across_trials )
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Endindices', trial_end_indexes)
# np.save('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Gains', final_tg)

###----------------------------------------------------------------------------END-------------- --------------------------------------------------------------------###


custom_lines = [Line2D([0], [0], color='red', lw=1, alpha = 0.3),
                Line2D([0], [0], color='green', lw=1)]

fig.legend(custom_lines, ['$K_p$', 'Gaussian filtered $K_p$'], loc = 'upper right')

custom_lines = [Line2D([0], [0], color='blue', lw=1, alpha = 0.3),
                Line2D([0], [0], color='green', lw=1)]

fig2.legend(custom_lines, ['$K_d$', 'Gaussian filtered $K_d$'], loc = 'upper right')


# Set common title
fig.suptitle('Participant ' + str(P_id) + ' - Evolution of $K_p$ across trials', fontsize=16)
fig2.suptitle('Participant ' + str(P_id) + ' - Evolution of $K_d$ across trials', fontsize=16)
# fig3.suptitle('Participant ' + str(P_id) + ' - Evolution of optimization window lengths across trials', fontsize=16)

# Adjust layout
# plt.tight_layout()

# Show plot
plt.show()

# print(len(theta_stacked), len(Kps_stacked))
# plt.scatter(theta_stacked, (Kps_stacked), c='black')
# y_pred, std_dev = fit_linear_regression(theta_stacked, (Kps_stacked))
# plt.plot(theta_stacked, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# corr, _ = pearsonr(theta_stacked, (Kps_stacked))
# plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
# plt.xlabel('theta')
# plt.ylabel('Kp')
# plt.legend()
# plt.show()

meanskp_across_trials = np.array(meanskp_across_trials)
meanskp_across_trials = meanskp_across_trials[meanskp_across_trials != np.array(None)]

meanskd_across_trials = np.array(meanskd_across_trials)
meanskd_across_trials = meanskd_across_trials[meanskd_across_trials != np.array(None)]

meanstau_across_trials = np.array(meanstau_across_trials)
meanstau_across_trials = meanstau_across_trials[meanstau_across_trials != np.array(None)]

stdkp_across_trials = np.array(stdkp_across_trials)
stdkp_across_trials = stdkp_across_trials[stdkp_across_trials != np.array(None)]

stdkd_across_trials = np.array(stdkd_across_trials)
stdkd_across_trials = stdkd_across_trials[stdkd_across_trials != np.array(None)]

stdtau_across_trials = np.array(stdtau_across_trials)
stdtau_across_trials = stdtau_across_trials[stdtau_across_trials != np.array(None)]


meankp_ones = np.array(meankp_ones)
meankd_ones = np.array(meankd_ones)
stdkp_ones = np.array(stdkp_ones)
stdkd_ones = np.array(stdkd_ones)
meantau_ones = np.array(meantau_ones)
stdtau_ones = np.array(stdtau_ones)


noise_vars = np.array(noise_vars)
noise_vars = noise_vars[noise_vars != np.array(None)]

skew_tau_across_trials = np.array(skew_tau_across_trials)
sparsities = np.array(sparsities)

final_tg = np.hstack(gain_across_trials)
final_tg = final_tg[final_tg != np.array(None)]

acfs = np.array(acfs)

indexes = np.where(final_tg!=1)[0]

plt.scatter(final_tg[indexes], meanskp_across_trials[indexes], c='black')
y_pred, std_dev = fit_linear_regression(final_tg[indexes], meanskp_across_trials[indexes])
plt.plot(final_tg[indexes], y_pred, color='red', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(final_tg[indexes], meanskp_across_trials[indexes])
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.xlabel('Gain')
plt.ylabel('Expectation of Kp guesses')
plt.legend()
plt.show()


plt.scatter(final_tg[indexes], stdkp_across_trials[indexes], c='black')
y_pred, std_dev = fit_linear_regression(final_tg[indexes], stdkp_across_trials[indexes])
plt.plot(final_tg[indexes], y_pred, color='red', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(final_tg[indexes], stdkp_across_trials[indexes])
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.xlabel('Gain')
plt.ylabel('Variance of Kp guesses')
plt.legend()
plt.show()



plt.scatter(final_tg[indexes], meanskd_across_trials[indexes], c = 'black')
y_pred, std_dev = fit_linear_regression(final_tg[indexes], meanskd_across_trials[indexes])
plt.plot(final_tg[indexes], y_pred, color='blue', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(final_tg[indexes], meanskd_across_trials[indexes])
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.xlabel('Gain')
plt.ylabel('Expectation of Kd guesses')
plt.legend()
plt.show()


plt.scatter(final_tg[indexes], stdkd_across_trials[indexes], label = 'Data', c='black')
y_pred, std_dev = fit_linear_regression(final_tg[indexes], stdkd_across_trials[indexes])
plt.plot(final_tg[indexes], y_pred, color='blue', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(final_tg[indexes], stdkd_across_trials[indexes])
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.xlabel('Gain')
plt.ylabel('Variance of Kd guesses')
plt.legend()
plt.show()



plt.scatter(stdkd_across_trials, acfs,c='black')
y_pred, std_dev = fit_linear_regression(stdkd_across_trials, acfs)
plt.plot(stdkd_across_trials, y_pred, color='green', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(acfs, stdkd_across_trials)
plt.ylabel('Autocorrelation of step data')
plt.xlabel('Reaction time variance')
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.legend()
plt.show()


plt.scatter(meanstau_across_trials, acfs,  c='black')
y_pred, std_dev = fit_linear_regression(meanstau_across_trials, acfs)
plt.plot(meanstau_across_trials, y_pred, color='green', label='Linear Regression Fit')  # Regression line
corr, _ = pearsonr(acfs, meanstau_across_trials)
plt.xlabel('Reaction time expectation')
plt.ylabel('Autocorrelation of step data')
plt.title('Participant ' + str(P_id))
plt.title('Participant ' + str(P_id) + ' corr = ' + str(corr))
plt.legend()
plt.show()


plt.plot(meantau_ones, c = 'green')
plt.title('Participant ' + str(P_id) + ' Evolution of expectation of reaction times for a visual gain of 1')
plt.xlabel('Trials')
# plt.ylabel('Expectation of Reaction time')
plt.legend()
plt.show()

plt.plot(stdkd_ones, c = 'green')
plt.title('Participant ' + str(P_id) + ' Evolution of variance of reaction times for a visual gain of 1')
plt.xlabel('Trials')
# plt.ylabel('Expectation of Reaction time')
plt.legend()
plt.show()
