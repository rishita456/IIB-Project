import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import config
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples


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


meanskp_across_trials = []
stdkp_across_trials = []
meanskd_across_trials = []
stdkd_across_trials = []
gain_across_trials = []
meankp_ones = []
stdkp_ones = []
meankd_ones = []
stdkd_ones = []
tau_ones = []


P_id = 9
training_gains = config.TRIAL_GAINS_9

##---------------------------------------------------------------------PHASE 1-----------------------------------------------------------------------------------------##


phase = 1
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
gain_across_trials.append(final_tg)

Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []

trial_end_indexes = []
trial_end_index = 0


print(len(time_data))


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

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 5, 5)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    
   

    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]

    trial_end_index = trial_end_index + len(opt_time)


    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)

    if gain == 1:
        meankp_ones.append(mean_kp)
        meankd_ones.append(mean_kd)
        tau_ones.append(opt_time)

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




##---------------------------------------------------------------------PHASE 2-----------------------------------------------------------------------------------------##


phase = 2
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
gain_across_trials.append(final_tg)
Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []

trial_end_indexes = []
trial_end_index = 0


print(len(time_data))


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

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 5, 5)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    
   

    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]

    trial_end_index = trial_end_index + len(opt_time)


    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)

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


print(len(time_data))


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

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1, 5, 5)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]
    
   

    opt_kp_values, opt_kp_samples, opt_kp_dist = get_noise_pdf_and_samples(opt_kp, len(opt_kp))
    opt_kd_values, opt_kd_samples, opt_kd_dist = get_noise_pdf_and_samples(opt_kd, len(opt_kd))

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]

    trial_end_index = trial_end_index + len(opt_time)


    timescales_across_trials.append(opt_time)
    Kps_across_trials.append(opt_kp)
    Kds_across_trials.append(opt_kd)
    meanskp_across_trials.append(mean_kp)
    meanskd_across_trials.append(mean_kd)
    stdkp_across_trials.append(std_dev_kp)
    stdkd_across_trials.append(std_dev_kd)

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





###----------------------------------------------------------------------------END-------------- --------------------------------------------------------------------###






meanskp_across_trials = np.array(meanskp_across_trials)
meanskp_across_trials = meanskp_across_trials[meanskp_across_trials != np.array(None)]

meanskd_across_trials = np.array(meanskd_across_trials)
meanskd_across_trials = meanskd_across_trials[meanskd_across_trials != np.array(None)]

stdkp_across_trials = np.array(stdkp_across_trials)
stdkp_across_trials = stdkp_across_trials[stdkp_across_trials != np.array(None)]

stdkd_across_trials = np.array(stdkd_across_trials)
stdkd_across_trials = stdkd_across_trials[stdkd_across_trials != np.array(None)]

final_tg = np.hstack(gain_across_trials)
final_tg = final_tg[final_tg != np.array(None)]





plt.scatter(final_tg, meanskp_across_trials, label = 'Data', c='black')

#Linear
y_pred, std_dev = fit_linear_regression(final_tg, meanskp_across_trials)
plt.plot(final_tg, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='red', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 2
y_pred, std_dev = fit_polynomial_regression(final_tg, meanskp_across_trials, 2)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='green', label='Polynomial Regression Fit degree 2')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='green', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 3
y_pred, std_dev = fit_polynomial_regression(final_tg, meanskp_across_trials, 3)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='blue', label='Polynomial Regression Fit degree 3')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='blue', alpha=0.6,
#              label='Standard Deviation', capsize=5)

plt.xlabel('Gain')
plt.ylabel('Mean guess for Kp')
plt.title('Participant ' + str(P_id))
plt.legend()
plt.show()


plt.scatter(final_tg, stdkp_across_trials, label = 'Data', c='black')

#Linear
y_pred, std_dev = fit_linear_regression(final_tg, stdkp_across_trials)
plt.plot(final_tg, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='red', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 2
y_pred, std_dev = fit_polynomial_regression(final_tg, stdkp_across_trials, 2)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='green', label='Polynomial Regression Fit degree 2')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='green', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 3
y_pred, std_dev = fit_polynomial_regression(final_tg, stdkp_across_trials, 3)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='blue', label='Polynomial Regression Fit degree 3')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='blue', alpha=0.6,
#              label='Standard Deviation', capsize=5)


plt.xlabel('Gain')
plt.ylabel('variance of Kp guesses')
plt.title('Participant ' + str(P_id))
plt.legend()
plt.show()



plt.scatter(final_tg, meanskd_across_trials, label = 'Data', c = 'black')

#Linear
y_pred, std_dev = fit_linear_regression(final_tg, meanskd_across_trials)
plt.plot(final_tg, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='red', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 2
y_pred, std_dev = fit_polynomial_regression(final_tg, meanskd_across_trials, 2)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='green', label='Polynomial Regression Fit degree 2')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='green', alpha=0.6,
#              label='Standard Deviation', capsize=5)

#Polynomial order 3
y_pred, std_dev = fit_polynomial_regression(final_tg, meanskd_across_trials, 3)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='blue', label='Polynomial Regression Fit degree 3')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='blue', alpha=0.6,
#              label='Standard Deviation', capsize=5)

plt.xlabel('Gain')
plt.ylabel('Mean guess for Kd')
plt.title('Participant ' + str(P_id))
plt.legend()
plt.show()


plt.scatter(final_tg, stdkd_across_trials, label = 'Data', c='black')

#Linear
y_pred, std_dev = fit_linear_regression(final_tg, stdkd_across_trials)
plt.plot(final_tg, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='red', alpha=0.6,
#              label='Standard Deviation', capsize=5)
#Polynomial order 2
y_pred, std_dev = fit_polynomial_regression(final_tg, stdkd_across_trials, 2)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='green', label='Polynomial Regression Fit degree 2')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='green', alpha=0.6,
#              label='Standard Deviation', capsize=5)

#Polynomial order 3
y_pred, std_dev = fit_polynomial_regression(final_tg, stdkd_across_trials, 3)
sort_indices = np.argsort(final_tg.squeeze())
plt.plot(final_tg[sort_indices], y_pred[sort_indices], color='blue', label='Polynomial Regression Fit degree 3')  # Regression curve
# plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='blue', alpha=0.6,
#              label='Standard Deviation', capsize=5)


plt.xlabel('Gain')
plt.ylabel('variance of Kd guesses')
plt.title('Participant ' + str(P_id))
plt.legend()
plt.show()

