import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import config
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from get_data import separate_data, get_joint_pdf, get_conditional_pdf, get_marginal_pdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels

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



P_id = 2
training_gains = config.TRIAL_GAINS_2
phase = 1
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)


Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []
mean_taus = []
sparsities = []
skew_tau_across_trials = []
kurt_tau_across_trials = []
joint_vars = []
noise_vars = []
max_acf_lag = []
trial_end_indexes = []
trial_end_index = 0
colormaps = ['Purples', 'Blues', 'Greens','Reds']
colors = ['purple', 'blue', 'green','red']

print(len(time_data))


# Create a figure with subplots
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(2, 4, height_ratios=[2, 3])  # 2 rows, 1 column, with the second row taller
ax5 = fig.add_subplot(gs[1, :])

plot_index = 0
for i in range(len(time_data)):
    time_og = np.array(time_data[i])
    position_og = np.array(position_data[i])
    theta_og = np.array(theta_data[i])
    gain = final_tg[i]
    if len(time_og) < 510:
        final_tg[i] = None
        trial_end_indexes.append(trial_end_index)
        continue

    steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, gain*theta_og, gain, 1)
    filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(filter_residuals, len(filter_residuals))
    fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(fit_residuals, len(fit_residuals))
    time = time_og[0:len(steps)]

    obs_val, action_val,  conditional = get_conditional_pdf(theta_og[0:len(steps)], steps)
    sparsity = len(np.where(conditional<0.01)[0])/40000
    # var = 0
    # for i in range(200):
    #     variance = np.trapz(action_val**2 * conditional[:,i], action_val) - (np.trapz(action_val * conditional[:,i], action_val))**2
    #     var = var + variance 

    # var = var/200
    # joint_vars.append(var)
    
    
    # plt.figure(figsize=(8, 6))
    # plt.contour(obs_val, action_val, conditional, cmap='viridis')
    # plt.xlabel('Observation')
    # plt.ylabel('Action')
    # plt.title('Conditional Distribution of actions given observations')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    

    # kp_val, kd_val, joint = get_joint_pdf(opt_kp, opt_kd)
    # plt.figure(figsize=(8, 6))
    # plt.contour(kp_val, kd_val, joint, cmap='viridis')
    # plt.colorbar(label='Estimated Joint Density')
    # plt.xlabel('Kp')
    # plt.ylabel("Kd")
    # plt.title('Estimated Joint Distribution P(Kp,Kd) using KDE')
    # plt.show()
    acf = autocorrelation(filtered_steps)
    max_acf_lag.append(acf[1])
    


    opt_kp_values, opt_kp_dist = get_marginal_pdf(opt_kp)
    opt_kd_values, opt_kd_dist = get_marginal_pdf(opt_kd)
    opt_tau_values, opt_tau_dist = get_marginal_pdf(opt_time)

 

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    variance_noise = np.trapz(filter_residual_values**2 * filter_noise_kde, filter_residual_values) - (np.trapz(filter_residual_values * filter_noise_kde, filter_residual_values))**2
    std_dev_noise = variance_kd

    mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]
    mean_tau = opt_tau_values[np.argmax(opt_tau_dist)]


    mean_taus.append(mean_tau)
    sparsities.append(sparsity)
    skew_tau_across_trials.append(skew(opt_time))
    kurt_tau_across_trials.append(kurtosis(opt_time))
    noise_vars.append(std_dev_noise)

    trial_end_index = trial_end_index + len(opt_time)

    if i%4 == 0:

        ax1 = fig.add_subplot(gs[0, plot_index])
        ax1.contour(obs_val, action_val, conditional, cmap=colormaps[plot_index])
        ax1.set_xlim((-5,5))
        ax1.set_ylim((-0.4,0.4))
        
       
        ax5.plot(opt_tau_values, opt_tau_dist, label = 'gain = ' + str(gain), c = colors[plot_index])
        ax5.legend()
        plot_index = plot_index+1



# Adjust layout and display the figure
plt.tight_layout()
plt.show()

# skew_tau_across_trials = np.array(skew_tau_across_trials)
# sparsities = np.array(sparsities)
# noise_vars = np.array(noise_vars)

# plt.scatter(skew_tau_across_trials, noise_vars)
# plt.show()

# plt.scatter(skew_tau_across_trials, sparsities, c = 'r')

# plt.scatter(skew_tau_across_trials, sparsities, label = 'Data', c='black')

# #Linear
# y_pred, std_dev = fit_linear_regression(skew_tau_across_trials, sparsities)
# plt.plot(skew_tau_across_trials, y_pred, color='red', label='Linear Regression Fit')  # Regression line
# # plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='red', alpha=0.6,
# #              label='Standard Deviation', capsize=5)
# #Polynomial order 2
# y_pred, std_dev = fit_polynomial_regression(skew_tau_across_trials, sparsities, 2)
# sort_indices = np.argsort(skew_tau_across_trials.squeeze())
# plt.plot(skew_tau_across_trials[sort_indices], y_pred[sort_indices], color='green', label='Polynomial Regression Fit degree 2')  # Regression curve
# # plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='green', alpha=0.6,
# #              label='Standard Deviation', capsize=5)
# #Polynomial order 3
# y_pred, std_dev = fit_polynomial_regression(skew_tau_across_trials, sparsities, 3)
# sort_indices = np.argsort(skew_tau_across_trials.squeeze())
# plt.plot(skew_tau_across_trials[sort_indices], y_pred[sort_indices], color='blue', label='Polynomial Regression Fit degree 3')  # Regression curve
# # plt.errorbar(final_tg.squeeze(), y_pred, yerr=std_dev, fmt='o', color='blue', alpha=0.6,
# #              label='Standard Deviation', capsize=5)

# plt.xlabel('Skew of reaction time distributions')
# plt.ylabel('')
# plt.title('Sparsity of conditional distribution P(a|o) ' + str(P_id))
# plt.legend()
# plt.show()

# plt.plot()

# final_tg = np.array(final_tg)

# final_tg = final_tg[final_tg != np.array(None)]


# plt.plot(mean_taus)
# plt.xticks(np.arange(1,len(mean_taus)+1), labels=final_tg)
# plt.ylabel('Mode of reaction times per trial')
# plt.xlabel('Consecutive trial gains')
# plt.show()

# plt.plot(max_acf_lag)
# plt.xticks(np.arange(1,len(mean_taus)+1), labels=final_tg)
# plt.ylabel('Lag corresponding to maximum atucorrelation')
# plt.xlabel('Consecutive trial gains')
# plt.show()

# plt.scatter(mean_taus, max_acf_lag)
# plt.show()


    