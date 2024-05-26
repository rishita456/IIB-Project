import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import config
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from get_data import separate_data, get_joint_pdf, get_conditional_pdf, get_marginal_pdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

P_id = 2
training_gains = config.TRIAL_GAINS_2
phase = 2
time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)
Kps_across_trials = []
timescales_across_trials = []
Kds_across_trials = []

trial_end_indexes = []
trial_end_index = 0


print(len(time_data))
colormaps = ['Purples', 'Blues', 'Greens','Reds']
colors = ['purple', 'blue', 'green','red']



# Create a figure with subplots
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(3, 4)  # 2 rows, 1 column, with the second row taller
ax5 = fig.add_subplot(gs[2, :])

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

    


    opt_kp_values, opt_kp_dist = get_marginal_pdf(opt_kp)
    opt_kd_values, opt_kd_dist = get_marginal_pdf(opt_kd)
    opt_tau_values, opt_tau_dist = get_marginal_pdf(opt_time)

 

    variance_kp = np.trapz(opt_kp_values**2 * opt_kp_dist, opt_kp_values) - (np.trapz(opt_kp_values * opt_kp_dist, opt_kp_values))**2
    std_dev_kp = variance_kp

    variance_kd = np.trapz(opt_kd_values**2 * opt_kd_dist, opt_kd_values) - (np.trapz(opt_kd_values * opt_kd_dist, opt_kd_values))**2
    std_dev_kd = variance_kd

    mean_kp = opt_kp_values[np.argmax(opt_kp_dist)]
    mean_kd = opt_kd_values[np.argmax(opt_kd_dist)]

    trial_end_index = trial_end_index + len(opt_time)

    if final_tg[i] != 1:

        
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.contour(obs_val, action_val, conditional, cmap=colormaps[plot_index%4])
        ax1.set_xlim((-5,5))
        ax1.set_ylim((-0.4,0.4))
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(opt_tau_values, opt_tau_dist, label = 'gain = ' + str(gain) + ' trial = ' + str(i), c = colors[plot_index%4])
        ax5.legend()
        plot_index = plot_index+1
    
            
       
        



# Adjust layout and display the figure
plt.tight_layout()
plt.show()


    