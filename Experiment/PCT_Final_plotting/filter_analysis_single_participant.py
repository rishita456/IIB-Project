
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from scipy.stats import entropy
from scipy.optimize import minimize
# from individual_participant_training import separate_data
import config
import matplotlib

def get_aggregate_data(time_data, position_data, theta_data, final_tg):

        unique_gains = np.unique(final_tg)
        aggregate_filter_res = np.array([])
        aggregate_fit_res = np.array([])
        for i in range(len(unique_gains)):
                pos = np.where(final_tg == unique_gains[i])
                all_filter_residuals = np.array([])
                all_fit_residuals =  np.array([])
                for p in pos:
                        time_og = np.array(time_data[p])
                        position_og = np.array(position_data[p])
                        theta_og = np.array(theta_data[p])

                        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
                        # all_filter_residuals.append(filter_residuals)
                        # all_fit_residuals.append(fit_residuals)
                        all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
                        all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

                aggregate_filter_res = np.append(aggregate_filter_res, all_filter_residuals)
                aggregate_fit_res = np.append(aggregate_fit_res, all_fit_residuals)

        
        return unique_gains, aggregate_filter_res, aggregate_fit_res
                



                

# P_id = 3
# training_gains = config.TRIAL_GAINS_3

# phase = 1
# time_data, position_data, theta_data, final_tg = separate_data(training_gains, phase, P_id)


# Create a figure and subplots
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 8))

P_id = 8
# gains = [2, 5, 7, 10, 12, 15, 17]
gains = [2, 5, 7, 10, 12, 15, 17]
trials = [1,2]
c = np.linspace(0,0.75,len(gains))
cmap = matplotlib.cm.get_cmap('magma')
cutoff = 0.2
fs = 120
order = 5
# cutoffs = np.arange(0.01,1,0.1)
cutoffs = np.arange(1,15,1)
cutoffs = [0.2, 0.5, 1 ,1.5, 2, 4, 5 ,7, 9, 10, 15]
# cutoffs = [0.2, 0.5, 1 ,1.5, 2, 4, 5]


for i in range(len(gains)):
        kl_divs = []
        gain = gains[i]
#     for trial in trials:
        for cutoff in cutoffs:
                all_filter_residuals = np.array([])
                all_fit_residuals =  np.array([])
                for trial in trials:

                        time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
                        position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
                        theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

                        steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
                        # all_filter_residuals.append(filter_residuals)
                        # all_fit_residuals.append(fit_residuals)
                        all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
                        all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

                all_filter_residuals = np.array(all_filter_residuals).flatten()
                all_fit_residuals = np.array(all_fit_residuals).flatten()

                filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
                fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
                time = time_og[0:len(steps)]

                kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
        # plt.show()
        rgba = cmap(c[i])
        # c = (rgba[0], rgba[1], rgba[2])

        axs[0].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
        textstr = '     '.join(["Participant" + str((P_id))])
        props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
        axs[0].text(0.5, 0.9, textstr, transform=axs[0].transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right', bbox=props)
# P_id = 4

# for i in range(len(gains)):
#         kl_divs = []
#         gain = gains[i]
# #     for trial in trials:
#         for cutoff in cutoffs:
#                 all_filter_residuals = np.array([])
#                 all_fit_residuals =  np.array([])
#                 for trial in trials:

#                         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#                         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#                         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

#                         steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
#                         # all_filter_residuals.append(filter_residuals)
#                         # all_fit_residuals.append(fit_residuals)
#                         all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
#                         all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

#                 all_filter_residuals = np.array(all_filter_residuals).flatten()
#                 all_fit_residuals = np.array(all_fit_residuals).flatten()

#                 filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
#                 fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
#                 time = time_og[0:len(steps)]

#                 kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
#         # plt.show()
#         rgba = cmap(c[i])
#         # c = (rgba[0], rgba[1], rgba[2])

#         axs[0, 1].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
#         textstr = '     '.join(["Participant" + str((P_id))])
#         props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
#         axs[0, 1].text(0.5, 0.9, textstr, transform=axs[0,1].transAxes, fontsize=10,
#                        verticalalignment='top', horizontalalignment='right', bbox=props)

# P_id = 5

# for i in range(len(gains)):
#         kl_divs = []
#         gain = gains[i]
# #     for trial in trials:
#         for cutoff in cutoffs:
#                 all_filter_residuals = np.array([])
#                 all_fit_residuals =  np.array([])
#                 for trial in trials:

#                         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#                         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#                         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

#                         steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
#                         # all_filter_residuals.append(filter_residuals)
#                         # all_fit_residuals.append(fit_residuals)
#                         all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
#                         all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

#                 all_filter_residuals = np.array(all_filter_residuals).flatten()
#                 all_fit_residuals = np.array(all_fit_residuals).flatten()

#                 filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
#                 fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
#                 time = time_og[0:len(steps)]

#                 kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
#         # plt.show()
#         rgba = cmap(c[i])
#         # c = (rgba[0], rgba[1], rgba[2])

#         axs[0, 2].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
#         textstr = '     '.join(["Participant" + str((P_id))])
#         props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
#         axs[0, 2].text(0.5, 0.9, textstr, transform=axs[0,2].transAxes, fontsize=10,
#                        verticalalignment='top', horizontalalignment='right', bbox=props)


# P_id = 6

# for i in range(len(gains)):
#         kl_divs = []
#         gain = gains[i]
# #     for trial in trials:
#         for cutoff in cutoffs:
#                 all_filter_residuals = np.array([])
#                 all_fit_residuals =  np.array([])
#                 for trial in trials:

#                         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#                         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#                         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

#                         steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
#                         # all_filter_residuals.append(filter_residuals)
#                         # all_fit_residuals.append(fit_residuals)
#                         all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
#                         all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

#                 all_filter_residuals = np.array(all_filter_residuals).flatten()
#                 all_fit_residuals = np.array(all_fit_residuals).flatten()

#                 filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
#                 fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
#                 time = time_og[0:len(steps)]

#                 kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
#         # plt.show()
#         rgba = cmap(c[i])
#         # c = (rgba[0], rgba[1], rgba[2])

#         axs[1, 0].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
#         textstr = '     '.join(["Participant" + str((P_id))])
#         props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
#         axs[1, 0].text(0.5, 0.9, textstr, transform=axs[1,0].transAxes, fontsize=10,
#                        verticalalignment='top', horizontalalignment='right', bbox=props)


# P_id = 8

# for i in range(len(gains)):
#         kl_divs = []
#         gain = gains[i]
# #     for trial in trials:
#         for cutoff in cutoffs:
#                 all_filter_residuals = np.array([])
#                 all_fit_residuals =  np.array([])
#                 for trial in trials:

#                         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#                         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#                         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

#                         steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
#                         # all_filter_residuals.append(filter_residuals)
#                         # all_fit_residuals.append(fit_residuals)
#                         all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
#                         all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

#                 all_filter_residuals = np.array(all_filter_residuals).flatten()
#                 all_fit_residuals = np.array(all_fit_residuals).flatten()

#                 filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
#                 fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
#                 time = time_og[0:len(steps)]

#                 kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
#         # plt.show()
#         rgba = cmap(c[i])
#         # c = (rgba[0], rgba[1], rgba[2])

#         axs[1, 1].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
#         textstr = '     '.join(["Participant" + str((P_id))])
#         props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
#         axs[1, 1].text(0.5, 0.9, textstr, transform=axs[1,1].transAxes, fontsize=10,
#                        verticalalignment='top', horizontalalignment='right', bbox=props)


P_id = 9

# for i in range(len(gains)):
#         kl_divs = []
#         gain = gains[i]
# #     for trial in trials:
#         for cutoff in cutoffs:
#                 all_filter_residuals = np.array([])
#                 all_fit_residuals =  np.array([])
#                 for trial in trials:

#                         time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
#                         position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
#                         theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                        

#                         steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, cutoff)
#                         # all_filter_residuals.append(filter_residuals)
#                         # all_fit_residuals.append(fit_residuals)
#                         all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
#                         all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))

#                 all_filter_residuals = np.array(all_filter_residuals).flatten()
#                 all_fit_residuals = np.array(all_fit_residuals).flatten()

#                 filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
#                 fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
#                 time = time_og[0:len(steps)]

#                 kl_divs.append(entropy(filter_noise_kde, fit_noise_kde))

        
#         # plt.show()
#         rgba = cmap(c[i])
#         # c = (rgba[0], rgba[1], rgba[2])

#         axs[1].plot(cutoffs, kl_divs, label = '$\gamma$ = ' + str(gain), alpha =0.7, c = (rgba[0], rgba[1], rgba[2]))
#         textstr = '     '.join(["Participant" + str((P_id))])
#         props = dict(boxstyle='round', facecolor='orange', alpha=0.15)
#         axs[1].text(0.5, 0.9, textstr, transform=axs[1].transAxes, fontsize=10,
#                        verticalalignment='top', horizontalalignment='right', bbox=props)




# Add x and y labels
# for ax in axs[-1,:]:
        # axs[].set_xlabel('Cut off frequency, $f_{cutoff}$ (Hz)')
for ax in axs[:]:
    ax.set_xlabel('Cut off frequency, $f_{cutoff}$ (Hz)')
    ax.set_ylabel("$D_{KL}(P(r_{filt}|{\gamma}, f_{cutoff})||P(r_{fit}|{\gamma}, f_{cutoff}))$")

# Define a common legend outside the subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')


fig.suptitle('KL divergence between $P(r_{filt}|{\gamma}, f_{cutoff})$ and $P(r_{fit}|{\gamma}, f_{cutoff})$')
# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


