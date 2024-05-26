
import numpy as np
import matplotlib.pyplot as plt
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from scipy.stats import entropy
from scipy.optimize import minimize, brute, basinhopping



def find_cutoff(R):

        all_filter_residuals = np.array([])
        all_fit_residuals =  np.array([])
        for trial in trials:

                time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
                position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
                theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
                

                steps, filtered_steps, step_est, filter_residuals, fit_residuals, opt_kp, opt_kd, opt_time, size, performance, pct_performance, pct = get_step_estimates(time_og, position_og, theta_og, gain, 1, R)
                # all_filter_residuals.append(filter_residuals)
                # all_fit_residuals.append(fit_residuals)
                all_filter_residuals = np.hstack((all_filter_residuals, filter_residuals))
                all_fit_residuals = np.hstack((all_fit_residuals, fit_residuals))
        filter_residual_values, filter_noise_samples, filter_noise_kde = get_noise_pdf_and_samples(all_filter_residuals, len(all_filter_residuals))
        fit_residual_values, fit_noise_samples, fit_noise_kde = get_noise_pdf_and_samples(all_fit_residuals, len(all_fit_residuals))
        
        kl_divs = entropy(filter_noise_kde, fit_noise_kde)
        print(R)



        return kl_divs






gain = 10
trial = 1
P_id = 3
# gains = [2, 5, 7, 10, 12, 15, 17]
gains = [2, 5, 7, 10, 12, 15, 17]
trials = [1,2]

range = (slice(1, 15, 0.25),slice(1, 15, 0.25))
cutoff = 0.2
fs = 120
order = 5
# cutoffs = np.arange(0.01,1,0.1)
cutoffs = []
for gain in gains:

        result = brute(find_cutoff, range)
        cutoffs.append(result[0])
        print(result[1])


# plt.show()


plt.plot(gains, cutoffs)


# plt.ylabel("$D_{KL}(P(r_{filt}|{\gamma}, f_{cutoff})||P(r_{fit}|{\gamma}, f_{cutoff}))$")
plt.ylabel('Cut off frequency (Hz)')
plt.xlabel('Gain')
plt.legend()
plt.show()

