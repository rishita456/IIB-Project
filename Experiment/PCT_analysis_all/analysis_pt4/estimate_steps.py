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

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def calc_func_find_Kp_0_part0(K, theta, theta_steps, traing, y, d):

   
    # global theta
    # global theta_steps
    # global position
    # global traing
    # global y
    Kp_0 = K[0]
    Kd_0 = d*K[1]
    pg = traing

    return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))/len(y)


def get_step_estimates(time, position, theta, gain, d):
    # gain = 5
    pct = 1
    # Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

    fs = 120      # sample rate, Hz
    cutoff = 5      # desired cutoff frequency of the filter, Hz 
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # high order due to slow roll off


    traing = gain
    size = np.arange(200,501, 10)
    time_og = time
    position_og = position
    theta_og = theta

    losses = np.zeros((len(size),len(time_og))) #total time samples = 500
    Kp = np.zeros((len(size),len(time_og)))
    Kd = np.zeros((len(size),len(time_og)))
    gamma = np.zeros((len(size),len(time_og)))

    # losses = np.zeros((len(size),len(time))) #total time samples = 500
    # Kp = np.zeros((len(size),len(time)))
    # Kd = np.zeros((len(size),len(time)))
    # gamma = np.zeros((len(size),len(time)))
    
    for j in range(len(size)):
        loss = []
        est = []
        gamma_est = []
        Kps = []
        Kds = []
        limits = np.arange(0,len(time_og)+1,size[j])
        for i in range(len(limits)-1):
            
            limit = (limits[i], limits[i+1])
            time = time_og[limit[0]:limit[1]]
            position = position_og[limit[0]:limit[1]]
            theta = theta_og[limit[0]:limit[1]]
    
            x = np.reshape(time, (len(time), 1))
            model = LinearRegression()
            # # pos = position[0:-2]-750
            # pos = position-750

            try:
                # pos = position[0:-2]-750
                pos = position-750

                print(len(x), len(pos))
                model.fit(x, pos)
            except ValueError:
                pos = position[0:-2]-750
                # pos = position-750

                print(len(x), len(pos))
                model.fit(x, pos)


            print(len(x), len(pos))
            model.fit(x, pos)
            # calculate trend and detrend position data
            trend = model.predict(x)
            detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]
            theta_steps = []
            sequence = []
            steps = []
            start = detrended_pos[0]

            for i in range(len(detrended_pos)-1):
                steps.append(detrended_pos[i+1]-detrended_pos[i])
                theta_steps.append(theta[i+1]-theta[i])
            steps = np.array(steps)
            for i in range(len(steps)):
                if steps[i]>0:
                    sequence.append(1)

                else:
                    sequence.append(0)
            print(len(steps))
        
            steps = steps[1::2]
            steps = np.repeat(steps, 2)
            steps = np.insert(steps,0, steps[0])
            print(len(steps))


            # Filter the data
            y = butter_lowpass_filter(steps, cutoff, fs, order)
            theta_steps = np.array(theta_steps)
            theta_steps = np.insert(theta_steps, 0,theta[0])

            

            result = minimize(calc_func_find_Kp_0_part0, (1,1), args=(theta, theta_steps, traing, y, d))
            print(result.x)
            loss.append(result.fun)
            
            est.append((result.x[0])*((gain)**-1)*theta + result.x[0]*((gain)**-1)*theta_steps)
            
            Kps.append(result.x[0])
            Kds.append(result.x[1])

        losses[j, 0:len(np.repeat(loss, size[j]))] = np.repeat(loss, size[j])
        Kp[j, 0:len(np.repeat(Kps, size[j]))] = np.repeat(Kps, size[j])
        Kd[j, 0:len(np.repeat(Kds, size[j]))] = np.repeat(Kds, size[j])
        est = np.array(est).flatten()

    time = time_og
    position = position_og
    theta = theta_og

    dt = time[502] - time[500]

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    # pos = position[0:-2]-750
    # model.fit(x, pos)

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


    theta_steps = []
    steps = []

    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
        theta_steps.append(theta[i+1]-theta[i])
    steps = np.array(steps)


    detrended_pos = np.array(detrended_pos)
    theta_steps = np.array(theta_steps)
    theta_steps = np.insert(theta_steps, 0,theta[0])
    print(len(steps))
    steps = steps[1::2]
    print(len(steps))
    steps = np.repeat(steps, 2)
    steps = np.insert(steps,0, steps[0])
    print(len(steps))

    # Filter the data
    y = butter_lowpass_filter(steps, cutoff, fs, order)

    # filter_residuals = steps - y
    filter_residuals = steps[500:len(time_og)-500] - y[500:len(time_og)-500]
    (filter_noise_mu, filter_noise_sigma) = norm.fit(filter_residuals)

    min_loss_indices = np.argmin(losses, axis=0)
    optimum_kp = np.zeros(len(time_og))
    optimum_kd = np.zeros(len(time_og))
    optimum_timescales = 0.0045*size[np.argmin(losses, axis=0)]

    performance = []
    for i in range(len(size)):
        e = Kp[i, 500:len(time_og)-500]*((traing)**-1)*theta[500:len(time_og)-500] + Kd[i, 500:len(time_og)-500]*((traing)**-1)*theta_steps[500:len(time_og)-500]
        performance.append((np.linalg.norm(y[500:len(time_og)-500] - e))/len(y[500:len(time_og)-500]))
        # print((np.linalg.norm(y[500:len(time_og)-500] - e))/len(y[500:len(time_og)-500]))

        
    performance = np.array(performance)
    best = np.argmin(performance)
    print(performance[best])

    e = Kp[best, 500:len(time_og)-500]*((traing)**-1)*theta[500:len(time_og)-500] + Kd[best, 500:len(time_og)-500]*((traing)**-1)*theta_steps[500:len(time_og)-500]



    for i in range(len(time)):
        optimum_kp[i] = Kp[min_loss_indices[i], i]
        optimum_kd[i] = Kd[min_loss_indices[i], i]

    # final_estimate = optimum_kp*((traing)**-1)*theta + optimum_kd*((traing)**-1)*theta_steps
    # final_estimate = optimum_kp[500:len(time_og)-500]*((traing)**-1)*theta[500:len(time_og)-500] + optimum_kd[500:len(time_og)-500]*((traing)**-1)*theta_steps[500:len(time_og)-500]
    
    for i in range(len(time_og)):
        optimum_kp[i] = Kp[min_loss_indices[i], i]
        optimum_kd[i] = Kd[min_loss_indices[i], i]

    final_estimate = optimum_kp[500:len(time_og)-500]*((traing)**-1)*theta[500:len(time_og)-500] + optimum_kd[500:len(time_og)-500]*((traing)**-1)*theta_steps[500:len(time_og)-500]
    optimum_estimate = final_estimate
    print(np.linalg.norm(y[500:len(time_og)-500] - final_estimate)/len(y[500:len(time_og)-500]))
    if performance[best] <= np.linalg.norm(y[500:len(time_og)-500] - final_estimate)/len(y[500:len(time_og)-500]):
        print('yes')
        pct = 0
        optimum_timescales = (0.0045*size[best])*np.ones_like(optimum_timescales)
        optimum_kp = (Kp[best,:])
        optimum_kd = (Kd[best,:])
        # plt.plot(time_og, theta_og)
        # plt.show()

    final_estimate = optimum_kp[500:len(time_og)-500]*((traing)**-1)*theta[500:len(time_og)-500] + optimum_kd[500:len(time_og)-500]*((traing)**-1)*theta_steps[500:len(time_og)-500]

    fit_residuals = y[500:len(time_og)-500] - final_estimate
    #note not always normal
    (fit_noise_mu, fit_noise_sigma) = norm.fit(fit_residuals)

    fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(final_estimate))
    print(traing)

    return steps[500:len(time_og)-500], y[500:len(time_og)-500], final_estimate, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, optimum_kp[500:len(time_og)-500], optimum_kd[500:len(time_og)-500], optimum_timescales[500:len(time_og)-500], size, performance, ((np.linalg.norm(y[500:len(time_og)-500] - optimum_estimate))/len(y[500:len(time_og)-500])), pct


gain = 5
trial = 2
P_id = 5

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

steps, filtered_steps, step_est, filter_noise_mu, filter_noise_sigma, fit_noise_mu, fit_noise_sigma, opt_kp, opt_kd, opt_time, size, perf, const_time, pct = get_step_estimates(time, position, theta, gain, 1)

filter_noise = np.random.normal(filter_noise_mu, filter_noise_sigma, len(filtered_steps))
fit_noise = np.random.normal(fit_noise_mu, filter_noise_sigma, len(step_est))

plt.plot(time[500:len(time)-500], steps)
plt.show()
plt.plot(time[500:len(time)-500], filtered_steps)
plt.plot(time[500:len(time)-500], step_est)
plt.show()

plt.plot(time[500:len(time)-500], filtered_steps + filter_noise)
plt.plot(time[500:len(time)-500], step_est + filter_noise + fit_noise)
plt.show()

