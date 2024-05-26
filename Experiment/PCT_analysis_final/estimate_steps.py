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

# reason for butterworth - no passband ripple, high attenuation and smooth roll off
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_noise_pdf_and_samples(residuals, n):
    
    # x = np.linspace(-1, 1, n)  # Range of x values for noise
    x = np.linspace(min(residuals), max(residuals), n)  # Range of x values general
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


def get_step_estimates(time, position, theta, gain, d, cutoff, order):

    #boolean value of whether optimal fit follows PCT
    pct = 1

    # Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction
    fs = 120      # sample rate, Hz
    # cutoff = 1   # desired cutoff frequency of the filter, Hz 
    # order = 10   # high order due to slow roll off


    traing = gain
    size = np.arange(200,501, 10) # window sizes over which parameters remain constant
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

    print(len(theta))
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

    losses = np.zeros((len(size),len(time_og))) 
    Kp = np.zeros((len(size),len(time_og)))
    Kd = np.zeros((len(size),len(time_og)))
    estimates= np.zeros((len(size),len(time_og))) 

    


    # iterate over window sizes
    for j in range(len(size)):
        dist = []
        est = []
        Kps = []
        Kds = []

        limits = np.arange(0,len(time_og)+1,size[j])
        for i in range(len(limits)-1):
            
            limit = (limits[i], limits[i+1])
            time = time_og[limit[0]:limit[1]]
            position = position_og[limit[0]:limit[1]]
            theta = theta_og[limit[0]:limit[1]]
            theta_steps = theta_steps_og[limit[0]:limit[1]]
            y = y_og[limit[0]:limit[1]]
            

            result = minimize(calc_func_find_Kp_0_part0, (1,1), args=(theta, theta_steps, traing, y, d))

            print(result.x)
            

            window_estimate = (result.x[0])*((gain)**-1)*theta + result.x[0]*((gain)**-1)*theta_steps
            est.append(window_estimate)
            dist.append(np.abs(y-window_estimate))
            Kps.append(result.x[0]*np.ones(len(window_estimate)))
            Kds.append(result.x[1]*np.ones(len(window_estimate)))

        est_final = np.hstack(np.array(est))
        loss_final = np.hstack(np.array(dist))
        Kps_final = np.hstack(np.array(Kps))
        Kds_final = np.hstack(np.array(Kds))
        print(len(Kps_final))

        losses[j, 0:len(loss_final)] = loss_final
        Kp[j, 0:len(loss_final)] = Kps_final
        Kd[j, 0:len(loss_final)] = Kds_final
        estimates[j, 0:len(loss_final)] = est_final

        length = len(loss_final)


    

    print(length)
    print(np.shape(estimates))
    print(np.shape(losses))
    time = time_og[0:length]
    position = position_og[0:length+2]
    theta = theta_og[0:length]
    y_og = y_og[0:length]
    steps_og2 = steps_og2[0:length]
    theta_steps_og = theta_steps_og[0:length]

    filter_residuals = steps_og2 - y_og


    min_loss_indices = np.argmin(losses, axis=0)[0:length]
    min_losses = np.zeros(length)
    optimum_kp = np.zeros(length)
    optimum_kd = np.zeros(length)
    optimum_estimate = np.zeros(length)
    optimum_timescales = 0.0045*size[np.argmin(losses, axis=0)[0:length]]
 
   
    performance = []
    for i in range(len(size)):
        e = estimates[i,0:length]
        performance.append((np.linalg.norm(np.abs(y_og - e)))/math.sqrt(len(y_og)))
     

        
    performance = np.array(performance)
    best = np.argmin(performance)

    
    for i in range(length):
        optimum_kp[i] = Kp[min_loss_indices[i], i]
        optimum_kd[i] = Kd[min_loss_indices[i], i]
        min_losses[i] = losses[min_loss_indices[i], i]
        optimum_estimate[i] = estimates[min_loss_indices[i],i]

    

    final_estimate = optimum_estimate
    pct_estimate = optimum_estimate

    # plt.plot(losses[0,:])
    # plt.plot(losses[len(size)-1,:], label='last')
    # plt.plot(min_losses, label = 'opt')
    # plt.plot(np.abs(y_og[0:len(pct_estimate)] - pct_estimate), label =  'pct')
    # plt.legend()
    # plt.show()

    pct_performance = ((np.linalg.norm(y_og[0:len(pct_estimate)] - pct_estimate)))/math.sqrt(len(pct_estimate))
    print(pct_performance)
    
    if performance[best] <= pct_performance:
        print('no pct')
        pct = 0
        optimum_timescales = (0.004*size[best])*np.ones_like(optimum_timescales)
        optimum_kp = (Kp[best,0:length])
        optimum_kd = (Kd[best,0:length])
        final_estimate = estimates[best, 0:length]
    else:
        print('yes pct')


    # final_estimate = optimum_kp*((traing)**-1)*theta + optimum_kd*((traing)**-1)*theta_steps_og

    fit_residuals = y_og[0:len(final_estimate)] - final_estimate

    # plt.plot(steps, label = 'og')
    # plt.plot(pct_estimate, label = 'pct')
    # plt.plot(final_estimate, label = 'final')
    # plt.legend()
    # plt.show()


    return steps_og2, y_og, final_estimate, filter_residuals, fit_residuals, optimum_kp, optimum_kd, optimum_timescales, size, performance, pct_performance, pct
