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
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def calc_func_find_Kp_0_part0(K):

   
    global theta
    global theta_steps
    global position
    global traing
    global y
    Kp_0 = K[0]
    Kd_0 = K[1]
    pg = traing

    return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))/len(y)


# Filter requirements - to try later - find optimum cutoff frequency and order for best estimate and best reconstruction

fs = 120      # sample rate, Hz
cutoff = 5      # desired cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # high order due to slow roll off


# training_gains = [2, 5, 7, 10, 12, 15, 17, 20, 22]
training_gains = [5,]
trial = 1
P_id = 3

size = np.arange(200,501, 2)
for gain in training_gains:
    traing = gain

    losses = np.zeros((len(size),5000)) #total time samples = 500
    Kp = np.zeros((len(size),5000))
    Kd = np.zeros((len(size),5000))
    gamma = np.zeros((len(size),5000))
    
    for j in range(len(size)):
        loss = []
        est = []
        gamma_est = []
        Kps = []
        Kds = []
        limits = np.arange(0,5001,size[j])
        for i in range(len(limits)-1):
            
            limit = (limits[i], limits[i+1])
            time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
            position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
            theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
            time = time[limit[0]:limit[1]]
            position = position[limit[0]:limit[1]]
            theta = theta[limit[0]:limit[1]]
    
            x = np.reshape(time, (len(time), 1))
            model = LinearRegression()
            # pos = position[0:-2]-750
            pos = position-750

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

            # Filter the data
            y = butter_lowpass_filter(steps, cutoff, fs, order)
            theta_steps = np.array(theta_steps)
            theta_steps = np.insert(theta_steps, 0,theta[0])
    

            result = minimize(calc_func_find_Kp_0_part0, (1,1))
            print(result.x)
            loss.append(result.fun)
            
            est.append((result.x[0])*((gain)**-1)*theta + result.x[0]*((gain)**-1)*theta_steps)
            
            Kps.append(result.x[0])
            Kds.append(result.x[1])

        losses[j, 0:len(np.repeat(loss, size[j]))] = np.repeat(loss, size[j])
        Kp[j, 0:len(np.repeat(Kps, size[j]))] = np.repeat(Kps, size[j])
        Kd[j, 0:len(np.repeat(Kds, size[j]))] = np.repeat(Kds, size[j])
        est = np.array(est).flatten()
        


time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')



dt = time[502] - time[500]

# plt.plot(time[500:4500], losses[0, 500:4500])
# plt.plot(time[500:4500], losses[10, 500:4500])
# plt.plot(time[500:4500], losses[20, 500:4500])
# plt.plot(time[500:4500], losses[30, 500:4500])
# plt.show()
# plt.show()


x = np.reshape(time, (len(time), 1))
model = LinearRegression()
pos = position[0:-2]-750
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

# Filter the data
y = butter_lowpass_filter(steps, cutoff, fs, order)

plt.plot(time[500:4500],steps[500:4500], label = 'raw step data')
plt.plot(time[500:4500],y[500:4500], label = 'filtered step data')
plt.xlabel('time(s)')
plt.ylabel('step size')
plt.legend()
plt.show()

# fft 
fft_result = np.fft.fft(steps)
freq = np.fft.fftfreq(len(time), time[2] - time[0])  # Frequency bins
fft_result2 = np.fft.fft(y)
freq2 = np.fft.fftfreq(len(time), time[2] - time[0])  # Frequency bins

# Plot FFT
plt.figure(figsize=(10, 6))
plt.plot(freq[0:-1], np.abs(fft_result), label = 'fft of step position data')
plt.plot(freq2[0:-1], np.abs(fft_result2), label = 'fft of filtered step position data')
plt.title('FFT of Time Series')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# plt.hist(y)
# plt.show()

residuals = steps[500:4500] - y[500:4500]

plt.plot(residuals)
plt.show()

# Fit a normal distribution to the residuals - note not always normal
(mu, sigma) = norm.fit(residuals)

# Plot histogram of the data
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')

# Plot the probability density function (PDF) of the fitted normal distribution
x = np.linspace(min(residuals), max(residuals), 1000)  # Range of x values
pdf = norm.pdf(x, mu, sigma)  # PDF of the normal distribution
plt.plot(x, pdf, color='red', linewidth=2, label='Normal Distribution')
plt.title('Histogram and Fitted Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

noise = np.random.normal(mu, sigma, len(time))
plt.plot(time[500:4500],noise[500:4500] + y[500:4500], label = 'reconstructed from filtered step data')
plt.plot(time[500:4500],steps[500:4500], label = 'raw step data')
plt.legend()
plt.show()

# plt.plot(time[500:4500],y[500:4500], label = 'filtered step data')
# plt.plot(time[500:4500],est[500:4500], label = 'step estimates using constant learning timescale of 2.5s')
# plt.xlabel('time(s)')
# plt.ylabel('step size')
# plt.legend()
# plt.show()

# plt.plot(time[500:4500],np.repeat(Kps,500)[500:4500])
# plt.title('Kp for a constant learning timescale of 2.5s and gain = '+ str(gain))
# plt.xlabel('time(s)')
# plt.ylabel('Kp')
# plt.show()

# plt.plot(time[500:4500],np.repeat(Kds,500)[500:4500])
# plt.title('Kds for a constant learning timescale of 2.5s and gain = '+ str(gain))
# plt.xlabel('time(s)')
# plt.ylabel('Kd')
# plt.show()


performance = []
for i in range(len(size)):
    e = Kp[i, 500:4500]*((traing)**-1)*theta[500:4500] + Kd[i, 500:4500]*((traing)**-1)*theta_steps[500:4500]
    performance.append((np.linalg.norm(y[500:4500] - e))/len(y[500:4500]))
    print((np.linalg.norm(y[500:4500] - e))/len(y[500:4500]))

    
performance = np.array(performance)

plt.plot(size, performance)
best = np.argmin(performance)

e = Kp[best, 500:4500]*((traing)**-1)*theta[500:4500] + Kd[best, 500:4500]*((traing)**-1)*theta_steps[500:4500]


min_loss_indices = np.argmin(losses, axis=0)
optimum_kp = np.zeros(5000)
optimum_kd = np.zeros(5000)
opt_times = 0.0045*size[np.argmin(losses, axis=0)]

for i in range(5000):
    optimum_kp[i] = Kp[min_loss_indices[i], i]
    optimum_kd[i] = Kd[min_loss_indices[i], i]

final_estimate = optimum_kp[500:4500]*((traing)**-1)*theta[500:4500] + optimum_kd[500:4500]*((traing)**-1)*theta_steps[500:4500]


plt.axhline(np.linalg.norm(y[500:4500] - final_estimate)/len(y[500:4500]))
plt.show()


if np.linalg.norm(y[500:4500] - e) < np.linalg.norm(y[500:4500] - final_estimate):

    opt_times = (0.0045*size[best])*np.ones_like(opt_times[500:4500])
    optimum_kp = (Kp[best,:])
    optimum_kd = (Kd[best,:])


plt.plot(time[500:4500], losses.min(axis=0)[500:4500])
plt.title('Minimum final losses across timescales for gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('loss')
plt.show()

plt.plot(time[500:4500], 0.0045*size[np.argmin(losses, axis=0)][500:4500])
plt.plot(time[500:4500], opt_times)
plt.title('optimum learning timescales for gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('learning timescales(s)')
plt.show()

plt.plot(time[500:4500], optimum_kp[500:4500])
plt.title('Optimum Kp values for gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('Kp')
plt.show()

plt.plot(time[500:4500], optimum_kd[500:4500])
plt.title('Optimum Kd values for gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('Kd')
plt.show()



fig, axs = plt.subplots(2)
fig.suptitle('Estimates')


axs[0].plot(time[500:4500],y[500:4500], label = 'filtered step data')
axs[0].plot(time[500:4500], e, label = 'old step estimate')
axs[0].legend()
axs[1].plot(time[500:4500], y[500:4500], label = 'filtered step position data')
axs[1].plot(time[500:4500], final_estimate,label = 'new step estimate', color = 'r')
axs[1].legend()

for ax in axs.flat:
    ax.set(xlabel='time(s)', ylabel='step size')


plt.show()

residuals = y[500:4500] - final_estimate
# Fit a normal distribution to the residuals - note not always normal
(mu, sigma) = norm.fit(residuals)

# Plot histogram of the data
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')

# Plot the probability density function (PDF) of the fitted normal distribution
x = np.linspace(min(residuals), max(residuals), 1000)  # Range of x values
pdf = norm.pdf(x, mu, sigma)  # PDF of the normal distribution
plt.plot(x, pdf, color='red', linewidth=2, label='Normal Distribution')
plt.title('Histogram and Fitted Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

noise = np.random.normal(mu, sigma, len(final_estimate))
y_new = y[500:4500]
accurate_points = []
for i in range(len(final_estimate)):
    if abs(final_estimate[i] - y_new[i]) < 0.1 * abs(y_new[i]):
        accurate_points.append(final_estimate[i])
    else:
        accurate_points.append(None)



plt.plot(time[500:4500],y[500:4500], label = 'filtered step data')
plt.plot(time[500:4500], final_estimate + noise, label = 'step estimate with  noise', alpha = 0.5)
plt.plot(time[500:4500], accurate_points,label = 'step estimate', marker = '.')
plt.legend()
plt.show()





# performance = []
# for i in range(len(size)):
#     e = Kp[i, 500:4500]*((traing)**-1)*theta[500:4500] + Kd[i, 500:4500]*((traing)**-1)*theta_steps[500:4500]
#     performance.append(np.linalg.norm(y[500:4500] - e))

    


# performance = np.array(performance)
# plt.plot(size, performance)
# plt.axhline(np.linalg.norm(y[500:4500] - final_estimate))
# plt.show()

