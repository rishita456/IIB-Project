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
from pykalman import KalmanFilter



def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def calc_func_find_Kp_0_part0(K):

    global time
    global theta
    global position
    global traing
    global P_id
    global trial
    global limit
    Kp_0 = K[0]
    Kd_0 = K[1]
    pg = traing
    # pg = 1

    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(traing) + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(traing) + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(traing) + '/trial' + str(trial) + '/angle.npy')
    time = time[limit[0]:limit[1]]
    position = position[limit[0]:limit[1]]
    theta = theta[limit[0]:limit[1]]
    # detrend position data

    x = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = pos = position-750
    model.fit(x, pos)
    # calculate trend
    trend = model.predict(x)
    detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

    # time = time[1::2]
    # detrended_pos = detrended_pos[1::2]
    # theta = theta[1::2]




    theta_steps = []
    steps = []

    for i in range(len(detrended_pos)-1):
        steps.append(detrended_pos[i+1]-detrended_pos[i])
        theta_steps.append(theta[i+1]-theta[i])
    steps = np.array(steps)


    detrended_pos = np.array(detrended_pos)
    theta_steps = np.array(theta_steps)
    theta_steps = np.insert(theta_steps, 0,theta[0])

    # Filter requirements.
    T = 30         # Sample Period
    fs = 120      # sample rate, Hz
    cutoff = 5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(steps, cutoff, fs, order)



    return np.linalg.norm((-y) + (Kp_0*((pg)**-1)*theta[0:-1] + Kd_0*((pg)**-1)*theta_steps[0:-1]))





# training_gains = [2, 5, 7, 10, 12, 15, 17, 20, 22]
training_gains = [5,]
trial = 1
P_id = 4

fracs = []
# limits = np.arange(0,5000,200)
# limits = [(0, 500), (500,1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000,3500), (3500, 4000), (4000, 4500), (4500, 5000),]
size = np.arange(200,501, 10)
for gain in training_gains:
    traing = gain
    # est = []
    # original = []
    # gamma_est = []
    # Kps = []
    # Kds = []
    losses = np.zeros((len(size),5000))
    Kp = np.zeros((len(size),5000))
    Kd = np.zeros((len(size),5000))
    gamma = np.zeros((len(size),5000))
    
    for j in range(len(size)):
        loss = []
        est = []
        original = []
        steps_unfiltered = []
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

            # Filter requirements.
            T = 30         # Sample Period
            fs = 120      # sample rate, Hz
            cutoff = 5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 2       # sin wave can be approx represented as quadratic
            n = int(T * fs) # total number of samples

            steps_unfiltered.append(steps)

            # Filter the data
            y = butter_lowpass_filter(steps, cutoff, fs, order)

            sequence = np.array(sequence)
            counts, _ = np.histogram(sequence)
            fracs.append(counts[-1]/(counts[0]+counts[-1]))
            step_counts, _ = np.histogram(steps,50, [-1,1], density=True)
            theta_step_counts, _ = np.histogram(theta_steps,50, [-1,1], density=True)
            
            result = minimize(calc_func_find_Kp_0_part0, (1,1))
            print(result.x)
            loss.append(result.fun)
            
            theta_steps = np.insert(theta_steps, 0,theta[0])
            est.append(-(result.x[0])*((gain)**-1)*theta - result.x[0]*((gain)**-1)*theta_steps)
            
            Kps.append(result.x[0])
            Kds.append(result.x[1])
            original.append(y)
        losses[j, 0:len(np.repeat(loss, size[j]))] = np.repeat(loss, size[j])
        Kp[j, 0:len(np.repeat(Kps, size[j]))] = np.repeat(Kps, size[j])
        Kd[j, 0:len(np.repeat(Kds, size[j]))] = np.repeat(Kds, size[j])
 
        original = np.array(original).flatten()
        est = np.array(est).flatten()
        steps_unfiltered = np.array(steps_unfiltered).flatten()


time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')



dt = time[1000] - time[500]


residuals = steps_unfiltered[500:4500] - original[500:4500]

plt.plot(time[500:4500], losses[0, 500:4500])
plt.plot(time[500:4500], losses[10, 500:4500])
plt.plot(time[500:4500], losses[20, 500:4500])
plt.plot(time[500:4500], losses[30, 500:4500])
plt.show()
plt.show()

plt.plot(time[500:4500],steps_unfiltered[500:4500], label = 'raw step data')
plt.plot(time[500:4500],original[500:4500], label = 'filtered step data')
plt.xlabel('time(s)')
plt.ylabel('step size')
plt.legend()
plt.show()


# Fit a normal distribution to the residuals
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

plt.plot(time[500:4500],original[500:4500], label = 'filtered step data')
plt.plot(time[500:4500],est[500:4500], label = 'step estimates using constant learning timescale of 2.5s')
plt.xlabel('time(s)')
plt.ylabel('step size')
plt.legend()
plt.show()



plt.plot(time[500:4500],np.repeat(Kps,500)[500:4500])
plt.title('Kp for a constant learning timescale of 2.5s and gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('Kp')
plt.show()

plt.plot(time[500:4500],np.repeat(Kds,500)[500:4500])
plt.title('Kds for a constant learning timescale of 2.5s and gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('Kd')
plt.show()




min_loss_indices = np.argmin(losses, axis=0)
optimum_kp = np.zeros(5000)
optimum_kd = np.zeros(5000)


for i in range(5000):
    optimum_kp[i] = Kp[min_loss_indices[i], i]
    optimum_kd[i] = Kd[min_loss_indices[i], i]


plt.plot(time[500:4500], losses.min(axis=0)[500:4500])
plt.title('Minimum final losses across timescales for gain = '+ str(gain))
plt.xlabel('time(s)')
plt.ylabel('loss')
plt.show()

plt.plot(time[500:4500], 0.0045*size[np.argmin(losses, axis=0)][500:4500])
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


x = np.reshape(time, (len(time), 1))
model = LinearRegression()
pos = pos = position[0:-2]-750
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

# Filter requirements.
T = 30         # Sample Period
fs = 120      # sample rate, Hz
cutoff = 5     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

# Filter the data
y = butter_lowpass_filter(steps, cutoff, fs, order)


fig, axs = plt.subplots(2)
fig.suptitle('Estimates')

axs[0].plot(time[500:4500],y[500:4500], label = 'filtered step data')
axs[0].plot(time[500:4500], est[500:4500], label = 'old step estimate')
axs[0].legend()
axs[1].plot(time[500:4500], y[500:4500], label = 'filtered step position data')
axs[1].plot(time[500:4500], (optimum_kp[500:4500]*((traing)**-1)*theta[500:4500] + optimum_kd[500:4500]*((traing)**-1)*theta_steps[500:4500]),label = 'new step estimate', color = 'r')
axs[1].legend()

for ax in axs.flat:
    ax.set(xlabel='time(s)', ylabel='step size')


plt.show()
noisy_data = steps_unfiltered[500:4500]
# Define Kalman filter parameters
initial_state_mean = noisy_data[0]  # Initial guess for the state
initial_state_covariance = 1  # Initial guess for the error covariance
transition_matrix = [1]  # Transition matrix for a simple 1D model
observation_matrix = [1]  # Observation matrix for a simple 1D model
observation_covariance = 1  # Variance of the measurement noise
transition_covariance = 1  # Variance of the process (noise in the system)
# Create Kalman filter
kf = KalmanFilter(
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance
)

# Apply Kalman filter smoothing
(filtered_state_means, filtered_state_covariances) = kf.smooth(noisy_data)

# Plot original noisy data and smoothed data
plt.figure(figsize=(10, 6))

plt.plot(noisy_data, label='Noisy Data', color='green', linestyle='-', alpha=0.5)
plt.plot(original[500:4500], label='Butterworth', color='blue', linestyle='-', alpha=0.5)
plt.plot(filtered_state_means, label='Kalman', color='red', linestyle='-', linewidth=2)
plt.title('Kalman Filter Smoothing with pykalman')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
