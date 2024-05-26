import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import config
from scipy.signal import butter,filtfilt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf


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
    pg = K[2]

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
P_id = 5

fracs = []
limits = np.arange(0,5000,200)
# limits = [(0, 500), (500,1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000,3500), (3500, 4000), (4000, 4500), (4500, 5000),]
size = np.arange(200,500, 10)
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
    
    for j in range(len(size)):
        loss = []
        est = []
        original = []
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
            # detrend position data

            x = np.reshape(time, (len(time), 1))
            model = LinearRegression()
            # pos = position[0:-2]-750
            pos = position-750

            print(len(x), len(pos))
            model.fit(x, pos)
            # calculate trend
            trend = model.predict(x)
            detrended_pos = [pos[i]-trend[i] for i in range(0, len(time))]

            # time = time[1::2]
            # detrended_pos = detrended_pos[1::2]
            # theta = theta[1::2]




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
            cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 2       # sin wave can be approx represented as quadratic
            n = int(T * fs) # total number of samples

            # Filter the data, and plot both the original and filtered signals.
            y = butter_lowpass_filter(steps, cutoff, fs, order)

            sequence = np.array(sequence)
            counts, _ = np.histogram(sequence)
            fracs.append(counts[-1]/(counts[0]+counts[-1]))
            step_counts, _ = np.histogram(steps,50, [-1,1], density=True)
            theta_step_counts, _ = np.histogram(theta_steps,50, [-1,1], density=True)
            # plt.plot(np.linspace(-1,1,50), step_counts, label = 'Gain - '+ str(gain))
            # plt.plot(np.linspace(-1,1,50), theta_step_counts, label = 'Gain - '+ str(gain))
            # order = 3
            result = minimize(calc_func_find_Kp_0_part0, (1,1, gain))
            print(result.x)
            loss.append(result.fun)
            # plt.plot(savgol_filter(steps, order + 2, order), label = 'filtered')
            theta_steps = np.insert(theta_steps, 0,theta[0])
            est.append(-(result.x[0])*((result.x[2])**-1)*theta - result.x[0]*((result.x[2])**-1)*theta_steps)
            # est.append(-abs(result.x[0])**theta - result.x[0]*(result.x[2]**-1)*theta_steps)
            gamma_est.append(result.x[2])
            Kps.append(result.x[0])
            Kds.append(result.x[1])
            original.append(y)
        losses[j, 0:len(np.repeat(loss, size[j]))] = np.repeat(loss, size[j])
        Kp[j, 0:len(np.repeat(Kps, size[j]))] = np.repeat(Kps, size[j])
        Kd[j, 0:len(np.repeat(Kds, size[j]))] = np.repeat(Kds, size[j])
        original = np.array(original).flatten()
        est = np.array(est).flatten()

        # plt.plot(original)
        # plt.plot(est, label = 'estimates')
        # plt.legend()
        # plt.show()
        # plt.plot(gamma_est)
        # plt.title('Gamma, gain - '+ str(gain))
        # plt.show()

        # plt.plot(Kps)
        # plt.title('Kp, gain - '+ str(gain))
        # plt.show()

        # plt.plot(Kds)
        # plt.title('Kds, gain - '+ str(gain))
        # plt.show()

        # print(result.x)

    
# plt.xlim(-1,1)
# plt.legend()
# plt.show()
print(len(theta))
print(len(theta_steps))

# plt.plot(training_gains, fracs)
# plt.show()


plt.plot(original)
plt.plot(est, label = 'estimates')
plt.legend()
plt.show()
plt.plot(gamma_est)
plt.title('Gamma, gain - '+ str(gain))
plt.show()

plt.plot(Kps)
plt.title('Kp, gain - '+ str(gain))
plt.show()

plt.plot(Kds)
plt.title('Kds, gain - '+ str(gain))
plt.show()

# print(result.x)

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
min_loss_indices = np.argmin(losses, axis=0)
optimum_kp = np.zeros(5000)
optimum_kd = np.zeros(5000)

for i in range(5000):
    optimum_kp[i] = Kp[min_loss_indices[i], i]
    optimum_kd[i] = Kd[min_loss_indices[i], i]

plt.plot(time[0:5000], losses.min(axis=0))
plt.show()

plt.plot(time[0:5000], size[np.argmin(losses, axis=0)])
plt.show()

plt.plot(time[0:5000], optimum_kp)
plt.show()

plt.plot(time[0:5000], optimum_kd)
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
cutoff = 2     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(steps, cutoff, fs, order)


fig, axs = plt.subplots(2)
fig.suptitle('Estimates')
axs[0].plot(y[500:4500])
axs[0].plot(est[500:4500], label = 'old estimate')
axs[1].plot(y[500:4500])
axs[1].plot((optimum_kp[500:4500]*((traing)**-1)*theta[500:4500] + optimum_kd[500:4500]*((traing)**-1)*theta_steps[500:4500]),label = 'new estimate', color = 'g')
plt.legend()
# plt.plot(y[500:4500])
# plt.plot(est[500:4500], label = 'estimates')
# plt.plot((optimum_kp[500:4500]*((traing)**-1)*theta[500:4500] + optimum_kd[500:4500]*((traing)**-1)*theta_steps[500:4500]))
plt.show()

# plt.hist(sequence)
# plt.show()

# step_counts, _ = np.histogram(steps,50, [-1,1], density=True)
# plt.hist((steps),1000, density=True)
# plt.plot(np.linspace(-1,1,50), step_counts)
