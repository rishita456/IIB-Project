import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from analyse_funcs import calc_func
# from Experiment.analysis_pt1.analysev4 import calc_func

# Fitting using detrended position data

def calc_func_find_Kp_0_full(K):

    global training_gain
    global Kp_0, Kd_0
    global settle_time, mse
    global theta_mag_factor
    global pos_mag_factor
    global P_id
    global trial

    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')
    deltat = time[6]-time[4]
    alpha = 10000000
    t_sep = 0
    gain1 = int(training_gain)
    gain2 = int(training_gain)
    Kp_0 = K[0]
    Kd_0 = K[1]
    c1 = 1
    c2 = 1

    settle_time = 0
    index = 0
    mse = 0


    for t in time[0:-2]:
        slice = theta[index:-1]
        if np.all(abs((slice)) < 5):
            mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            mse = np.sqrt(mse)
            settle_time = t
            break
        index = index + 1

    # if settle_time == 0:
    #     settle_time = 29.991
    #     loss = np.nan
    #     theta_mag_factor = np.nan
    #     Kp_0 = np.nan
    #     Kd_0 = np.nan

    # settle_index = np.where(time>settle_time)

    
    dtheta = np.zeros_like(theta)
    

    gamma_old = gain1
    gamma_true = gain1 * np.ones_like(time)
    gamma_star = gain1 * np.ones_like(time)

    i=0

    for t in time:
        # if t > t_sep:
        #     gamma_true[i] = gain2
        #     gamma_star[i] = (gamma_true[i]-gamma_old)*(1 - np.exp(-alpha*(t-t_sep))) + gamma_old


        # else:
        #     gamma_star[i] = (gamma_true[i])*(1 - np.exp(-alpha*(t)))
        #     if time[i+1] > t_sep:
        #         gamma_old = gamma_star[i]
            

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    
    x_t = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position[0:-2]-750
    model.fit(x_t, pos)
    # calculate trend
    trend = model.predict(x_t)
    detrended = [pos[i]-trend[i] for i in range(0, len(time))]
    x = -gamma_star*(theta*Kp + dtheta*Kd) # why -
    pos_mag_factor = np.std(detrended[15:-1])/np.std(x[15:-1])
    x = x * pos_mag_factor
    time = time - 0.15
    
    # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]

    theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
    theta_est = theta_est * theta_mag_factor

    if pearsonr(theta_est, theta)[0] < -0.5:

        pos_mag_factor = -pos_mag_factor
        x = -x
        # time = time - 0.15

        g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50

        theta_est = np.convolve(x, g_t, 'full')

        theta_est = theta_est[len(time)-2:-1]
        theta = theta[0:len(theta_est)]

        theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
        theta_est = theta_est * theta_mag_factor


    for th in theta:
        if th > 0:
            theta[np.where(theta==th)] = th%360
        if th < 0:
            theta[np.where(theta==th)] = th%(-360)

    for th in theta_est:
        if th > 0:
            theta_est[np.where(theta_est==th)] = th%360
        if th < 0:
            theta_est[np.where(theta_est==th)] = th%(-360)

    loss = np.linalg.norm(gamma_true*theta - gamma_star*theta_est)

    if settle_time == 0:
        settle_time = 29.991
        # loss = np.nan
        # theta_mag_factor = np.nan
        # Kp_0 = np.nan
        # Kd_0 = np.nan
        # pos_mag_factor = np.nan



    return loss

P_id = 7
trial = 1




training_gain = '7'

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')
position_diff = np.zeros_like(position[0:-1])

x = np.reshape(time, (len(time), 1))
model = LinearRegression()
pos = position[0:-2]-750
model.fit(x, pos)
# calculate trend
trend = model.predict(x)
detrended = [pos[i]-trend[i] for i in range(0, len(time))]

# plot trend
plt.plot(position-750)
plt.plot(trend)
plt.show()

plt.plot(detrended)
plt.show()
# for t in range(len(time)-1):

#     position_diff[t+1] = position[t+1] - position[t]


# plt.plot(position_diff)
# plt.plot(position - 750)
# plt.show()

training_gains =['2','5', '7','10','12','15','17', '20','22' ]
Kps_full = []
Kps_ss = []
Kds_full = []
Kds_ss = []
mags_full = []
posmags_full = []
mags_ss = []
posmags_ss = []
loss_full = []
loss_ss = []
P_id = 3
trial = 2

for gain in training_gains:
    training_gain = gain
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')
    result = minimize(calc_func_find_Kp_0_full, (1,1))
    Kp_0 = result.x[0]
    Kd_0 = result.x[1]
    Kps_full.append(Kp_0)
    Kds_full.append(Kd_0)
    loss_full.append(result.fun)
    mags_full.append(theta_mag_factor)
    posmags_full.append(pos_mag_factor)
    calc_func(training_gain, time, theta, position, 10000000, Kp_0, Kd_0, 0, int(training_gain), int(training_gain), 1, 1)


plt.plot(training_gains, Kps_full, label = 'full')

plt.title('Kps')
plt.legend()
plt.show()

plt.plot(training_gains, Kds_full, label = 'full')

plt.title('Kds')
plt.legend()
plt.show()

plt.plot(training_gains, mags_full, label = ' theta mag full')
plt.plot(training_gains, posmags_full, label = 'pos mag full')

plt.legend()
plt.show()

plt.plot(training_gains, loss_full, label = 'full')

plt.title('loss')
plt.legend()
plt.show()