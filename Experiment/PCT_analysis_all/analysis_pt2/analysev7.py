import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from analyse_funcs import calc_func
import config
# from Experiment.analysis_pt1.analysev4 import calc_func

# Analysing individual section of 2 disturbance trials
def calc_func_find_Kp_0_part0(K):

    global training_gain
    global Kp_0, Kd_0
    global settle_time, mse
    global theta_mag_factor
    global pos_mag_factor
    global P_id
    global trial
    global lf

    global loss_0
    
    try: 

        time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
        position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
        theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')

    except FileNotFoundError as e:
        print(e, training_gain)
        
    deltat = 0.008
    alpha = 10000000
    t_sep = 0
    # gain1 = int(training_gain)
    # gain2 = int(training_gain)
    gain1 = 1
    gain2 = 1
    Kp_0 = K[0]
    Kd_0 = K[1]
    c1 = 1
    c2 = 1

    settle_time = 0
    index = 0
    mse = 0
    calibration_index =  np.where(time < 30)
    time = time[0:calibration_index[-1][-1]]
    theta = theta[0:calibration_index[-1][-1]]
    position = position[0:calibration_index[-1][-1]]


    for t in time[0:-2]:
        slice = theta[index:-1]
        mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
        mse = np.sqrt(mse)
        if np.all(abs((slice)) < 5):
            # mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            # mse = np.sqrt(mse)
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
    gamma_star = lf * gain1 * np.ones_like(time)

    i=0

    for t in time:

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    
    x_t = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position-750
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

    # print(loss)
    loss_0.append(loss)

    return loss


def calc_func_find_Kp_0_part1(K):

    global training_gain
    global Kp_0, Kd_0
    global settle_time, mse
    global theta_mag_factor
    global pos_mag_factor
    global P_id
    global trial
    global lf

    global loss_1
    
    try: 

        time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
        position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
        theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')

    except FileNotFoundError as e:
        print(e, training_gain)
        
    deltat = 0.008
    alpha = 10000000
    t_sep = 0
    # gain1 = int(training_gain)
    # gain2 = int(training_gain)
    gain1 = training_gain
    gain2 = training_gain
    Kp_0 = K[0]
    Kd_0 = K[1]
    c1 = 1
    c2 = 1

    settle_time = 0
    index = 0
    mse = 0
    calibration_index1 =  np.where(time < 30)
    calibration_index2 =  np.where(time < 40)
    time = time[calibration_index1[-1][-1]:calibration_index2[-1][-1]]
    theta = theta[calibration_index1[-1][-1]:calibration_index2[-1][-1]]
    position = position[calibration_index1[-1][-1]:calibration_index2[-1][-1]]


    for t in time[0:-2]:
        slice = theta[index:-1]
        mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
        mse = np.sqrt(mse)
        if np.all(abs((slice)) < 5):
            # mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            # mse = np.sqrt(mse)
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
    gamma_star = lf * gain1 * np.ones_like(time)

    i=0
    

    for t in time:

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    
    x_t = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    pos = position-750
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

    # print(loss)
    loss_1.append(loss)
    return loss



def calc_func_find_Kp_0_part2(K):

    global training_gain
    global Kp_0, Kd_0
    global settle_time, mse
    global theta_mag_factor
    global pos_mag_factor
    global P_id
    global trial
    global lf

    global loss_2
    try: 

        time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
        position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
        theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')

    except FileNotFoundError as e:
        print(e, training_gain)
        
    deltat = 0.008
    alpha = 10000000
    t_sep = 0
    # gain1 = int(training_gain)
    # gain2 = int(training_gain)
    gain1 = training_gain
    gain2 = training_gain
    Kp_0 = K[0]
    Kd_0 = K[1]
    c1 = 1
    c2 = 1

    settle_time = 0
    index = 0
    mse = 0
    calibration_index =  np.where(time < 40)
    time = time[calibration_index[-1][-1]:-1]
    theta = theta[calibration_index[-1][-1]:-1]
    position = position[calibration_index[-1][-1]:-1]


    for t in time[0:-2]:
        slice = theta[index:-1]
        mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
        mse = np.sqrt(mse)
        if np.all(abs((slice)) < 5):
            # mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            # mse = np.sqrt(mse)
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
    gamma_star = lf * gain1 * np.ones_like(time)

    i=0

    for t in time:

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    
    x_t = np.reshape(time, (len(time), 1))
    model = LinearRegression()
    # pos = position-750
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

    # print(loss)
    loss_2.append(loss)
    return loss



P_id = 2
trial = 1
lf = 1


loss_0 = []
loss_1 = []
loss_2 = []

training_gains1 = config.TRIAL_GAINS_2[2][:]
training_gain = training_gains1[0]
result = minimize(calc_func_find_Kp_0_part0, (1,1))
# lf = pos_mag_factor

Kps_0 = []
Kps_1 = []
Kps_2 = []
Kds_0 = []
Kds_1 = []
Kds_2 = []
mags_0 = []
mags_1 = []
mags_2 = []
posmags_0 = []
posmags_1 = []
posmags_2 = []

mses_0 = []
mses_1 = []
mses_2 = []

trial = 1
index = 1
for gain in training_gains1:
    slice = training_gains1[0:index]
    training_gain = gain
    print(slice)
    trial = slice.count(gain)
    print(trial)
    try: 

        time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
        position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
        theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')

    except FileNotFoundError as e:
        print(e, gain)
        Kps_0.append(np.nan)
        Kds_0.append(np.nan)
        loss_0.append(np.nan)
        mags_0.append(np.nan)
        posmags_0.append(np.nan)
        mses_0.append(np.nan)
        Kps_1.append(np.nan)
        Kds_1.append(np.nan)
        loss_1.append(np.nan)
        mags_1.append(np.nan)
        posmags_1.append(np.nan)
        mses_1.append(np.nan)
        Kps_2.append(np.nan)
        Kds_2.append(np.nan)
        loss_2.append(np.nan)
        mags_2.append(np.nan)
        posmags_2.append(np.nan)
        mses_2.append(np.nan)
        index = index + 1
        continue
    if len(time) == 0:
        index = index + 1
        Kps_0.append(np.nan)
        Kds_0.append(np.nan)
        loss_0.append(np.nan)
        mags_0.append(np.nan)
        posmags_0.append(np.nan)
        mses_0.append(np.nan)
        Kps_1.append(np.nan)
        Kds_1.append(np.nan)
        loss_1.append(np.nan)
        mags_1.append(np.nan)
        posmags_1.append(np.nan)
        mses_1.append(np.nan)
        Kps_2.append(np.nan)
        Kds_2.append(np.nan)
        loss_2.append(np.nan)
        mags_2.append(np.nan)
        posmags_2.append(np.nan)
        mses_2.append(np.nan)
        continue
    print(time[0:10])
    result = minimize(calc_func_find_Kp_0_part0, (1,1))
    Kp_0 = result.x[0]
    Kd_0 = result.x[1]
    Kps_0.append(Kp_0)
    Kds_0.append(Kd_0)
    loss_0.append(result.fun)
    mags_0.append(theta_mag_factor)
    posmags_0.append(pos_mag_factor)
    mses_0.append(mse)
    # lf = pos_mag_factor

    result2 = minimize(calc_func_find_Kp_0_part1, (result.x[0],result.x[1]))
    Kp_0 = result2.x[0]
    Kd_0 = result2.x[1]
    Kps_1.append(Kp_0)
    Kds_1.append(Kd_0)
    loss_1.append(result2.fun)
    mags_1.append(theta_mag_factor)
    posmags_1.append(pos_mag_factor)
    mses_1.append(mse)
    # lf = pos_mag_factor

    result3 = minimize(calc_func_find_Kp_0_part2, (result2.x[0],result2.x[1]))
    Kp_0 = result3.x[0]
    Kd_0 = result3.x[1]
    Kps_2.append(Kp_0)
    Kds_2.append(Kd_0)
    loss_2.append(result3.fun)
    mags_2.append(theta_mag_factor)
    posmags_2.append(pos_mag_factor)
    mses_2.append(mse)
    # lf = pos_mag_factor

    calibration_index =  np.where(time < 40)
    time = time[calibration_index[-1][-1]:-1]
    theta = theta[calibration_index[-1][-1]:-1]
    position = position[calibration_index[-1][-1]:-1]
    # calc_func(training_gain*1.5, time, theta, position, 10000000, Kp_0, Kd_0, 0, int(training_gain), int(training_gain), 1, 1)
    index = index + 1

print(len(Kps_1))

#np.arange(len(training_gains1))

plt.scatter(np.arange(len(training_gains1)), Kps_0, label = '0-30s', alpha=0.85, edgecolors='black')
plt.scatter(np.arange(len(training_gains1)), Kps_1, label = '30-40s', alpha=0.85, edgecolors='black')
plt.scatter(np.arange(len(training_gains1)), Kps_2, label = '40-55s', alpha=0.85, edgecolors='black')
plt.title('Kps')
plt.xlabel('trials')
plt.legend()
plt.show()

plt.scatter(np.arange(len(training_gains1)), Kds_0, label = '0-30s', alpha=0.85, edgecolors='black')
plt.scatter(np.arange(len(training_gains1)), Kds_1, label = '30-40s', alpha=0.85, edgecolors='black')
plt.scatter(np.arange(len(training_gains1)), Kds_2, label = '40-55s', alpha=0.85, edgecolors='black')
plt.xlabel('trials')
plt.title('Kds')
plt.legend()
plt.show()

# plt.scatter(Kps_0, mags_0, label = ' theta mag 0-30s', alpha=0.3, edgecolors='black', c = 'red')
plt.scatter(Kps_0, posmags_0, label = 'pos mag 0-30s', alpha=0.3, edgecolors='black', c = 'blue')
# plt.scatter(Kps_1, mags_1, label = ' theta mag 30-40s', alpha=0.6, edgecolors='black', c = 'red')
plt.scatter(Kps_1, posmags_1, label = 'pos mag 30-40s', alpha=0.6, edgecolors='black', c = 'blue')
# plt.scatter(Kps_2, mags_2, label = ' theta mag 40-55s', alpha=0.9, edgecolors='black', c = 'red')
plt.scatter(Kps_2, posmags_2, label = 'pos mag 40-55s', alpha=0.9, edgecolors='black', c = 'blue')
plt.xlabel('Kp0s')
plt.legend()
plt.show()

# plt.plot(np.arange(len(training_gains1)), loss_0, label = '0-30s')
# plt.plot(np.arange(len(training_gains1)), loss_1, label = '30-40s')
# plt.plot(np.arange(len(training_gains1)), loss_2, label = '40-55s')
# plt.xlabel('trials')

# plt.title('loss')
# plt.legend()
# plt.show()

plt.plot(loss_0)
plt.show()

plt.plot(loss_1)
plt.show()

plt.plot(loss_2)
plt.show()