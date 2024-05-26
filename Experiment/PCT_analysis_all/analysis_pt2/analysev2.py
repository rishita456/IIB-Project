import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
# from Experiment.analysis_pt1.analysev4 import calc_func

# Fitting steady state Kp and Kd using the whole trial

def calc_func(time, theta, position, alpha, Kp_0, Kd_0, t_sep, gain1, gain2, c1, c2):
    
    dtheta = np.zeros_like(theta)
    deltat = time[2]-time[0]
    gamma_zero = 1
    gamma_old = gain1
    gamma_true = gain1 * np.ones_like(time)
    gamma_star = gain1 * np.ones_like(time)

    i=0

    for t in time:
        if t > t_sep:
            gamma_true[i] = gain2
            gamma_star[i] = (gamma_true[i]-gamma_old)*(1 - np.exp(-(alpha)*(t-t_sep))) + gamma_old

        else:
            gamma_star[i] = (gamma_true[i] - gamma_zero)*(1 - np.exp(-alpha*(t))) + gamma_zero
            if time[i+1] > t_sep:
                gamma_old = gamma_star[i]
            
        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
   
    Kp = Kp_0 #* (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    

    x = -gamma_star*(theta*Kp + dtheta*Kd) 
    time = time - 0.15
    
    # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
    g_t = 0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time))

    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]

    theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
    theta_est = theta_est * theta_mag_factor

    loss = np.linalg.norm(gamma_true*theta - gamma_star*theta_est)


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

    plt.plot(time, theta, label = 'theta')
    plt.plot(time, theta_est, label = 'estimate')
    plt.title('Gain = '+str(training_gain))
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.legend()
    plt.show()

    return loss, gamma_star, theta, dtheta, theta_est, gamma_true, x, time, g_t, theta_mag_factor

def calc_func_find_Kp_0(K):

    global training_gain
    global Kp_0, Kd_0
    global settle_time, mse
    global theta_mag_factor
    global P_id
    global trial

    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')
    deltat = time[6]-time[4]
    alpha = 1
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

    if settle_time == 0:
        settle_time = 29.991
    settle_index = np.where(time>settle_time)

    
    dtheta = np.zeros_like(theta)
    

    gamma_old = gain1
    gamma_true = gain1 * np.ones_like(time)
    gamma_star = gain1 * np.ones_like(time)

    i=0

    for t in time:
        if t > t_sep:
            gamma_true[i] = gain2
            gamma_star[i] = (gamma_true[i]-gamma_old)*(1 - np.exp(-alpha*(t-t_sep))) + gamma_old


        else:
            gamma_star[i] = (gamma_true[i])*(1 - np.exp(-alpha*(t)))
            if time[i+1] > t_sep:
                gamma_old = gamma_star[i]
            

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
    gamma_true = np.ones_like(time)

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    

    x = -gamma_star*(theta*Kp + dtheta*Kd) # why -
    time = time - 0.15
    
    # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
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

    



    return loss

P_id = 3
trial = 2

training_gains = ['2','5', '7','10','12','15','17', '20','22' ]
Kps = []
Kds = []
settle_times = []
mses = []
mags = []

# time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '2' + '/trial' + str(trial) + '/time.npy')
# position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '2' + '/trial' + str(trial) + '/position.npy')
# theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '2' + '/trial' + str(trial) + '/angle.npy')

# print(time[0:20])

# plt.plot(time, theta)
# plt.show()

for gain in training_gains:
    training_gain = gain
    result = minimize(calc_func_find_Kp_0, (0.1,0.1))
    Kp_0 = result.x[0]
    Kd_0 = result.x[1]
    Kps.append(Kp_0)
    Kds.append(Kd_0)
    mags.append(theta_mag_factor)
    settle_times.append(settle_time)
    print(settle_time)
    mses.append(mse)
    time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
    position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
    theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')
    index = 0
    for t in time[0:-2]:
        slice = theta[index:-1]
        if np.all(abs((slice)) < 5):
            mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            mse = np.sqrt(mse)
            settle_time = t

            break
        index = index + 1

    if settle_time == 0:
        settle_time = 29.991
    
    # if settle_time != 29.991:
    #     settle_index = np.where(time>settle_time)
    #     time = time[int(settle_index[0][0]):-1]
    #     theta = theta[int(settle_index[0][0]):-1]
    #     position = position[int(settle_index[0][0]):-1]
    # calc_func(time, theta, position, 1, Kp_0, Kd_0, 0, int(gain), int(gain), 1, 1)


plt.scatter(training_gains, Kps, label = 'Kp0')
plt.scatter(training_gains, Kds, label = 'Kd0')


plt.xlabel('visual feedback gains')
plt.ylabel('Kp0s and Kd0s')
plt.title('Kps and Kds fitted using full trial with constant feedback gain')
plt.legend()
plt.show()

plt.scatter(settle_times, Kps, c='blue', label = 'Kp')
plt.scatter(settle_times, Kds, c='red', label = 'Kd')
plt.xlabel('settling times')
plt.legend()
plt.show()
# settle_time = 0
# index = 0
# mse = 0


# for t in time:
#     slice = theta[index:-1]
#     if np.all(abs((slice)) < 5 ):
#         mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
#         mse = np.sqrt(mse)
#         settle_time = t
#         break
#     index = index + 1


# settle_index = np.where(time>settle_time)

# steady_state_calibration_time = time[settle_index[0][0]:-1]
# steady_state_calibration_theta = theta[settle_index[0][0]:-1]
# steady_state_calibration_position = position[settle_index[0][0]:-1]

# training_gain = '5'

# result = minimize(calc_func_find_Kp_0, (0.2,0.2))

# print(result.x)
# plt.plot(steady_state_calibration_time, steady_state_calibration_theta)
# plt.show()

# plt.plot(time, theta)
# plt.show()