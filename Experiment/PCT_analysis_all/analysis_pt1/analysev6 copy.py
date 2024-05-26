import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from analysev4 import calc_func
from analysev8 import split_data

def calc_func_find_Kp_0(K):

    global training_gain2
    global Kp_0, Kd_0

    time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/time.npy")
    position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/position.npy")
    theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/angle.npy")
    # raw_hand_position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep12/run1/raw_position.npy")
    alpha = 10000000
    t_sep = 0
    gain1 = int(training_gains[0])
    gain2 = int(training_gains[0])
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


    for t in time:
        slice = theta[index:-1]
        if np.all(abs((slice)) < 5 ):
            mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
            mse = np.sqrt(mse)
            settle_time = t
            break
        index = index + 1


    settle_index = np.where(time>settle_time)
    steady_state_calibration_time = time[settle_index[0][0]:calibration_index[-1][-1]]
    steady_state_calibration_theta = theta[settle_index[0][0]:calibration_index[-1][-1]]
    steady_state_calibration_position = position[settle_index[0][0]:calibration_index[-1][-1]]

    time = steady_state_calibration_time
    theta = steady_state_calibration_theta
    position = steady_state_calibration_position
    
    dtheta = np.zeros_like(theta)
    deltat = time[2]-time[0]
  
    gamma_old = gain1
    gamma_true = gain1 * np.ones_like(time)
    gamma_star = gain1 * np.ones_like(time)
    del_gamma_star = np.ones_like(time)

    i=0

    for t in time:
        if t > t_sep:
            gamma_true[i] = gain2
            gamma_star[i] = (gamma_true[i]-gamma_old)*(1 - np.exp(-alpha*(t-t_sep))) + gamma_old
            del_gamma_star[i] = (gamma_true[i] - gamma_old)*(t-t_sep)*np.exp(-alpha*(t-t_sep))


        else:
            gamma_star[i] = (gamma_true[i])*(1 - np.exp(-alpha*(t)))
            del_gamma_star[i] = gamma_true[i]*(t)*np.exp(-alpha*(t))
            if time[i+1] > t_sep:
                gamma_old = gamma_star[i]
               

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
    gamma_true = np.ones_like(time)

    Kp = Kp_0 #* (gamma_star**(-c1))
    Kd = Kd_0 #* (gamma_star**(-c2))
    

    x = -gamma_star*(theta*Kp + dtheta*Kd) # why -
    time = time - 0.15
    
    g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time))*(deltat**2) #why multiply by -deltatsquared
    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]
    loss = np.linalg.norm(theta - theta_est)

    return loss

def calc_func_find_alpha(x):

    global training_gains
    global Kp_0, Kd_0
    global tsep

   
    time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/time.npy")
    position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/position.npy")
    theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/" + training_gains[1] + "_to_" + training_gains[2] + "/angle.npy")
    t_sep = 30
    gain1 = 1
    gain2 = training_gain2
    
    # Kp_0 = 0.530406506 
    # Kd_0 = -1.611970449
    alpha = x[0]
    c1 = x[1]
    c2 = x[2]
    # c1 = 1
    # c2 = 1

    # result = minimize(calc_func_find_Kp_0, (0.1,0.1))
    # Kp_0 = result.x[0]
    # Kd_0 = result.x[1]

 
    calibration_index =  np.where(time > 30)
    calibration_time = time[calibration_index[0][0]]
    t_sep = calibration_time
    time = time[calibration_index[0][0]:-1]
    theta = theta[calibration_index[0][0]:-1]
    position = position[calibration_index[0][0]:-1]

    disturbance_index = np.where(time>59)
    time = time[0:disturbance_index[0][0]]
    theta = theta[0:disturbance_index[0][0]]
    position = position[0:disturbance_index[0][0]]

    dtheta = np.zeros_like(theta)
    deltat = time[2]-time[0]

    gamma_zero = 1
  
    gamma_old = gain1
    gamma_true = gain1 * np.ones_like(time)
    gamma_star = gain1 * np.ones_like(time)
    
    del_gamma_star = np.ones_like(time)

    i=0

    for t in time:
        if t > t_sep:
            gamma_true[i] = gain2
            gamma_star[i] = (gamma_true[i]-gamma_old)*(1 - np.exp(-(alpha)*(t-t_sep))) + gamma_old
            del_gamma_star[i] = (gamma_true[i] - gamma_old)*(t-t_sep)*np.exp(-(alpha)*(t-t_sep))


        else:
            gamma_star[i] = (gamma_true[i] - gamma_zero)*(1 - np.exp(-alpha*(t))) + gamma_zero
            del_gamma_star[i] = gamma_true[i]*(t)*np.exp(-alpha*(t))
            if time[i+1] > t_sep:
                gamma_old = gamma_star[i]
               

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
 

    Kp = Kp_0 * (gamma_star**(-c1))
    Kd = Kd_0 * (gamma_star**(-c2))
    

    x = -gamma_star*(theta*Kp + dtheta*Kd) # why -
    time = time - 0.15

    
    
    g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time))*(deltat**2) #why multiply by -deltatsquared
    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]
    # loss = np.linalg.norm(theta - theta_est)
    loss = np.linalg.norm(gamma_true*theta - gamma_star*theta_est)

    return loss


def train(training_gain):
    global training_gain2
    training_gain2 = training_gain
    test_gain2 = training_gain

    time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/time.npy")
    position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/position.npy")
    theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/angle.npy")
    raw_hand_position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/raw_position.npy")



    result = minimize(calc_func_find_Kp_0, (0.1,0.1))
    Kp_0 = result.x[0]
    Kd_0 = result.x[1]

    print('Kp0 - ', Kp_0)
    print('Kd0 - ', Kd_0)


    calibration_index =  np.where(time > 30)
    calibration_time = time[calibration_index[0][0]]
    time = time[calibration_index[0][0]:-1]
    theta = theta[calibration_index[0][0]:-1]
    position = position[calibration_index[0][0]:-1]
    raw_hand_position = np.repeat(raw_hand_position,2)
    raw_hand_position = raw_hand_position[calibration_index[0][0]:-1]

    disturbance_index = np.where(time>59)
    time = time[0:disturbance_index[0][0]]
    theta = theta[0:disturbance_index[0][0]]
    position = position[0:disturbance_index[0][0]]


    bnds = ((0,None),(0,None), (0,None))

    result2 = minimize(calc_func_find_alpha,(0.5,1, 1), bounds=bnds)
    alpha = result2.x[0]
    c1 = result2.x[1]
    c2 = result2.x[2]
    print('alpha - ', alpha)
    print('c1 - ', c1)
    print('c2 - ', c2)
    print('loss function - ',result.fun)



    loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, dell  = calc_func(time, theta, position, alpha, Kp_0, Kd_0, calibration_time, 1, int(test_gain2), c1, c2)


    #why are the first 15 indices wrong
    #why is theta_est and x_t of by different orders of magnitude

    # theta_est = gamma_star*theta_est
    # theta = gamma_true*theta
    theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
    plt.plot(time[15:-1], theta[15:-1])
    plt.plot(time[15:-1], theta_mag_factor*theta_est[15:-1])
    # plt.ylim(-8,8)
    plt.show()


    position = position - np.mean(position)
    x_t = x_t
    x_t = x_t - np.mean(x_t[15:-1])
    pos_mag_factor = np.std(position[15:-1])/np.std(x_t[15:-1])
    # print(x_t[0:10])

    plt.plot(time, position)
    plt.plot(time[15:-1], pos_mag_factor*x_t[15:-1]) # why multiply by 10000
    # plt.ylim(-20,20)
    plt.show()

    plt.plot(time, gamma_true)
    plt.plot(time, gamma_star)
    plt.xlim(30.01,35)
    plt.show()
    print(loss_sqrt)

    return Kp_0, Kd_0, alpha, c1, c2, theta_mag_factor, pos_mag_factor

training_gains = ['1','5','20']
time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/time.npy")
position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/position.npy")
theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/angle.npy")

train("20")

# gains = [2,5,7,10,12,15,17,20,25,30]
# theta_mag_factors = []
# pos_mag_factors = []
# Kd_0s = []
# Kp_0s = []
# alphas = []
# c1s = []
# c2s = []

# for gain in gains:
#     training_gain2 = str(gain)
#     test_gain2 = str(gain)

#     time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/time.npy")
#     position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/position.npy")
#     theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + test_gain2 + "/angle.npy")

#     result = minimize(calc_func_find_Kp_0, (0.1,0.1))
#     Kp_0 = result.x[0]
#     Kd_0 = result.x[1]
#     bnds = ((0,None),(0,None), (0,None))
#     result2 = minimize(calc_func_find_alpha,(0.5,1, 1), bounds=bnds)
#     alpha = result2.x[0]
#     c1 = result2.x[1]
#     c2 = result2.x[2]

#     calibration_index =  np.where(time > 30)
#     calibration_time = time[calibration_index[0][0]]
#     time = time[calibration_index[0][0]:-1]
#     theta = theta[calibration_index[0][0]:-1]
#     position = position[calibration_index[0][0]:-1]

#     loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, dell  = calc_func(time, theta, position, alpha, Kp_0, Kd_0, calibration_time, 1, int(test_gain2), c1, c2)
#     theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
#     position = position - np.mean(position)
#     x_t = x_t
#     x_t = x_t - np.mean(x_t[15:-1])
#     pos_mag_factor = np.std(position[15:-1])/np.std(x_t[15:-1])

#     theta_mag_factors.append(theta_mag_factor)
#     pos_mag_factors.append(pos_mag_factor)
#     Kp_0s.append(Kp_0)
#     Kd_0s.append(Kd_0)
#     alphas.append(alpha)
#     c1s.append(c1)
#     c2s.append(c2)



# plt.plot(gains, theta_mag_factors)
# plt.xlabel('gains')
# plt.ylabel('theta scaling factor')
# plt.show()
# plt.plot(gains, pos_mag_factors)
# plt.xlabel('gains')
# plt.ylabel('position scaling factor')
# plt.show()
# plt.plot(gains, alphas)
# plt.xlabel('gains')
# plt.ylabel('alpha (internal learning rate)')
# plt.show()
# plt.plot(gains, c1s)
# plt.xlabel('gains')
# plt.ylabel('c1')
# plt.show()
# plt.plot(gains, c2s)
# plt.xlabel('gains')
# plt.ylabel('c2')
# plt.show()
