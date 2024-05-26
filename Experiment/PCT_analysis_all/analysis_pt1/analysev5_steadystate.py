
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from analysev4 import calc_func

def calc_func_find_Kp_0(K):

    time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_10/time.npy")
    position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_10/position.npy")
    theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_10/angle.npy")
    # raw_hand_position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep12/run1/raw_position.npy")
    alpha = 10000000
    t_sep = 0
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


result = minimize(calc_func_find_Kp_0, (0.1,0.1))
print(result.fun)
print(result.x)

time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_20/time.npy")
position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_20/position.npy")
theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/single_disturbance/1_to_20/angle.npy")
# raw_hand_position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep12/run1/raw_position.npy")
alpha = 10000000000
t_sep = 0
gain1 = 1
gain2 = 1
Kp_0 = result.x[0]
Kd_0 = result.x[1]

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

loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, dell  = calc_func(steady_state_calibration_time, steady_state_calibration_theta, steady_state_calibration_position, alpha, Kp_0,Kd_0, 0, 1, 1, 1, 1)

plt.plot(time, theta_est, c = 'red', label = '$\gamma$')
plt.plot(time, theta_est, c = 'blue', label = '$\gamma^*$')
plt.plot(time,theta)
plt.xlabel('$\gamma$')
plt.ylabel('$c_2$')

plt.legend()

plt.show()

plt.plot(time, steady_state_calibration_position-750)
plt.plot(time, x_t)
plt.show()