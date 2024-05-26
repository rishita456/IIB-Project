import numpy as np
from glob import glob
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def calc_func_find_Kp_0(K):

    global training_gain2
    global training_gain3
    global tsep
    global run
    global Kp_0, Kd_0
    global settle_time, mse
    global switch

    if switch == 0:
        time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_" + training_gain2 + "_g2_" + training_gain3 + "/" + tsep + "/" + run + "/time.npy")
        position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_" + training_gain2 + "_g2_" + training_gain3 + "/" + tsep + "/" + run + "/position.npy")
        theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_" + training_gain2 + "_g2_" + training_gain3 + "/" + tsep + "/" + run + "/angle.npy")

    if switch == 1:
        time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + training_gain2 + "/time.npy")
        position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + training_gain2 + "/position.npy")
        theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/single_disturbance/1_to_" + training_gain2 + "/angle.npy")
    
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

Kps = []
Kds = []
settletimes = []
mses = []
training_gain2s = ['2', '5', '7', '10', '12', '15', '17', '20', '25', '30']
training_gain3s = ['10', '20']
tseps = ['tsep2', 'tsep4', 'tsep6', 'tsep12']
runs = ['run1', 'run2', 'run3']
switches = [0,1]


for s in switches:
    for gain2 in training_gain2s:
        for gain3 in training_gain3s:
            for t in tseps:
                for r in runs:

                    try:
                        switch = s
                        training_gain2 = gain2
                        training_gain3 = gain3
                        tsep = t
                        run = r

                        result = minimize(calc_func_find_Kp_0, (0.1,0.1))
                        Kp_0 = result.x[0]
                        Kd_0 = result.x[1]

                        Kps.append(Kp_0)
                        Kds.append(Kd_0)
                        settletimes.append(settle_time)
                        mses.append(mse)

                    except Exception as e:
                        pass


plt.scatter(settletimes, Kps, c='r', label = 'Kp0s')
plt.scatter(settletimes, Kds, c='b', label = 'Kd0s')
plt.xlabel('Settling time from t= 0(seconds)')
plt.ylabel('Kp0s and Kd0s')
plt.legend()
plt.show()

plt.scatter(mses, Kps, c='r', label = 'Kp0s')
plt.scatter(mses, Kds, c='b', label = 'Kd0s')
plt.xlabel(' Root Mean Squared Steady State Error (degrees)')
plt.ylabel('Kp0s and Kd0s')
plt.legend()
plt.show()
