import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
# from Experiment.analysis_pt1.analysev4 import calc_func


# Plotting gamma
def calc_func(time, theta, position,Kp_0, Kd_0, gamma):

    global P_id
    global trial
    global training_gain

    gamma_true = int(training_gain)*np.ones(len(time))
    
    dtheta = np.zeros_like(theta)
    deltat = time[2]-time[0]
    c1 = 1
    c2 = 1

    i=0

    for t in time:
            
        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
   
    Kp = Kp_0 * (gamma**(-c1))
    Kd = Kd_0 * (gamma**(-c2))
    

    x = -gamma*(theta*Kp + dtheta*Kd) 
    time = time - 0.15
    
    # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
    # g_t = 0.02766*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time))
    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50

    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]

    theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
    theta_est = theta_est

    loss = np.linalg.norm(gamma_true*theta - gamma*theta_est)


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

    # plt.plot(np.fft.fftfreq(len(time), deltat), np.fft.fft(theta), label = 'theta')
    # plt.plot(np.fft.fftfreq(len(time), deltat), np.fft.fft(theta_est), label = 'estimate')
    # plt.title('Gain = '+str(training_gain))
    # plt.xlabel('Frequency')
    # plt.xlim((-5,5))
    # plt.legend()
    # plt.show()
    print(loss)

    return loss, gamma, theta, dtheta, theta_est, gamma_true, x, time, g_t, theta_mag_factor

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

    if settle_time == 0:
        settle_time = 29.991
    settle_index = np.where(time>settle_time)
    if settle_time != 29.991:

        steady_state_calibration_time = time[int(settle_index[0][0]):-1]
        steady_state_calibration_theta = theta[int(settle_index[0][0]):-1]
        steady_state_calibration_position = position[int(settle_index[0][0]):-1]
    
        time = steady_state_calibration_time
        theta = steady_state_calibration_theta
        position = steady_state_calibration_position
        
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


        Kp = Kp_0 * (gamma_star**(-c1))
        Kd = Kd_0 * (gamma_star**(-c2))
        

        x = -gamma_star*(theta*Kp + dtheta*Kd) # why -
        time = time - 0.15
        
        # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
        # g_t = 0.02766*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time))
        g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
        theta_est = np.convolve(x, g_t, 'full')

        theta_est = theta_est[len(time)-2:-1]
        theta = theta[0:len(theta_est)]

        theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
        theta_est = theta_est


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

    else:
        loss = np.nan
        theta_mag_factor = np.nan
        Kp_0 = np.nan
        Kd_0 = np.nan
    



    return loss

def calc_func_find_gamma(gamma):

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
    alpha = 10000000
    t_sep = 0
    gain1 = int(training_gain)
    gain2 = int(training_gain)
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
    if settle_time != 29.991:

        steady_state_calibration_time = time[int(settle_index[0][0]):-1]
        steady_state_calibration_theta = theta[int(settle_index[0][0]):-1]
        steady_state_calibration_position = position[int(settle_index[0][0]):-1]
    
        time = steady_state_calibration_time
        theta = steady_state_calibration_theta
        position = steady_state_calibration_position
        
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


        Kp = Kp_0 * (gamma**(-c1))
        Kd = Kd_0 * (gamma**(-c2))
        
        Kp = Kp[2:-1]
        Kd = Kd[2:-1]
        gamma = gamma[2:-1]

   
        theta = theta[0:len(Kp)]
        dtheta = dtheta[0:len(Kd)]
        x = -gamma*(theta*Kp + dtheta*Kd) # why -
        time = time - 0.15
        
        # g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time)) #why multiply by -deltatsquared
        # g_t = 0.02766*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time))
        g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
        theta_est = np.convolve(x, g_t, 'full')

        theta_est = theta_est[len(time)-2:-1]
        theta = theta[0:len(theta_est)]

        theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
        theta_est = theta_est


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

        loss = np.linalg.norm(gamma_true*theta - gamma*theta_est)

    else:
        loss = np.nan
        theta_mag_factor = np.nan
        Kp_0 = np.nan
        Kd_0 = np.nan
    

    print(loss)

    return loss


P_id = 1
trial = 2

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '17' + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '17' + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + '17' + '/trial' + str(trial) + '/angle.npy')
training_gain = '17'
result = minimize(calc_func_find_Kp_0, (1,1))
Kp_0 = result.x[0]
Kd_0 = result.x[1]
print(result.fun)
result2 = minimize(calc_func_find_gamma, 17*np.ones(len(time)), tol=1)
print(result2.fun)

calc_func(time, theta, position,Kp_0, Kd_0, result2.x)



