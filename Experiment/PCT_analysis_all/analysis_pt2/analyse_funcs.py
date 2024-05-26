import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression


# Reused analyses functions

def calc_func(training_gain, time, theta, position, alpha, Kp_0, Kd_0, t_sep, gain1, gain2, c1, c2, delay = 0.15):

    
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

    pos_mag_factor = np.std(position[15:-1])/np.std(x[15:-1])
    x = x * pos_mag_factor

    
    time = time - delay

    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50

    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]

    theta_mag_factor = np.std(theta[15:-1])/np.std(theta_est[15:-1])
    theta_est = theta_est * theta_mag_factor

    if pearsonr(theta_est, theta)[0] < -0.5:

        pos_mag_factor = -pos_mag_factor
        x = -x
        time = time - 0.15

        g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50

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


    # plt.plot(time, (position[1:-1] - 750), label = 'real')
    #     # plt.plot(time, x*pos_mag_factor, label= 'estimate')
    #     # plt.show()
    plt.plot(time, detrended, label = 'position')
    plt.plot(time, x, label = 'estimate')
    plt.title('Gain = '+str(training_gain))
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

    # plt.plot(np.fft.fftfreq(len(time), deltat), np.fft.fft(theta), label = 'theta')
    # plt.plot(np.fft.fftfreq(len(time), deltat), np.fft.fft(theta_est), label = 'estimate')
    # plt.title('Gain = '+str(training_gain))
    # plt.xlabel('Frequency')
    # plt.xlim((-5,5))
    # plt.legend()
    # plt.show()

    return loss, gamma_star, theta, dtheta, theta_est, gamma_true, x, time, g_t, theta_mag_factor, pos_mag_factor


