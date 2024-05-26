import numpy as np
import matplotlib.pyplot as plt

def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def transfer(x, time):

    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    x = x-750
    # x_shifted1 = shift_elements(x, 1, 0)
    # x_shifted2 = shift_elements(x, 2, 0)
    # deltat = time[6] - time[4]
    

    # cart_acceleration = (x - (2*x_shifted1) + x_shifted2)/(deltat**2)
    # cart_acceleration = cart_acceleration/50
    # pendulum_acceleration = (9.81*np.sin(theta)/(200)) - (cart_acceleration*np.cos(theta)/(200))

    # velocity = pendulum_acceleration * deltat
    # theta_est_2= velocity * deltat
    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(time)-2:-3]

    return theta_est

P_id = 1
trial = 2
training_gain = '17'

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + training_gain + '/trial' + str(trial) + '/angle.npy')

print(np.shape(theta))

theta_est = transfer(position, time)
plt.plot(time, theta, label = 'theta')
plt.plot(time, theta_est, label = 'estimate')
plt.show()