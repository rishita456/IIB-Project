#testing simulate to check if they return accurate results


import numpy as np
import config
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import config
import math
import obspy
from obspy.signal.detrend import polynomial
def simulate(step_estimates, start, dt, time, order = 6):


    pos_estimates = np.zeros(len(step_estimates)+1)
    steps2 = []
    pos_estimates[0] = start[0]
    
    for i in range(len(step_estimates)):
        pos_estimates[i+1] = pos_estimates[i]+step_estimates[i]

    cart_vel = step_estimates/dt
    for i in range(len(step_estimates)-1):
        steps2.append(cart_vel[i+1]-cart_vel[i])

    steps2 = np.array(steps2)
    cart_acc = steps2/(dt)
    


    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    theta_est = np.convolve(pos_estimates -750, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta_est = polynomial(theta_est, order=order, plot=False)
    theta_est = theta_est/1.9


   
    return pos_estimates, theta_est, cart_acc

gain = 5
trial = 1
P_id = 9

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')


steps = []
steps2 = []
theta_steps = []
theta_steps2 = []

print(np.shape(time))


for i in range(len(position)-1):
    steps.append(position[i+1]-position[i])

for i in range(len(steps)-1):
    steps2.append(steps[i+1]-steps[i])
    
for i in range(len(theta)-1):
    theta_steps.append(theta[i+1]-theta[i])

for i in range(len(theta_steps)-1):
    theta_steps2.append(theta_steps[i+1]-theta_steps[i])

steps = np.array(steps)
steps2 = np.array(steps2)
theta_steps = np.array(theta_steps)
theta_steps2 = np.array(theta_steps2)

start = (position[1], theta[1])
pos_es, ang_es, cartacc = simulate(steps, start, 0.017, time)




# plt.plot(time, cartacc, label = 'simulation')
# plt.plot(time, steps2/(0.017**2))
# plt.legend()
# plt.show()


# plt.plot(time, theta, label = 'simulation')
# plt.plot(time, ang_es[1:-1])
# plt.legend()
# plt.show()


plt.plot(time, pos_es[1:-1], label = 'simulation')
plt.plot(time, position[1:-1])
plt.legend()
plt.show()

plt.plot(time, ang_es[1:-1], label = 'simulation')
plt.plot(time, theta)
plt.legend()
plt.show()







