
import numpy as np
import config
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import config
import math
def simulate(step_estimates, start, dt, time):


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
    


    # g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    # theta_est = np.convolve(pos_estimates -750, g_t, 'full')

    # theta_est = theta_est[len(time)-2:-1]
    # # theta = theta[0:len(theta_est)]


    # pos_estimates = pos_estimates[0::2]

    pendulum_angle = np.zeros(len(pos_estimates))
    pen_vel = np.zeros(len(pos_estimates))
    pen_acc = np.zeros(len(pos_estimates))
    pendulum_angle[0] = start[1]
    

  
    for i in range(len(step_estimates)-1):
        # cart_acceleration = (step_estimates[i+1] - step_estimates[i])/(dt)
        cart_acceleration = cart_acc[i]/config.POSITION_GAIN 
        pendulum_acceleration = (config.G*np.sin(pendulum_angle[i])/(config.PEN_LENGTH)) - (cart_acceleration*dt*np.cos(pendulum_angle[i])/(config.PEN_LENGTH))
        pen_acc[i] = pendulum_acceleration
        cart_acc[i] = cart_acceleration*50
        pen_vel[i+1] = pen_vel[i] + pen_acc[i] * dt
        pendulum_angle[i+1] = pendulum_angle[i] + pen_vel[i] * dt

        
    
    print(pen_acc[0:10])
    #pos_estimates = np.repeat(pos_estimates,2)
    # pendulum_angle = np.repeat(pendulum_angle,2)

    return pos_estimates, pendulum_angle, pen_acc[0::2], cart_acc[0::2], pen_vel[0::2]

gain = 5
trial = 1
P_id = 7

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')
time = time[0::2]

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
pos_es, ang_es, penacc, cartacc, penvel = simulate(steps, start, 0.017, gain)

pendulum_acceleration = (config.G*np.sin(theta)/(config.PEN_LENGTH)) - ((steps2/(50*0.017))*np.cos(theta)/(config.PEN_LENGTH))
print(theta_steps2[0:10]/(0.017**2))
print(ang_es[0:10])
print(theta[0:10])
plt.plot(time, pendulum_acceleration[0::2])
plt.plot(time[0:-1], theta_steps2[0::2]/(0.017**2), label = 'simulation')
plt.legend()
plt.show()



plt.plot(time, cartacc, label = 'simulation')
plt.plot(time, steps2[0::2]/(0.017**2))
plt.legend()
plt.show()


plt.plot(time, penvel[0:-1], label = 'simulation')
plt.plot(time, theta_steps[0::2]/0.017)
plt.legend()
plt.show()

plt.plot(time, theta[0::2], label = 'simulation')
plt.plot(time, ang_es[0::2][0:-1])
plt.legend()
plt.show()


plt.plot(time[0:-1], penacc[0:-2], label = 'simulation')
plt.plot(time[0:-1], theta_steps2[0::2]/(0.017**2))
plt.legend()
plt.show()

# plt.plot(time, theta[0::2], label = 'simulation')
# plt.legend()
# plt.show()


plt.plot(time[0:-1], theta_steps2[0::2], label = 'simulation')

plt.legend()
plt.show()






plt.plot(time, pos_es[0:-1], label = 'simulation')
plt.plot(time, position[0::2][0:-1])
plt.legend()
plt.show()

plt.plot(time, ang_es[0:-1], label = 'simulation')
plt.plot(time, theta[0::2])
plt.legend()
plt.show()






