
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
    
    pos_estimates = pos_estimates - np.mean(pos_estimates)


    g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
    theta_est = np.convolve(pos_estimates, g_t, 'full')

    theta_est = theta_est[len(time)-2:-1]
    theta_est = polynomial(theta_est, order=order, plot=False)
    theta_est = theta_est/4


   
    return pos_estimates + 750, theta_est, cart_acc
