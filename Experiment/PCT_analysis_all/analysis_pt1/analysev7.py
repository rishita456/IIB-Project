import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from analysev6 import train
from analysev4 import calc_func

from io import BytesIO

from scipy.ndimage import convolve
from scipy.stats import zscore
from scipy.fft import fft


gains = [2,5,7,10,12,15,17,20,25,30]
theta_mag_factors = []
pos_mag_factors = []
Kd_0s = []
Kp_0s = []
alphas = []
c1s = []
c2s = []
settle_times = []
mses = []


for gain in gains:
    # training_gain2 = str(gain)
    # test_gain2 = str(gain)

    Kp_0, Kd_0, alpha, c1, c2, theta_mag_factor, pos_mag_factor, settle_time, mse = train(str(gain))

    

    theta_mag_factors.append(theta_mag_factor)
    pos_mag_factors.append(pos_mag_factor)
    Kp_0s.append(Kp_0)
    Kd_0s.append(Kd_0)
    alphas.append(alpha)
    c1s.append(c1)
    c2s.append(c2)
    settle_times.append(settle_time)
    mses.append(mse)


# plt.scatter(settle_times, Kp_0s, c='r', label = 'Kp0s')
# plt.scatter(settle_times, Kd_0s, c='b', label = 'Kd0s')
# plt.show()



plt.plot(gains, theta_mag_factors)
plt.xlabel('New gain post disturbance (trials conducted in ascending order with gain for trial 1 = 2)')
plt.ylabel('theta scaling factor')
plt.title('Single disturbance trials with initial gain 1')
plt.show()

plt.plot(gains, pos_mag_factors)
plt.xlabel('New gain post disturbance (trials conducted in ascending order with gain for trial 1 = 2)')
plt.ylabel('position scaling factor')
plt.title('Single disturbance trials with initial gain 1')
plt.show()

plt.plot(gains, alphas)
plt.xlabel('New gain post disturbance (trials conducted in ascending order with gain for trial 1 = 2)')
plt.ylabel('alpha (internal learning rate)')
plt.title('Single disturbance trials with initial gain 1')
plt.show()

plt.plot(gains, c1s)
plt.xlabel('New gain post disturbance (trials conducted in ascending order with gain for trial 1 = 2)')
plt.ylabel('c1')
plt.title('Single disturbance trials with initial gain 1')
plt.show()

plt.plot(gains, c2s)
plt.xlabel('New gain post disturbance (trials conducted in ascending order with gain for trial 1 = 2)')
plt.ylabel('c2')
plt.title('Single disturbance trials with initial gain 1')
plt.show()