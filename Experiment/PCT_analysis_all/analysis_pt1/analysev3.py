
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, summation, exp

from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.ndimage import convolve
from scipy.stats import zscore
from scipy.fft import fft


# data = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot2_exp_data/g1_5_g2_20/tsep_12_run1.npy')
time = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/g1_5_g2_20/tsep12/run1/time.npy")
position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/g1_5_g2_20/tsep12/run1/position.npy")
theta = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/g1_5_g2_20/tsep12/run1/angle.npy")
raw_hand_position = np.load("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot2_exp_data/g1_5_g2_20/tsep12/run1/raw_position.npy")
print(np.where(time < 30))

settle_time = 0
index = 0
mse = 0
calibration_index =  np.where(time < 30)
print((calibration_index[-1][-1]))
time = time[0:calibration_index[-1][-1]]
theta = theta[0:calibration_index[-1][-1]]
position = position[0:calibration_index[-1][-1]]
# plt.plot(time,theta)
# plt.show()

for t in time:
    # print(int(t))
    slice = theta[index:-1]
    # print(slice)

    if np.all(abs((slice)) < 4 ):
        mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
        mse = np.sqrt(mse)
        settle_time = t
        break
    index = index + 1

print(settle_time)
print(mse)

settle_index = np.where(time>settle_time)
steady_state_calibration_time = time[settle_index[0][0]:calibration_index[-1][-1]]
steady_state_calibration_theta = theta[settle_index[0][0]:calibration_index[-1][-1]]


def calc_func(time, theta, position, alpha, Kp_0, Kd_0, t_sep, gain1, gain2):
    
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
    # gamma_true = np.ones_like(time)
    # gamma_star = gamma_true

    Kp = Kp_0 * (gamma_star**(-1))
    Kd = Kd_0 
    

    x = -gamma_star*(theta*Kp + dtheta*Kd)
    # plt.plot(x)
    # plt.plot(position-750)
    # plt.plot
    # plt.show()
    time = time - 2
    g_t = 0.0015*(np.exp(1)**(-0.313*time) - np.exp(1)**(0.313*time))
    # g_t = (0.2066*(-np.exp(1)**(-0.226*time) + np.exp(1)**(0.226*time)) -0.0933*time)/(deltat**2)

    theta_est = np.convolve(x, g_t, 'full')

    # plt.plot(theta_est)
    # plt.show()


    theta_est = theta_est[len(time)-2:-1]
    theta = theta[0:len(theta_est)]
    theta = theta/np.linalg.norm(theta)
    theta_est = -theta_est/np.linalg.norm(theta_est)
  
    loss = np.linalg.norm(theta - theta_est)


    # plt.plot(time, theta)
    # plt.plot(time, theta_est)
    # # plt.plot(time, dtheta)
    # # plt.plot(time, x)
    # plt.show()

    return loss, gamma_star, theta, dtheta, theta_est, gamma_true, x, time, g_t, del_gamma_star

alpha = 1
Kp_0 = 0.01#-0.05
Kd_0 = 0.01
epochs = 50
x = np.arange(0,1, 0.1)
loss = np.ones(len(x))
loss_iter = np.ones(epochs)
Kp_0s = np.ones(epochs)
Kd_0s = np.ones(epochs)
alphas = np.ones(epochs)
lr = 0.05
loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, dell  = calc_func(time, theta, position, alpha, Kp_0, Kd_0, 12, 5, 20)
plt.plot(gamma_star)
plt.plot(gamma_true)
plt.show()

for iter in range(epochs):

    loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, del_gamma_star  = calc_func(steady_state_calibration_time, steady_state_calibration_theta, position, alpha, Kp_0, Kd_0, 12, 5, 20)


    # grad_alpha = np.convolve((del_gamma_star*dtheta*Kd_0), g_t, 'full')
    # grad_alpha = grad_alpha[len(time)-2:-1]
    # grad_alpha = grad_alpha/np.linalg.norm(grad_alpha)
    # alpha = alpha + 2*lr*(np.dot(abs(theta_est - theta), grad_alpha))

    grad_Kp_0 = np.convolve(theta, g_t, 'full')
    grad_Kp_0 = grad_Kp_0[len(time)-2:-1]
    grad_Kp_0 = grad_Kp_0/np.linalg.norm(grad_Kp_0)

    Kp_0 = Kp_0 + 2*lr*(np.dot(abs(theta_est - theta), grad_Kp_0))

    # grad_Kd_0 = np.convolve((gamma_star*dtheta), g_t, 'full')
    # grad_Kd_0 = grad_Kd_0[len(time)-2:-1]
    # grad_Kd_0 = grad_Kd_0/np.linalg.norm(grad_Kd_0)
    # Kd_0 = Kd_0 + 2*lr*(np.dot(abs(theta_est - theta), grad_Kd_0))
    # if iter == 1 or iter ==99:
    #     plt.plot(abs(theta_est-theta))
    loss_iter[iter] = loss_sqrt**2
    Kp_0s[iter] = Kp_0
    Kd_0s[iter] = Kd_0
    alphas[iter] = alpha

    print(Kp_0, Kd_0, alpha)

# # for iter in range(len(x)):

# #     loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t  = calc_func(data, x[iter], Kp_0, Kd_0, 12, 2, 50)
# #     loss[iter] = loss_sqrt**2
    

# # plt.plot(time, theta_est)   

# # plt.plot(time,theta)
# loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, del_gamma_star  = calc_func(data, alpha, Kp_0, Kd_0, 12, 5, 50)
# test_data = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/pilot_exp_data/g1_2_g2_10/tsep_10.npz.npy')
# loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time, g_t, del_gamma_star  = calc_func(test_data, alpha, Kp_0, Kd_0, 10, 2, 10)
# print(loss_sqrt)
# # plt.plot(abs(theta_est-theta))

plt.plot(gamma_true, label = "actual feedback gain (gamma true)")
plt.plot(gamma_star, label = "percieved feedback gain (gamma star)")

# # plt.plot(x,loss)
# # plt.plot(alphas, label = "internal learning rate (alphas) over epochs")
# # fig2, (axa, axb) = plt.subplots(2)
# # axa.plot(loss_iter, label = "loss function over epochs")
# # axa.legend()
# # axb.plot(time, theta_est, label = 'theta estimate')  
# # axb.plot(time, theta, label = 'observed theta')
# # axb.legend()
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(Kd_0s, label = 'K_d0 over epochs')
# ax2.plot(Kp_0s, label = 'K_p0 over epochs')
# ax3.plot(alphas, label = 'alpha over epochs')
# ax1.legend()
# ax2.legend()
# ax3.legend()
# plt.show()
# plt.plot(steady_state_calibration_time, theta_est, label = 'theta estimate')  
# plt.plot(steady_state_calibration_time, steady_state_calibration_theta, label = 'observed theta')
# plt.legend()

plt.show()


