
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

def gaussian_kernel(size, sigma):
    """Generate a 1D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x: (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-size//2)**2/(2*sigma**2)),
        (size,), 
        dtype=int
    )
    return kernel / kernel.sum()

def gaussian_smoothing(data, kernel_size, sigma):
    """Apply Gaussian smoothing to the input data."""
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_data = convolve(data, kernel, mode='nearest')
    return smoothed_data


# Assuming samples2 is a NumPy array containing your data
# maxLag = 10 # You can adjust this value based on your needs

# # Compute autocorrelation
# acf = np.correlate(run1[1], run1[1], mode='full')[-len(run1[1]):]

data = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/pilot_exp_data/g1_2_g2_50/tsep_12.npz.npy')
smoothed_data = gaussian_smoothing(data[1], 2, 6000)
max_y = max(data[1])
error_max = float(0.1*max_y)
gamma_true = np.zeros_like(data[0])


settle_time = 0
index = 0
mse = 0

transformed_output = fft(data[1])

for t in data[0]:
    # print(int(t))
    slice = data[1][index:-1]
    # print(slice)

    if np.all(abs((slice)) < error_max ):
        mse = np.square(np.subtract(np.zeros_like(slice), slice)).mean() 
        mse = np.sqrt(mse)
        settle_time = t
        break
    index = index + 1

print(settle_time)
print(mse)
# Plot autocorrelation function
# plt.plot(data[0],data[1])
# plt.plot(data[0],smoothed_data)
# plt.plot((0, 30), (0, 0), 'black')
# plt.plot((0, 30), (error_max, error_max), 'r', label='error band')
# plt.plot((0, 30), (-error_max, -error_max),'r')
# plt.axvline(color = 'green', x=settle_time, label='settling time = %.2fs' %settle_time )
# plt.axvline(color = 'black', x=12, label='time of disturbance = 12s' )
# plt.xlabel('Time (s)')
# plt.ylabel('Pendulum angle (degrees)')
# plt.title('Time response')
# plt.legend()
# plt.show()


def calc_func(data, alpha, Kp_0, Kd_0, c1, c2):

    time = np.array(data[0])
    theta = np.array(data[1])
    dtheta = np.zeros_like(theta)
    deltat = time[2]-time[0]
    t_sep = 12
    gain1=2
    gain2=50
    # alpha = 0.1
    # Kp_0 = 2
    # Kd_0 = 1
    # c1 = 2
    # c2 = 2
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
                # print(gamma_old)

        if i>2 and i%2==0:
            dtheta[i] = (theta[i]-theta[i-2])/deltat
            dtheta[i-1]=dtheta[i]
        i+=1
    # gamma_star = gamma_true
    Kp = Kp_0 * (gamma_star**(-1))
    Kd = Kd_0 #(gamma_star**(-1))
    # plt.plot(Kp)
    # plt.plot(Kd)
    # plt.show()

    x = -gamma_star*(theta*Kp + dtheta*Kd)
    # plt.plot(time,x)
    # plt.show()

    g_t = 0.015*(np.exp(1)**(-0.295*time) - np.exp(1)**(0.295*time))/(deltat**2)

    theta_est = np.convolve(x, g_t, 'full')

    theta_est = theta_est[len(x)+200:-1]
    theta = theta[0:len(theta_est)]
    time = time[0:len(theta_est)]
    
    theta = theta/np.linalg.norm(theta)
    theta_est = -theta_est/np.linalg.norm(theta_est)
    # plt.plot(time,theta)
    # plt.plot(time, theta_est)
    # plt.show()
    loss = np.linalg.norm(theta - theta_est)


    # plt.plot(time, theta)
    # plt.plot(time, theta_est)
    # # plt.plot(time, dtheta)
    # # plt.plot(time, x)
    # plt.show()

    return loss, gamma_star, theta, dtheta, theta_est, gamma_true, x, time

alpha = -5
Kp_0 = 0.01
Kd_0 = 0.8
c1 = 2
c2 = 2
x = np.arange(0,1, 0.1)
loss = np.ones(len(x))

lr = 0.1
loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time  = calc_func(data, alpha, Kp_0, Kd_0, c1, c2)

for iter in range(len(x)):

    loss_sqrt, gamma_star, theta, dtheta, theta_est, gamma_true, x_t, time  = calc_func(data, x[iter], Kp_0, Kd_0, c1, c2)

    # Kp_0 = Kp_0 + lr*(2*loss_sqrt/len(theta))*(np.sum((theta)))
    # Kd_0 = Kd_0 + lr*(2*loss_sqrt/len(theta))*(np.sum((-gamma_star*dtheta)))
    # Kp_0 = Kp_0 + lr*(2*loss_sqrt)*(((theta)))
    # Kd_0 = Kd_0 + lr*(2*loss_sqrt)*((-gamma_star*dtheta))
    
    loss[iter] = loss_sqrt**2
    

#     print(Kp_0, Kd_0)
# print(len(theta))
# theta_est = theta_est * (max(theta)/max(theta_est))
# plt.plot(time,theta_est, label = 'theta estimate')
# time = time - 2
# plt.plot(time,theta, label = 'theta actual')
# plt.plot(gamma_true)
# plt.plot(gamma_star)
plt.plot(x,loss)
# plt.plot(x_t)
# plt.legend()
# plt.xlabel('time')
# plt.ylabel('degrees')
plt.show()



