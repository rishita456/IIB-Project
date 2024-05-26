import numpy as np
import control as ct
import matplotlib.pyplot as plt

P_id = 2
training_gain = 19
trial = 1

time = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
position = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
theta = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')
dt = time[6] - time[4]


calibration_index =  np.where(time < 30)
time = time[0:calibration_index[-1][-1]]
theta = theta[0:calibration_index[-1][-1]]
position = position[0:calibration_index[-1][-1]]

# time = time[0::2]
# theta = theta[0::2]
# theta = theta[0::2]

time_sim = np.linspace(0, 30, len(time))
dt_sim = time_sim[4] - time_sim[3]

# pendulum system impulse response
num = [1, 0, 0]
den = [-4, 0, 9.8]
g_t = (0.000553399*(np.exp(1)**(-0.221359*time) - np.exp(1)**(0.221359*time)))/50
sys = ct.tf(num, den, dt_sim)

t, out = ct.impulse_response(sys)

plt.plot(t, out)
plt.show()

