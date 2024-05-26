import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import config
from estimate_steps import get_step_estimates, get_noise_pdf_and_samples
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

def get_marginal_pdf(data):
    

    data_values = np.linspace(min(data), max(data), 200)  # Range of x values general
    kde = gaussian_kde(data)
    marginal_density = kde.evaluate(data_values)

    return data_values, marginal_density

def get_conditional_pdf(data1, data2): # data2 conditional on data1

    data1_values, data2_values, joint_density = get_joint_pdf(data1, data2)
    # Fit a separate KDE for X
    kde_data1 = gaussian_kde(data1)
    marginal_density_data1 = kde_data1.evaluate(data1_values)

    conditional_density = np.zeros_like(joint_density)
    # print(np.shape(joint_density))
    for i in range(len(data1_values)):
        conditional_density[i, :] = joint_density[i,:]/marginal_density_data1[i]

    return data1_values, data2_values, conditional_density



def get_joint_pdf(data1, data2):
    
    data1_values = np.linspace(min(data1),max(data1), 200)
    data2_values = np.linspace(min(data2),max(data2), 200)
    d_1, d_2 = np.meshgrid(data1_values, data2_values)

    positions = np.vstack([d_1.ravel(), d_2.ravel()])
    values = np.vstack([data1, data2])
    density = gaussian_kde(values)
    
    # print(d_1.shape)
    joint_density = np.reshape(density(positions).T, d_1.shape)
    # print(np.shape(joint_density))

 
    return data1_values, data2_values, joint_density


def fit_linear_regression(gains, data):
    model = LinearRegression()
    gains = gains.reshape((1,-1)).T
    model.fit(gains, data)
    y_pred = model.predict(gains)
    residuals = gains - y_pred
    std_dev = np.std(residuals)
    return y_pred, std_dev

def fit_polynomial_regression(gains, data, degree):

    poly_features = PolynomialFeatures(degree=degree)
    model = LinearRegression()
    gains = gains.reshape((1,-1)).T
    gains_poly = poly_features.fit_transform(gains)
    model.fit(gains_poly,data)
    y_pred = model.predict(gains_poly)
    residuals = gains - y_pred
    std_dev = np.std(residuals)

    return y_pred, std_dev


def separate_data(training_gains, phase, P_id):

    time_data = []
    position_data = []
    theta_data = []
    final_tg = []

    if phase == 1:
        training_gains = [2, 5, 7, 10, 12, 15, 17, 20]
        trials = [1,2]

        for gain in training_gains:
            for trial in trials:

                try:

                    time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/time.npy')
                    position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/position.npy')
                    theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial) + '/angle.npy')

                    time_data.append(time_og)
                    position_data.append(position_og)
                    theta_data.append(theta_og)
                    final_tg.append(gain)
                except FileNotFoundError:
                    pass
        



    if phase == 2:
        training_gains = training_gains[1][:]
        index = 0
        for gain in training_gains:
            index = index + 1
            slice = training_gains[0:index]
            trial = slice.count(gain)
            training_gain = gain
            time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/time.npy')
            position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/position.npy')
            theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/1_disturbance/g0_1_g1_' + str(training_gain) + '/trial' + str(trial) + '/angle.npy')
            
            calibration_index =  np.where(time_og < 30)

            try:
                time1 = time_og[0:calibration_index[-1][-1]]
                theta1 = theta_og[0:calibration_index[-1][-1]]
                position1 = position_og[0:calibration_index[-1][-1]]
                final_tg.append(1)

            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)
            

            try:
                time2 = time_og[calibration_index[-1][-1]:]
                theta2 = theta_og[calibration_index[-1][-1]:]
                position2 = position_og[calibration_index[-1][-1]:]
                final_tg.append(gain)
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time2)
            position_data.append(position2)
            theta_data.append(theta2)
            
            

    if phase == 3:
        training_gains = training_gains[2][0:8]
        index = 0
        for gain in training_gains:
            index = index + 1
            slice = training_gains[0:index]
            trial = slice.count(gain)
            training_gain = gain
            time_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain) + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/time.npy')
            position_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/position.npy')
            theta_og = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/data/participant_' + str(P_id) + '/2_disturbance/g0_1_g1_' + str(training_gain)  + '_g2_' + str(training_gain*1.5) + '/trial' + str(trial) + '/angle.npy')
                    
            calibration_index =  np.where(time_og < 30)
            change_index = np.where(time_og < 40)

            
            try:
                time1 = time_og[0:calibration_index[-1][-1]]
                theta1 = theta_og[0:calibration_index[-1][-1]]
                position1 = position_og[0:calibration_index[-1][-1]]
                final_tg.append(1)
            except IndexError:
                time1 = np.zeros(200)
                theta1 = np.zeros(200)
                position1 = np.zeros(200)
                final_tg.append(None)
                

            time_data.append(time1)
            position_data.append(position1)
            theta_data.append(theta1)
            

            try:
                time2 = time_og[calibration_index[-1][-1]:change_index[-1][-1]]
                theta2 = theta_og[calibration_index[-1][-1]:change_index[-1][-1]]
                position2 = position_og[calibration_index[-1][-1]:change_index[-1][-1]]
                final_tg.append(gain)
            except IndexError:
                time2 = np.zeros(200)
                theta2 = np.zeros(200)
                position2 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time2)
            position_data.append(position2)
            theta_data.append(theta2)
            

            try:
                time3 = time_og[change_index[-1][-1]:]
                theta3 = theta_og[change_index[-1][-1]:]
                position3 = position_og[change_index[-1][-1]:]
                final_tg.append(1.5*gain)

            except IndexError:
                time3 = np.zeros(200)
                theta3 = np.zeros(200)
                position3 = np.zeros(200)
                final_tg.append(None)

            time_data.append(time3)
            position_data.append(position3)
            theta_data.append(theta3)
            

    return time_data, position_data, theta_data, final_tg
