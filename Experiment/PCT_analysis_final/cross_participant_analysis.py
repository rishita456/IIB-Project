import numpy as np

P_ids = [0,1,2,3,4,6,7,8,9]
phases = [1,2,3]

for P_id in P_ids:
    for phase in phases:

        Kps = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kps')
        Kds = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Kds') 
        Taus = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Taus')
        Gains = np.load('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/PCT_analysis_final/data/participant' + str(P_id) + '/phase' + str(phase) + '/Gains')
        
                  