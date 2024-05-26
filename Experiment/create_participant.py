import os

import numpy as np

def create(id):
    gain0s = np.array([2, 5, 7, 10, 12, 15, 17, 20, 22, 25])
    fields = ['/time', '/position', '/angle', '/raw_position']
    participant_id = id
    trial_gains = np.ones((3, 10))
    write = True
    try:

        path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id)
        os.makedirs(path)

        path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/info.txt'
        file = open(path, 'a')
        file.close()
        for field in fields:
            path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/test_trials/85sec_trial' + field
            os.makedirs(path)
        for field in fields:
            path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/test_trials/84sec_trial' + field
            os.makedirs(path)
        for field in fields:
            path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/test_trials/83sec_trial' + field
            os.makedirs(path)
        for field in fields:
            path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/test_trials/82sec_trial' + field
            os.makedirs(path)

    except FileExistsError:
        pass
    
    for disturbance in range(3):

        try:

            if disturbance == 0:
                trial_gains[0] = gain0s
                trials = 2
                for trial in range(trials): 
                    for gain in gain0s:
                        for field in fields:
                            path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/0_disturbance/g0_' + str(gain) + '/trial' + str(trial + 1) + field
                            os.makedirs(path)
            

            if disturbance == 1:
                trials = 10
                gain0 = 1
                gain1s = np.random.randint(low=10, high=20, size = trials)
                trial_gains[1] = gain1s
                unique, counts = np.unique(gain1s, return_counts=True)
                print(gain1s)

                for n in unique:
                    indexes = np.where(gain1s == n)
                    for i in indexes:

                        try:
                            for trial in range (counts[np.where(unique==n)][0]):
                                for field in fields:
                                    path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/1_disturbance/g0_' + str(gain0) + '_g1_' + str(n) + '/trial' + str(trial + 1) + field
                                    os.makedirs(path)

                        except FileExistsError as e:
                            print(e)

            if disturbance == 2:
                trials = 10
                gain0 = 1
                gain1s = np.random.randint(low=10, high=20, size = trials)
                trial_gains[2] = gain1s
                unique, counts = np.unique(gain1s, return_counts=True)
                print(gain1s)

                for n in unique:
                    indexes = np.where(gain1s == n)
                    for i in indexes:

                        try:
                            for trial in range (counts[np.where(unique==n)][0]):
                                for field in fields:
                                    path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/2_disturbance/g0_' + str(gain0) + '_g1_' + str(n) + '_g2_' + str(n*1.5) + '/trial' + str(trial + 1) + field
                                    os.makedirs(path)

                        except FileExistsError as e:
                            print(e)

            if disturbance == 3:
                trials = 10
                gain0 = 1
                gain1s = np.random.randint(low=10, high=20, size = trials)
                trial_gains[3] = gain1s
                unique, counts = np.unique(gain1s, return_counts=True)
                print(gain1s)

                for n in unique:
                    indexes = np.where(gain1s == n)
                    for i in indexes:

                        try:
                            for trial in range (counts[np.where(unique==n)][0]):
                                for field in fields:
                                    path = '/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/participant_' + str(participant_id) + '/3_disturbance/g0_' + str(gain0) + '_g1_' + str(n) + '_g2_' + str(n*1.5) + '_g3_' + str(n*2) + '/trial' + str(trial + 1) + field
                                    os.makedirs(path)

                        except FileExistsError as e:
                            print(e)

        except FileExistsError as e:
            write = False
            break


    if write == True: 

        str_gains = np.array2string(trial_gains)
        txt = str_gains.replace('.', ',')
        txt = txt.replace(']', '],')
        txt = txt.replace(',]', ']')
        txt = txt.replace(']],', ']]')

        f = open("/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/config.py", "a")
        f.write('\nTRIAL_GAINS_' + str(participant_id) +  '=' + txt + '\n' + 'TRIAL_INDEX_' + str(participant_id) + ' = 0')
        f.close()

    elif write == False:

        lines = open('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/config.py', 'r').readlines()
        txt = lines[-1].split('= ')
        txt = int(txt[-1])
        txt = txt + 1
        txt = str(txt)
        lines[-1] = 'TRIAL_INDEX_' + str(participant_id) + ' = ' + txt + '\n'
        open('/Users/rishitabanerjee/Desktop/ProjectStuff/BrainMachineInterfaces/Experiment/config.py', 'w').writelines(lines)
                        
    return trial_gains

create(11) # enter idol