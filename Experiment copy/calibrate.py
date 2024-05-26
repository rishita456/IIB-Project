import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from multiprocessing import shared_memory
import pandas as pd
import numpy as np
import config
import quaternions_exp as quaternions_exp
import datetime


def get_data(varsPerDataType,noDataTypes,sharedMemoryName,frameLength = 1000,sim = False,idxesToPlot = None, rotation = False, transform = None):

    if sim == False:
        dataEntries = varsPerDataType * noDataTypes
        SHARED_MEM_NAME = sharedMemoryName
        shared_block = shared_memory.SharedMemory(size= dataEntries * 8, name=SHARED_MEM_NAME, create=False)
        shared_array = np.ndarray(shape=(noDataTypes,varsPerDataType), dtype=np.float64, buffer=shared_block.buf)
        if idxesToPlot is not None:
            df = pd.DataFrame(shared_array[idxesToPlot])
        else:
            df = pd.DataFrame(shared_array)
    else:
        shared_array = np.random.randint(0,5,size = (43,7))
    # load the most recent shared memory onto a dataframe
    
    # we will get structure of database as rigidBody1 - [q_X,q_Y,q_Z,q_W,X,Y,Z]
    if sim == False:
        df_locations = [df.iloc[a,:] for a in range(0,df.shape[0])] # split rows into list
        df_locations_quaternionObjs = [quaternions_exp.quaternionVector(loc = [a.iloc[4]/1000,a.iloc[5]/1000,a.iloc[6]/1000],quaternion=[a.iloc[3],a.iloc[0],a.iloc[1],a.iloc[2]]) for a in df_locations]
        df_directions =  pd.DataFrame([q.qv_mult(q.quaternion,[0,-1,0]) for q in df_locations_quaternionObjs])
        dfPlot = pd.DataFrame({'x':df.iloc[:,4],'y':df.iloc[:,5],'z':df.iloc[:,6],'dirX':df_directions.iloc[:,0],'dirY':df_directions.iloc[:,1],'dirZ':df_directions.iloc[:,2]})
       
    print(dfPlot)
    print('retrieved data from shared memory - ', datetime.datetime.now())
    if rotation ==False:
        dir = np.zeros((3,1))
        # dir[0,0] = dfPlot.iloc[:,3]-dfPlot.iloc[:,0] 
        # dir[1,0] = dfPlot.iloc[:,4]-dfPlot.iloc[:,1]
        # dir[2,0] = dfPlot.iloc[:,5]-dfPlot.iloc[:,2]
        return dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5], dir
    
    else:
        vector = [dfPlot.iloc[:,3],dfPlot.iloc[:,4],dfPlot.iloc[:,5]]
        rotated_vector = np.matmul(transform, vector)
        print(rotated_vector)

        return rotated_vector[0], rotated_vector[1], rotated_vector[2], 0
        

    
  # correct rotations TO DO

        



class Calibrate:

    def __init__(self):
        self.origin = [0,0,0]
        self.left_scale = 1
        self.right_scale = 1
        self.dist = 0
        self.left_dir = -1
        self.right_dir = 1
        self.previous_point = [0,0]
        self.transform = np.zeros((3,3))
        self.Rotate = False
        self.calibration_end = False


    def calibrate_origin(self):
        
        x,y,z, origin_dir = get_data(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26])
        self.origin = [float(x.iloc[0]), float(y.iloc[0]), float(z.iloc[0])]
        if self.Rotate ==True:
            angle = np.cosh(np.dot(np.transpose(origin_dir), np.transpose([[1,0,0]]))/np.linalg.norm(origin_dir))
            self.transform[0] = np.array([np.cos(angle[0,0]), -np.sin(angle[0,0]), 0] )
            self.transform[1] = np.array([np.sin(angle[0,0]), np.cos(angle[0,0]), 0])
            self.transform[2] = np.array([0, 0, 1])
        self.previous_point = [self.origin[0], self.origin[1]]
        print(self.origin)
        return self.origin[1]
        print(self.origin)

    def calibrate_xy_plane_left(self):

        x_l, y_l, z_l, dir = get_data(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26], rotation= self.Rotate, transform=self.transform)
        left_end = [float(x_l.iloc[0]), float(y_l.iloc[0])]
        centre = [self.origin[0], self.origin[1]]
        self.left_dir = np.sign(centre[config.COORD] - left_end[config.COORD])

        self.left_scale = self.left_dir * ((config.WIDTH * 0.5) / math.dist(left_end, centre))



    def calibrate_xy_plane_right(self):

        x_r, y_r, z_r, dir = get_data(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26], rotation= self.Rotate, transform=self.transform)
        right_end = [float(x_r.iloc[0]), float(y_r.iloc[0])]
        centre = [self.origin[0], self.origin[1]]
        self.right_dir = np.sign(centre[config.COORD] - right_end[config.COORD])

        self.right_scale = self.right_dir * ((config.WIDTH * 0.5) / math.dist(right_end, centre))
        print(self.right_scale)
        print(self.left_scale)


    def get_dist(self):

        x, y, z, dir = get_data(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26], rotation= self.Rotate, transform=self.transform)
        print("mocap input recieved - " , pygame.time.get_ticks(), ' ', datetime.datetime.now())
        current_point = [float(x.iloc[0]), float(y.iloc[0])]
        if np.sign(self.previous_point[0] - current_point[0]) == self.left_dir:
            self.dist = self.left_scale*math.dist(self.previous_point, current_point)
        if np.sign(self.previous_point[0] - current_point[0]) == self.right_dir:
            self.dist = self.right_scale*math.dist(self.previous_point, current_point)
        self.previous_point = current_point
        print(self.dist)
        
        return -self.dist
    
    def get_current_point(self):
        offset = (config.WIDTH - config.CART_WIDTH) // 2
        x, y, z, dir = get_data(7,51,sharedMemoryName= 'Test Rigid Body',idxesToPlot = [26], rotation= self.Rotate, transform=self.transform)
        print("mocap input recieved - " , pygame.time.get_ticks(), ' ', datetime.datetime.now())
        current_point = [float(x.iloc[0]), float(y.iloc[0])]
        print(current_point)
        if np.sign(self.origin[config.COORD] - current_point[config.COORD]) == self.left_dir:
            self.dist = self.left_scale*abs(self.origin[config.COORD] - current_point[config.COORD])
        if np.sign(self.origin[config.COORD] - current_point[config.COORD]) == self.right_dir:
            self.dist = self.right_scale*abs(self.origin[config.COORD] - current_point[config.COORD])
        self.previous_point = current_point
        self.dist = config.COORD_SIGN * self.dist
        print(self.dist)
       
        
        return offset + (self.dist), current_point[config.COORD]

        return current_point[0]
