import numpy as np
from scipy.integrate import quad
from numpy import pi, sin, cos

# Define constant
mu0 = 4*pi*1e-7

class single_coil:
    def __init__(self, r, R, I):
        self.r = r
        self.R = R
        self.I = I    
        
    def tangential_vector(self, theta):
        return np.array([0, -sin(theta), cos(theta)]) * self.R  
    
    def radial_vector(self, theta):
        return np.array([0,  cos(theta), sin(theta)]) * self.R
    
    def separation_vector(self, r_field, theta):
        return r_field - ( self.r + self.radial_vector(theta))  
    
    def dBdtheta(self, r_field, theta):
        r = self.separation_vector(r_field, theta)
        r_3 = np.power(np.dot(r,r), 1.5)
        return np.array(self.I * mu0  * np.cross ( self.tangential_vector(theta), self.separation_vector(r_field, theta) ) / r_3)

    def B(self, r_field):
        dBdthetai = lambda theta, i: self.dBdtheta(r_field, theta)[i]

        Bx = quad(dBdthetai, 0, 2*pi, args=(0,))
        By = quad(dBdthetai, 0, 2*pi, args=(1,))
        Bz = quad(dBdthetai, 0, 2*pi, args=(2,))
        
        return np.array([Bx, By, Bz])
    
    def get_field(self, series_of_point_in_space):
        for point in series_of_point_in_space:
            #self.B(point.get_position())
            point.set_field( self.B(point.get_position()) )         
            
class point_in_space:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def set_field(self, B):
        self.B = B
        
    def get_position(self):
        return np.array([self.x, self.y, self.z])
        

coil1 = single_coil([0,0,0], 0.1, 0.1)
point1 = point_in_space(0,0,0)
coil1.get_field([point1])

#print point1.get_position()

