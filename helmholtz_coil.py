import numpy as np
from scipy.integrate import quad
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

# Define constant
mu0 = 4*pi*1e-7
r_min = 1e-3

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
        if r_3 < r_min**3:
            r_3 = r_min**3
        return np.array(self.I * mu0 / (4 * pi)  * np.cross ( self.tangential_vector(theta), self.separation_vector(r_field, theta) ) / r_3)

    def B(self, r_field):
        dBdthetai = lambda theta, i: self.dBdtheta(r_field, theta)[i]

        Bx = quad(dBdthetai, 0, 2*pi, args=(0,))[0]
        By = quad(dBdthetai, 0, 2*pi, args=(1,))[0]
        Bz = quad(dBdthetai, 0, 2*pi, args=(2,))[0]
        
        return np.array([Bx, By, Bz])
    
    def get_field(self, series_of_point_in_space):
        for point in series_of_point_in_space:
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
    
    def get_field(self):
        return np.array(self.B)
        
def generate_mesh(X, Y, Z):
    points = [point_in_space(x,y,z) for x in np.arange(X[0], X[1], X[2]) for y in np.arange(Y[0], Y[1], Y[2]) for z in np.arange(Z[0], Z[1], Z[2])]
    return points


coil1 = single_coil([0,0,0], 1, 1/mu0)
mesh = generate_mesh([-4,4,0.1], [-4,4,0.1], [0,1,1])
coil1.get_field(mesh)

X = [point.get_position()[0] for point in mesh]
Y = [point.get_position()[1] for point in mesh]
Bx = [point.get_field()[0] for point in mesh]
By = [point.get_field()[1] for point in mesh]

print np.shape(X), np.shape(Y), np.shape(Bx), np.shape(By)

plt.quiver(X,Y,Bx,By)
plt.show()
#print point1.get_position()

