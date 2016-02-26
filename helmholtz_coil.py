import numpy as np
from scipy.integrate import quad
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from time import time
from scipy.integrate import simps

# Define constant
mu0 = 4*pi*1e-7


### Define a single Helmholtz coil object
class single_coil:
    def __init__(self, r, R, I, radius_of_wire=1e-3):
        self.r = r
        self.R = R
        self.I = I    
        self.r_min = radius_of_wire
# Direction of dl        
    def tangential_vector(self, theta):
        return np.array([0, -sin(theta), cos(theta)]) * self.R  
# Position of current elements
    def radial_vector(self, theta):
        return np.array([0,  cos(theta), sin(theta)]) * self.R
# Seperation vector from field point to source point.     
    def separation_vector(self, r_field, theta):
        return r_field - ( self.r + self.radial_vector(theta))  
# Expression for dB/dtheta    
    def dBdtheta(self, r_field, theta):
        r = self.separation_vector(r_field, theta)
        r_3 = np.power(np.dot(r,r), 1.5)
        
        # Do not consider field within a certain radius of sphere near the wire
        r_min = self.r_min
        if r_3 < r_min**3:
            r_3 = r_min**3
        
        return np.array(self.I * mu0 / (4 * pi)  * np.cross ( self.tangential_vector(theta), self.separation_vector(r_field, theta) ) / r_3)
### Computational algorithms

# Slower version using adaptive quadrature method. Due to restriction of quad, need to evaluate cross product 3 times.
    def adaptive_B(self, r_field):
        dBdthetai = lambda theta, i: self.dBdtheta(r_field, theta)[i]

        Bx = quad(dBdthetai, 0, 2*pi, args=(0,))[0]
        By = quad(dBdthetai, 0, 2*pi, args=(1,))[0]
        Bz = quad(dBdthetai, 0, 2*pi, args=(2,))[0]
        
        return np.array([Bx, By, Bz])

# Faster version, using simpson's rule in estimation.
    def faster_B(self, r_field):        
        N = 10
        theta_range = np.linspace(0, 2*pi, N)
         
        dB = np.array([self.dBdtheta(r_field, theta) for theta in theta_range])

        dBx = dB[:,0]
        dBy = dB[:,1]
        dBz = dB[:,2]
         
        Bx =  simps(dBx,theta_range)
        By =  simps(dBy,theta_range)
        Bz =  simps(dBz,theta_range)
 
        return np.array([Bx, By, Bz])                 
    
# Vectorized version. This is very memory iefficient, though vectorized. Not used
    def v_B(self,series_of_point_in_space):
        r_field = np.array([point.get_position() for point in series_of_point_in_space])
        N = 10
        dtheta = np.linspace(0, 2*pi, N)
        Idl = self.I * np.array([[0, -sin(theta), cos(theta)] for theta in dtheta for ii in r_field])
        r_coil = self.R * np.array([[0, cos(theta), sin(theta)] for theta in dtheta for ii in r_field])
        r = [r_field_point - r_coil_point - self.r for r_field_point in r_field for r_coil_point in r_coil]
        r_3 = np.power(np.dot(r,r),1.5)
        
        Idlcrossr = np.cross(Idl,r)
        
        dB = mu0 / (4 * pi) * np.divide(Idlcrossr, r_3)
        
        return np.array(simps(dB, dtheta))         
        
### Driver for these algorithms        
# Depending on how close to the coil, use different algorithms
    def choose_algorithm(self,r_field):
        x = r_field[0]
        x_distance_to_coil = np.abs(x - self.r[0])
        # Within the length scale of the coil, field rapidly changing with theta, need adaptive routine
        threshold = self.R
        if x_distance_to_coil < threshold:
            return self.adaptive_B
        else:
            return self.faster_B 
                     
    def add_field(self, series_of_point_in_space):
        for point in series_of_point_in_space:
            point.add_field( self.choose_algorithm(point.get_position())(point.get_position()) )      

## If vectorized code is to be used.
#         B = self.v_B(series_of_point_in_space)
#         for ii in range(0, len(series_of_point_in_space)):
#             series_of_point_in_space[ii].add_field(B[ii,:])



## Defining a vector at the point.        
class point_in_space:
    
    # Consisting of a position vector r = [x, y, z]
    # and a magnetic flux density vector B = [Bx, By, Bz]
    
    def __init__(self, x, y, z):
        # Location of the point
        self.x = x
        self.y = y
        self.z = z
        # Initialise field to zero at creation
        self.B = np.zeros(3)
        
    def add_field(self, B):
        # Superimpose a field
        self.B = self.B + B
    
    def get_position(self):
        return np.array([self.x, self.y, self.z])
    
    def get_field(self):
        return np.array(self.B)
        
### Auxillary functions for easy use     
def generate_mesh(X, Y, Z):
    # Generating a list of point object, with locations given by, X, Y, Y where
    # X = [xstart, xend, xstep]
    
    # if any of these coordinates is a one-element array, 
    if len(X) == 1:
        X = [X[0], X[0] + 1, 1]
    if len(Y) == 1:
        Y = [Y[0], Y[0] + 1, 1]
    if len(Z) == 1:
        Z = [Z[0], Z[0] + 1, 1]
    
    # Generate mesh    
    points = [point_in_space(x,y,z) for x in np.arange(X[0], X[1], X[2]) for y in np.arange(Y[0], Y[1], Y[2]) for z in np.arange(Z[0], Z[1], Z[2])]
    return points
    
### Comparing with theoretical field on axis
def theoretical_on_axis(I,r,R,x):
    return mu0 * I * R**2 / ( 2 * np.power((R**2 + (x-r[0]) **2),1.5) )
# A driver for running the test
def test_on_axis():
    I = 1/mu0
    R = 0.01
    r = [0,0,0]
    coil = single_coil(r, R, I)
    points = generate_mesh([-3,3,0.01], [0,1,1], [0,1,1])
    coil.add_field(points)    
    X = [point.get_position()[0] for point in points]
    Bx = [point.get_field()[0] for point in points]
    Bx_theory = [theoretical_on_axis(I,r,R,x) for x in X]
    # Plot field
    plt.plot(X, Bx, X, Bx_theory)
    plt.show()
    # Plot error
    plt.plot(X, np.array(Bx_theory) - np.array(Bx))
    plt.show()
    
# test_on_axis()    


# Performing calculation
t1 = time() # 0.000000016
## Building coils
coil1 = single_coil([-1,0,0], 1, 1/mu0)
coil2 = single_coil([ 1,0,0], 1, -1/mu0)
t2 = time() # 0.0000024
## Generate mesh
mesh = generate_mesh([-4,4,0.1], [-2,2,0.1], [0])
t3 = time() # 0.018
## Superimposing field 
coil1.add_field(mesh)
coil2.add_field(mesh)
t4 = time() # 0.000004


## Plotting
X = [point.get_position()[0] for point in mesh]
Y = [point.get_position()[1] for point in mesh]
Bx = [point.get_field()[0] for point in mesh]
By = [point.get_field()[1] for point in mesh]
t5 = time()
# Quiver plot
plt.quiver(X,Y,Bx,By)

## Axial plot
#plt.plot(X,Bx)
plt.show()

## Returning timing info per point in space on average.
N = len(mesh)
print np.array([t2 - t1, t3 - t2, t4 - t3, t5 - t4])/N/2


