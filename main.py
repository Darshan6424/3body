import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pylot as plt

#solutions initial conditions

y01 = []
y02 = []
y03 = []
y04 = []
y05 = []
y06 = []
y07 = []
y08 = []

solutionList = [y01,y02,y03,y04,y05,y06,y07,y08]
inp = int(input("Choose a solution from 1 to 8"))
y0 = solutionList[inp - 1]

# The masses of the 4 bodies

m1 = m2 = m3 = 1
m4 = 0.0053

#Time thingy

N= 670000
T = 0.001

#Defining the fucntion

def ThreeBodyProb(t,y,m1,m2,m3,m4):
    f = np.zeros(16)
    #velocities of the 3bodies
    f[0] = y[8]
    f[1] = y[9]
    f[2] = y[10]
    f[3] = y[11]
    f[4] = y[12]
    f[5] = y[13]
    f[6] = y[14]
    f[7] = y[15]

    #now calculating the acceleration for each body respectively
    f[8] = -m2*(y[0]-y[2]) / (((y[0]-y[2])**2 + (y[1]-y[3])**2)**(3/2)) - m3 * (y[0]-y[4]) / (((y[0]-y[4]**2 + ())))














def VectorStability(g):
    x = 50
    distance = []
    if g=='m':
        for in in stable_masses:
            used_mass = i
            














