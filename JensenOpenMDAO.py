from __future__ import print_function
from math import tan, pi, sqrt, acos, cos, sin
import numpy as np
from openmdao.api import IndepVarComp, Component, Problem, Group
from openmdao.api import ScipyOptimizer
import time
		


class wakeOverlap(Component):
	

	def __init__(self, nTurbs):
		super(wakeOverlap, self).__init__()

		self.fd_options['form'] = 'central'
		self.fd_options['step_size'] = 1.0e-6
		self.fd_options['step_type'] = 'relative'

		self.nTurbs = nTurbs
		self.add_param('xr', val=np.zeros(nTurbs))
		self.add_param('yr', val=np.zeros(nTurbs))
		self.add_param('z', val=np.zeros(nTurbs))
		self.add_param('r', val=np.zeros(nTurbs))
		self.add_param('alpha', val=tan(0.1))

		self.add_output('overlap', val=np.eye(nTurbs))

	def solve_nonlinear(self, params, unknowns, resids):
		
		x = params['xr']
		y = params['yr']
		z = params['z']
		r = params['r']
		alpha = params['alpha']
		nTurbs = len(x)
		
		overlap_fraction = np.eye(nTurbs)
		for i in range(nTurbs):
			for j in range(nTurbs):
				dx = x[i]-x[j]
				dy = abs(y[i]-y[j])
				dz = abs(z[i]-z[j])
				d = sqrt(dy**2+dz**2)
				R = r[j]+dx*alpha
				A = r[i]**2*pi
				
				if dx <= 0:
					overlap_fraction[i][j] = 0.0
				else:
					if d <= R-r[i]:
						if A <= pi*R**2:
							overlap_fraction[i][j] = 1.0
						else: 
							overlap_fraction[i][j] = pi*R**2/A
					elif d >= R+r[i]:
						overlap_fraction[i][j] = 0.0
					else:
						overlap_area = r[i]**2.*acos((d**2.+r[i]**2.-R**2.)/(2.0*d*r[i]))+R**2.*aos((d**2.+R**2.-r[i]**2.)/(2.0*d*R))-0.5*sqrt((-d+r[i]+R)*(d+r[i]-R)*(d-r[i]+R)*(d+r[i]+R))
                overlap_fraction[i][j] = overlap_area/A
				
				
		unknowns['overlap'] = overlap_fraction



class jensenPower(Component):
	

	def __init__(self, nTurbs):
		super(jensenPower, self).__init__()
		
		self.fd_options['form'] = 'central'
		self.fd_options['step_size'] = 1.0e-6
		self.fd_options['step_type'] = 'relative'
		
		self.nTurbs = nTurbs
		self.add_param('xr', val=np.zeros(nTurbs))
		self.add_param('r', val=np.zeros(nTurbs))
		self.add_param('alpha', val=tan(0.1))
		self.add_param('windSpeed', val=0.0)
		self.add_param('rho', val=0.0)
		self.add_param('a', val=1./3.)
		self.add_param('Cp', val=4.*a*(1-a)**2.)
		self.add_param('overlap', val=np.empty([nTurbs, nTurbs]))

		self.add_output('Power', val=np.zeros(nTurbs))

	def solve_nonlinear(self, params, unknowns, resids):
	
		x = params['xr']
		r = params['r']
		alpha = params['alpha']
		a = params['a']
		windSpeed = params['windSpeed']
		self.add_param('rho', val=1.1716)
		Cp = params['Cp']
		nTurbs = len(x)
		loss = np.zeros(nTurbs)
		power = np.zeros(nTurbs)
		overlap = params['overlap']
	
		for i in range(nTurbs):
			for j in range(nTurbs):
				dx = x[i]-x[j]
				loss[j] = overlap[i][j]*2.0*a*(r[j]/(r[j]+alpha*dx))**2
				loss[j] = loss[j]**2
			totalLoss = sqrt(np.sum(loss))
			effectiveVelocity = (1-totalLoss)*windSpeed
			A = pi*r[i]**2
			power[i] = 0.5*rho*A*Cp*effectiveVelocity**3
		unknowns['Power'] = np.sum(power)


#Rotate the turbines to be in the reference frame of the wind
class rotate(Component):

	def __init__(self, nTurbs):
		super(rotate, self).__init__()
		
		self.fd_options['form'] = 'central'
		self.fd_options['step_size'] = 1.0e-6
		self.fd_options['step_type'] = 'relative'
		
		self.add_param(['x'], val=np.zeros(nTurbs))
		self.add_param(['y'], val=np.zeros(nTurbs))
		self.add_param(['windDir'], val=0) #wind direction in radians
		
		self.add_output(['xr'], val=np.zeros(nTurbs))
		self.add_output(['yr'], val=np.zeros(nTurbs)) 

	def solve_nonlinear(self, params, unknowns, resids):
		x = params['x']
		y = params['y']
		windDir = params['windDir']
		
        x_r = x*cos(windDir)-y*sin(windDir)
        y_r = y*cos(windDir)+x*sin(windDir)
        xr = np.zeros(nTurbs)
        yr = np.zeros(nTurbs)
        xr[:] = x_r
        yr[:] = y_r
        unknowns['xr'] = xr
        unknowns['yr'] = yr

class Jensen(Group):
	"""Group with all the components for the Jensen model"""

	def __init__(self, nTurbs):
		super(Jensen, self).__init__()
		
		self.add('f_1', rotate(nTurbs), promotes=['*'])
		self.add('f_2', wakeOverlap(nTurbs), promotes=['*'])
		self.add('f_3', jensenPower(nTurbs), promotes=['*'])

if __name__=="__main__":

    # define turbine locations in global reference frame
    x = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    y = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
    z = np.array([150, 150, 150, 150, 150, 150])
	
    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorRadius = np.ones(nTurbs)*40.

    # Define flow properties
    windSpeed = 8.0
    windDir_deg = 270. #wind direction in degrees
    windDir = windDir_deg*pi/180. #Convert wind direction to radians

    #setup problem
    prob = Problem(root=Jensen(nTurbs))

    #initialize problem
    prob.setup()
    
    #assign values to parameters
    prob['x'] = turbineX
    prob['y'] = turbineY
    prob['z'] = turbineZ
    prob['windSpeed'] = windSpeed
    prob['windDir'] = windDir

    #run the problem
    print('start Jensen run')
    tic = time.time()
    prob.run()
    toc = time.time()

    #print the results
    print('Time to run: ', toc-tic)
    print ('Wind Farm Power: ', prob['Power'])
