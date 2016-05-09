import numpy as np
from openmdao.api import IndepVarComp, Component, Problem, Group
from openmdao.api import ScipyOptimizer
import time
		

#this component is to find the fraction of the all the rotors that are in the wakes of the other turbines
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
		self.add_param('alpha', val=np.tan(0.1))

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
			for j in range(nTurbs): #overlap_fraction[i][j] is the fraction of the area of turbine i in the wake from turbine j
				dx = x[i]-x[j]
				dy = abs(y[i]-y[j])
				dz = abs(z[i]-z[j])
				d = np.sqrt(dy**2+dz**2)
				R = r[j]+dx*alpha
				A = r[i]**2*np.pi
				overlap_area = 0
				if dx <= 0: #if turbine i is in front of turbine j
					overlap_fraction[i][j] = 0.0
				else:
					if d <= R-r[i]: #if turbine i is completely in the wake of turbine j
						if A <= np.pi*R**2: #if the area of turbine i is smaller than the wake from turbine j
							overlap_fraction[i][j] = 1.0
						else: #if the area of turbine i is larger than tha wake from turbine j
							overlap_fraction[i][j] = np.pi*R**2/A
					elif d >= R+r[i]: #if turbine i is completely out of the wake
						overlap_fraction[i][j] = 0.0
					else: #if turbine i overlaps partially with the wake
						overlap_area = r[i]**2.*np.arccos((d**2.+r[i]**2.-R**2.)/(2.0*d*r[i]))+R**2.*np.arccos((d**2.+R**2.-r[i]**2.)/(2.0*d*R))-0.5*np.sqrt((-d+r[i]+R)*(d+r[i]-R)*(d-r[i]+R)*(d+r[i]+R))
						overlap_fraction[i][j] = overlap_area/A
				
		print "Overlap Fraction Matrix: ", overlap_fraction
		unknowns['overlap'] = overlap_fraction


#slove for the effective wind velocity at each turbine
class effectiveVelocity(Component):
	

	def __init__(self, nTurbs):
		super(effectiveVelocity, self).__init__()
		
		self.fd_options['form'] = 'central'
		self.fd_options['step_size'] = 1.0e-6
		self.fd_options['step_type'] = 'relative'
		
		self.nTurbs = nTurbs
		self.add_param('xr', val=np.zeros(nTurbs))
		self.add_param('r', val=np.zeros(nTurbs))
		self.add_param('alpha', val=np.tan(0.1))
		self.add_param('windSpeed', val=0.0)
		self.add_param('a', val=1./3.)
		self.add_param('overlap', val=np.empty([nTurbs, nTurbs]))

		self.add_output('hubVelocity', val=np.zeros(nTurbs))

	def solve_nonlinear(self, params, unknowns, resids):
	
		
		x = params['xr']
		r = params['r']
		alpha = params['alpha']
		a = params['a']
		windSpeed = params['windSpeed']
		nTurbs = len(x)
		loss = np.zeros(nTurbs)
		hubVelocity = np.zeros(nTurbs)
		overlap = params['overlap']
	
		for i in range(nTurbs):
			for j in range(nTurbs):
				dx = abs(x[i]-x[j])
				loss[j] = overlap[i][j]*2.0*a*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
				loss[j] = loss[j]**2
			totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
			hubVelocity[i] = (1-totalLoss)*windSpeed #effective hub velocity
		unknowns['hubVelocity'] = hubVelocity


#Rotate the turbines to be in the reference frame of the wind
class rotate(Component):

    def __init__(self, nTurbs):
        super(rotate, self).__init__()

        self.add_param('x', val=np.zeros(nTurbs))
        self.add_param('y', val=np.zeros(nTurbs))
        self.add_param('windDir', val=0) #wind direction in radians

        self.add_output('xr', val=np.zeros(nTurbs))
        self.add_output('yr', val=np.zeros(nTurbs)) 

    def solve_nonlinear(self, params, unknowns, resids):
        x = params['x']
        y = params['y']
        windDir = params['windDir']
		
        x_r = x*np.cos(windDir)-y*np.sin(windDir)
        y_r = y*np.cos(windDir)+x*np.sin(windDir)
        unknowns['xr'] = x_r
        unknowns['yr'] = y_r

    def linearize(self, params, unknowns, resids):
        x = params['x']
        y = params['y']
        windDir = params['windDir']

        J = {}
        J['xr', 'x'] = np.cos(windDir)
        J['xr', 'y'] = -np.sin(windDir)
        J['xr', 'windDir'] = -x*np.sin(windDir)-y*np.cos(windDir)
        J['yr', 'x'] = np.sin(windDir)
        J['yr', 'y'] = np.cos(windDir)
        J['yr', 'windDir'] = -y*np.sin(windDir)+x*np.cos(windDir)

        return J

class Jensen(Group):
	#Group with all the components for the Jensen model

	def __init__(self, nTurbs):
		super(Jensen, self).__init__()
		
		self.add('f_1', rotate(nTurbs), promotes=['*'])
		self.add('f_2', wakeOverlap(nTurbs), promotes=['*'])
		self.add('f_3', effectiveVelocity(nTurbs), promotes=['*'])

if __name__=="__main__":

	# define turbine locations in global reference frame
	x = np.array([0,500,1000])
	y = np.array([0,0,0])
	z = np.array([150, 250, 350])
	
	# initialize input variable arrays
	nTurbs = np.size(x)
	rotorRadius = np.ones(nTurbs)*40.

	# Define flow properties
	windSpeed = 8.0
	windDir_deg = 8.5 #wind direction in degrees
	windDir = windDir_deg*np.pi/180. #Convert wind direction to radians

	#setup problem
	prob = Problem(root=Jensen(nTurbs))

	#initialize problem
	prob.setup()
    
	#assign values to parameters
	prob['x'] = x
	prob['y'] = y
	prob['z'] = z
	prob['r'] = rotorRadius
	prob['windSpeed'] = windSpeed
	prob['windDir'] = windDir

	#run the problem
	print 'start Jensen run'
	tic = time.time()
	prob.run()
	toc = time.time()

	#print the results
	print 'Time to run: ', toc-tic
	print 'Hub Velocity at Each Turbine: ', prob['hubVelocity']
