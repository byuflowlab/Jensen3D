from __future__ import print_function, division, absolute_import
import time

import numpy as np

import openmdao.api as om


#this component is to find the fraction of the all the rotors that are in the wakes of the other turbines
class wakeOverlap(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")

    def setup(self):
        nTurbs = self.options['nTurbines']

        self.add_input('xr', val=np.zeros(nTurbs))
        self.add_input('yr', val=np.zeros(nTurbs))
        self.add_input('z', val=np.zeros(nTurbs))
        self.add_input('r', val=np.zeros(nTurbs))
        self.add_input('alpha', val=np.tan(0.1))

        self.add_output('overlap', val=np.eye(nTurbs))

        self.declare_partials('*', '*', method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs):
        nTurbs = self.options['nTurbines']

        x = inputs['xr']
        y = inputs['yr']
        z = inputs['z']
        r = inputs['r']
        alpha = inputs['alpha']

        overlap_fraction = np.eye(nTurbs, dtype=inputs._data.dtype)

        # i represents the index for the current turbine. j represents the index for other turbines.
        # Loops to find the overlap fraction of the jth turbine's wake on the ith turbine.
        for i in range(nTurbs):
            for j in range(nTurbs): #overlap_fraction[i][j] is the fraction of the area of turbine i in the wake from turbine j
                dx = x[i]-x[j]				# x-distance between ith and jth turbines.
                dy = abs(y[i]-y[j])			# y-distance between ith and jth turbines.
                dz = abs(z[i]-z[j])			# z-distance (elevation change) between ith and jth turbines.
                d = np.sqrt(dy**2+dz**2)	# resultant distance on y-z plane between ith and jth turbines.
                R = r[j]+dx*alpha			# radius of wake at distance dx downstream from jth turbine.
                A = r[i]**2*np.pi			# wind-swept area of ith turbine.

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

        print("Overlap Fraction Matrix: ", overlap_fraction)
        outputs['overlap'] = overlap_fraction


#slove for the effective wind velocity at each turbine
class effectiveVelocity(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")

    def setup(self):
        nTurbs = self.options['nTurbines']

        self.add_input('xr', val=np.zeros(nTurbs))
        self.add_input('r', val=np.zeros(nTurbs))
        self.add_input('alpha', val=np.tan(0.1))
        self.add_input('windSpeed', val=0.0)
        self.add_input('a', val=1./3.)
        self.add_input('overlap', val=np.empty([nTurbs, nTurbs]))

        self.add_output('hubVelocity', val=np.zeros(nTurbs))

        self.declare_partials('*', '*', method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs):
        nTurbs = self.options['nTurbines']

        x = inputs['xr']
        r = inputs['r']
        alpha = inputs['alpha']
        a = inputs['a']
        windSpeed = inputs['windSpeed']
        overlap = inputs['overlap']

        loss = np.zeros(nTurbs, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbs, dtype=inputs._data.dtype)

        for i in range(nTurbs):
            for j in range(nTurbs):
                dx = abs(x[i]-x[j])
                loss[j] = overlap[i][j]*2.0*a*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
                loss[j] = loss[j]**2
            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1-totalLoss)*windSpeed #effective hub velocity
        outputs['hubVelocity'] = hubVelocity

        """def linearize(self, params, unkowns, resids):

        x = params['xr']
		r = params['r']
		alpha = params['alpha']
		a = params['a']
		windSpeed = params['windSpeed']
		nTurbs = len(x)
		loss = np.zeros(nTurbs)
		hubVelocity = np.zeros(nTurbs)
		overlap = params['overlap']

        J = {}
        J['hubVelocity', 'r'] =
        J['hubVelocity', 'xr'] =
        J['hubVelocity', 'alpha'] =
        J['hubVelocity', 'a'] =
        J['hubVelocity', """


#Rotate the turbines to be in the reference frame of the wind
class rotate(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")

    def setup(self):
        nTurbs = self.options['nTurbines']

        self.add_input('x', val=np.zeros(nTurbs))
        self.add_input('y', val=np.zeros(nTurbs))
        self.add_input('windDir', val=0) #wind direction in radians

        self.add_output('xr', val=np.zeros(nTurbs))
        self.add_output('yr', val=np.zeros(nTurbs))

        self.declare_partials('*', '*', method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        windDir = inputs['windDir']

        x_r = x*np.cos(windDir) - y*np.sin(windDir)
        y_r = y*np.cos(windDir) + x*np.sin(windDir)

        outputs['xr'] = x_r
        outputs['yr'] = y_r

    """def linearize(self, params, unknowns, resids):
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

        return J"""


class Jensen(om.Group):
    #Group with all the components for the Jensen model

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")

    def setup(self):
        nTurbs = self.options['nTurbines']

        ivc = self.add_subsystem('des_var', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', val=np.zeros(nTurbs))
        ivc.add_output('y', val=np.zeros(nTurbs))
        ivc.add_output('z', val=np.zeros(nTurbs))
        ivc.add_output('r', val=np.zeros(nTurbs))
        ivc.add_output('windSpeed', val=0.0)
        ivc.add_output('windDir', val=0.0)

        self.add_subsystem('f_1', rotate(nTurbines=nTurbs), promotes=['*'])
        self.add_subsystem('f_2', wakeOverlap(nTurbines=nTurbs), promotes=['*'])
        self.add_subsystem('f_3', effectiveVelocity(nTurbines=nTurbs), promotes=['*'])


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
    prob = om.Problem(model=Jensen(nTurbines=nTurbs))

    prob.driver = om.ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['optimizer'] = 'SNOPT'

    prob.model.add_design_var('x', lower=np.ones(nTurbs)*0, upper=np.ones(nTurbs)*1000)
    prob.model.add_design_var('y', lower=np.ones(nTurbs)*0, upper=np.ones(nTurbs)*1000)
    prob.model.add_design_var('z', lower=np.ones(nTurbs)*50, upper=np.ones(nTurbs)*150)
    prob.model.add_objective('hubVelocity')

    prob.setup()

    #assign values to parameters
    prob['x'] = x
    prob['y'] = y
    prob['z'] = z
    prob['r'] = rotorRadius
    prob['windSpeed'] = windSpeed
    prob['windDir'] = windDir

    #run the problem
    print('start Jensen run')
    tic = time.time()
    prob.run_model()
    toc = time.time()

    #print the results
    print('Time to run: ', toc-tic)
    print('Hub Velocity at Each Turbine: ', prob['hubVelocity'])
