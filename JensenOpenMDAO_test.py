import numpy as np
from openmdao.api import IndepVarComp, Component, Problem, Group
from openmdao.api import ScipyOptimizer
from florisse.GeneralWindFarmComponents import WindFrame

import time
        

#this component is to find the fraction of the all the rotors that are in the wakes of the other turbines
class wakeOverlap(Component):
    
    def __init__(self, nTurbines, direction_id=0):
        super(wakeOverlap, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.add_param('turbineXw', val=np.zeros(nTurbines))
        self.add_param('turbineYw', val=np.zeros(nTurbines))
        self.add_param('turbineZ', val=np.zeros(nTurbines))
        self.add_param('rotorDiameter', val=np.zeros(nTurbines))

        #unused but required for compatibility
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines))
        self.add_param('hubHeight', np.zeros(nTurbines))
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines))
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines))
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_param('Ct', np.zeros(nTurbines))


        self.add_param('model_params:alpha', val=np.tan(0.1))

        self.add_output('overlap', val=np.eye(nTurbines))

    def solve_nonlinear(self, params, unknowns, resids):
        
        turbineX = params['turbineXw']
        turbineY = params['turbineYw']
        turbineZ = params['turbineZ']
        r = 0.5*params['rotorDiameter']
        alpha = params['model_params:alpha']
        nTurbines = self.nTurbines
        
        overlap_fraction = np.eye(nTurbines)
        for i in range(nTurbines):
            for j in range(nTurbines): #overlap_fraction[i][j] is the fraction of the area of turbine i in the wake from turbine j
                dx = turbineX[i]-turbineX[j]
                dy = abs(turbineY[i]-turbineY[j])
                dz = abs(turbineZ[i]-turbineZ[j])
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
                
        # print "Overlap Fraction Matrix: ", overlap_fraction
        unknowns['overlap'] = overlap_fraction


#slove for the effective wind velocity at each turbine
class effectiveVelocity(Component):
    

    def __init__(self, nTurbines, direction_id=0):
        super(effectiveVelocity, self).__init__()
        
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True
        
        self.nTurbines = nTurbines
        self.direction_id = direction_id

        self.add_param('turbineXw', val=np.zeros(nTurbines))
        self.add_param('rotorDiameter', val=np.zeros(nTurbines))
        self.add_param('model_params:alpha', val=np.tan(0.1))
        self.add_param('wind_speed', val=0.0)
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)
        self.add_param('overlap', val=np.empty([nTurbines, nTurbines]))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines))

    def solve_nonlinear(self, params, unknowns, resids):
    
        
        turbineX = params['turbineXw']
        r = 0.5*params['rotorDiameter']
        alpha = params['model_params:alpha']
        a = params['axialInduction']
        windSpeed = params['wind_speed']
        nTurbines = self.nTurbines
        direction_id = self.direction_id
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)
        overlap = params['overlap']
    
        for i in range(nTurbines):
            for j in range(nTurbines):
                dx = abs(turbineX[i]-turbineX[j])
                loss[j] = overlap[i][j]*2.0*a[i]*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
                loss[j] = loss[j]**2
            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1-totalLoss)*windSpeed #effective hub velocity
        unknowns['wtVelocity%i' % direction_id] = hubVelocity


#Rotate the turbines to be in the reference frame of the wind
class rotate(Component):

    def __init__(self, nTurbs):
        super(rotate, self).__init__()
    
        self.add_param('turbineX', val=np.zeros(nTurbs))
        self.add_param('turbineY', val=np.zeros(nTurbs))
        self.add_param('windDir', val=0) #wind direction in radians
    
        self.add_output('turbineXw', val=np.zeros(nTurbs))
        self.add_output('turbineYw', val=np.zeros(nTurbs)) 

    def solve_nonlinear(self, params, unknowns, resids):
        turbineX = params['turbineX']
        turbineY = params['turbineY']
        windDir = params['windDir']
        
        x_r = turbineX*np.cos(windDir)-turbineY*np.sin(windDir)
        y_r = turbineY*np.cos(windDir)+turbineX*np.sin(windDir)
        unknowns['turbineXw'] = x_r
        unknowns['turbineYw'] = y_r
    

    def linearize(self, params, unknowns, resids):
        turbineX = params['turbineX']
        turbineY = params['turbineY']
        windDir = params['windDir']

        J = {}
        J['turbineXw', 'turbineX'] = np.cos(windDir)
        J['turbineXw', 'turbineY'] = -np.sin(windDir)
        J['turbineXw', 'windDir'] = -turbineX*np.sin(windDir)-turbineY*np.cos(windDir)
        J['turbineYw', 'turbineX'] = np.sin(windDir)
        J['turbineYw', 'turbineY'] = np.cos(windDir)
        J['turbineYw', 'windDir'] = -turbineY*np.sin(windDir)+turbineX*np.cos(windDir)

        return J


class Jensen(Group):
    #Group with all the components for the Jensen model

    def __init__(self, nTurbs, direction_id=0, model_options=None):
        super(Jensen, self).__init__()

        self.add('f_2', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
        self.add('f_3', effectiveVelocity(nTurbs, direction_id=direction_id), promotes=['*'])


if __name__=="__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([0, 100, 200])
    turbineY = np.array([0, 0, 0])
    turbineZ = np.array([150, 150, 150])
    
    # initialize input variable arrays
    nTurbs = np.size(turbineX)
    rotorRadius = np.ones(nTurbs)*40.

    # Define flow properties
    windSpeed = 8.1
    wind_direction = 270.0

    #setup problem
    prob = Problem(root=Jensen(nTurbs))

    #initialize problem
    prob.setup()
    
    #assign values to parameters
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['turbineZ'] = turbineZ
    prob['rotorDiameter'] = rotorRadius
    prob['windSpeed'] = windSpeed
    prob['wind_direction'] = wind_direction
    prob['model_params:alpha'] = 0.1

    #run the problem
    print 'start Jensen run'
    tic = time.time()
    prob.run()
    toc = time.time()

    #print the results
    print 'Time to run: ', toc-tic
    print 'Hub Velocity at Each Turbine: ', prob['hubVelocity']
