import numpy as np
from scipy.integrate import quad
from openmdao.api import Component, Group, Problem, IndepVarComp

from plantenergy.GeneralWindFarmComponents import WindFrame
import _jensen
import time

from JensenOpenMDAOconnect_extension import *

def get_cosine_factor_original(X, Y, R0, bound_angle=20.0):

    n = np.size(X)
    bound_angle = bound_angle*np.pi/180.0
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/bound_angle                           # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))

    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R0/np.tan(bound_angle)               # distance from fulcrum to wake producing turbine

                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))

                if -bound_angle < theta < bound_angle:

                    f_theta[i][j] = (1. + np.cos(q*theta))/2.

    return f_theta

def add_jensen_params_IndepVarComps(openmdao_object, model_options):

    openmdao_object.add('jp0', IndepVarComp('model_params:alpha', 2.0, pass_by_obj=True,
                                             desc='spread of cosine smoothing factor (multiple of sum of wake and '
                                                  'rotor radii)'),
                        promotes=['*'])

    if model_options['variant'] is 'Cosine' or model_options['variant'] is 'CosineNoOverlap':
        openmdao_object.add('jp1', IndepVarComp('model_params:spread_angle', 2.0, pass_by_obj=True,
                                                desc='spread of cosine smoothing factor (multiple of sum of wake and '
                                                     'rotor radii)'),
                            promotes=['*'])


# this component is to find the fraction of the all the rotors that are in the wakes of the other turbines
class JensenTopHat(Component):
    
    def __init__(self, nTurbines, direction_id=0):
        super(JensenTopHat, self).__init__()

        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')

        #unused but required for compatibility
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_param('Ct', np.zeros(nTurbines))

        self.add_param('model_params:alpha', val=0.1)
        
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')


    def solve_nonlinear(self, params, unknowns, resids):
        

        nTurbines = self.nTurbines
        direction_id = self.direction_id

        unknowns['wtVelocity%i' % direction_id] = _jensen.jensen(params['turbineXw'],
                                                                 params['turbineYw'],
                                                                 params['rotorDiameter'],
                                                                 params['model_params:alpha'],
                                                                 params['wind_speed'],
                                                                 params['axialInduction'])

class JensenCosine(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(JensenCosine, self).__init__()

        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6
        self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.direction_id = direction_id
        try:
            self.radius_multiplier = options['radius multiplier']
        except:
            self.radius_multiplier = 1.0


        #unused but required for compatibility
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_param('Ct', np.zeros(nTurbines))

        # used in this version of the Jensen model
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('model_params:alpha', val=0.1)
        self.add_param('model_params:spread_angle', val=20.0, desc="spreading angle in degrees")
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

    def solve_nonlinear(self, params, unknowns, resids):

        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        r = 0.5*params['rotorDiameter']
        alpha = params['model_params:alpha']
        bound_angle = params['model_params:spread_angle']
        a = params['axialInduction']
        windSpeed = params['wind_speed']
        nTurbines = self.nTurbines
        direction_id = self.direction_id
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)

        f_theta = get_cosine_factor_original(turbineXw, turbineYw, R0=r[0]*self.radius_multiplier, bound_angle=bound_angle)
        # print f_theta

        for i in range(nTurbines):
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineXw[i] - turbineXw[j]
                # if turbine j is upstream, calculate the deficit
                if dx > 0.0:

                  # calculate velocity deficit
                  loss[j] = 2.0*a[j]*(f_theta[j][i]*r[j]/(r[j]+alpha*dx))**2 #Jensen's formula

                  loss[j] = loss[j]**2

            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1.-totalLoss)*windSpeed #effective hub velocity
            # print hubVelocity
        unknowns['wtVelocity%i' % direction_id] = hubVelocity


class Jensen(Group):
    #Group with all the components for the Jensen model

    def __init__(self, nTurbs, direction_id=0, model_options=None):
        super(Jensen, self).__init__()

        try:
            model_options['variant']
        except:
            model_options = {'variant': 'Tophat'}

        # typical variants
        if model_options['variant'] is 'TopHat':
            self.add('f_1', JensenTopHat(nTurbs, direction_id=direction_id), promotes=['*'])
        elif (model_options['variant'] is 'Cosine'):
            self.add('f_1', JensenCosine(nTurbs, direction_id=direction_id, options=model_options),
                 promotes=['*'])

        # non-typical variants for various research purposes
        else:
                #self.add('f_2', effectiveVelocity(nTurbs, direction_id=direction_id), promotes=['*'])
            #elif model_options['variant'] is 'Cosine':
                #self.add('f_1', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
                #self.add('f_2', effectiveVelocityCosineOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
            if (model_options['variant'] is 'CosineNoOverlap_1R') or (model_options['variant'] is 'CosineNoOverlap_2R'):
                from JensenOpenMDAOconnect_extension import effectiveVelocityCosineNoOverlap
                self.add('f_1', effectiveVelocityCosineNoOverlap(nTurbs, direction_id=direction_id, options=model_options),
                         promotes=['*'])
            elif model_options['variant'] is 'Conference':
                from JensenOpenMDAOconnect_extension import effectiveVelocityConference
                self.add('f_1', effectiveVelocityConference(nTurbines=nTurbs, direction_id=direction_id), promotes=['*'])
            elif (model_options['variant'] is 'CosineYaw_1R') or (model_options['variant'] is 'CosineYaw_2R'):
                from JensenOpenMDAOconnect_extension import JensenCosineYaw
                self.add('f_1', JensenCosineYaw(nTurbines=nTurbs, direction_id=direction_id, options=model_options),
                         promotes=['*'])
            elif model_options['variant'] is 'CosineYawIntegral':
                from JensenOpenMDAOconnect_extension import JensenCosineYawIntegral
                self.add('f_1', JensenCosineYawIntegral(nTurbines=nTurbs, direction_id=direction_id, options=model_options),
                         promotes=['*'])
            elif model_options['variant'] is 'CosineYaw':
                from JensenOpenMDAOconnect_extension import JensenCosineYaw
                self.add('f_1', JensenCosineYaw(nTurbines=nTurbs, direction_id=direction_id, options=model_options),
                         promotes=['*'])


if __name__=="__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([0, 100, 200])
    turbineY = np.array([0, 30, -31])
    turbineZ = np.array([150, 150, 150])
    
    # initialize input variable arrays
    nTurbs = np.size(turbineX)
    rotorRadius = np.ones(nTurbs)*40.

    # Define flow properties
    windSpeed = 8.1
    wind_direction = 270.0

    # set model options
    # model_options = {'variant': 'Original'}
    # model_options = {'variant': 'CosineOverlap'}
    # model_options = {'variant': 'Cosine'}
    model_options = {'variant': 'CosineYaw_1R'}

    #setup problem
    prob = Problem(root=Group())

    prob.root.add('windframe', WindFrame(nTurbs), promotes=['*'])
    prob.root.add('jensen', Jensen(nTurbs, model_options=model_options), promotes=['*'])

    #initialize problem
    prob.setup()
    
    #assign values to parameters
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['turbineZ'] = turbineZ
    prob['rotorDiameter'] = rotorRadius
    prob['wind_speed'] = windSpeed
    prob['wind_direction'] = wind_direction
    prob['model_params:alpha'] = 0.1

    #run the problem
    print 'start Jensen run'
    tic = time.time()
    prob.run()
    toc = time.time()

    #print the results
    print 'Time to run: ', toc-tic
    print 'Hub Velocity at Each Turbine: ', prob['wtVelocity0']
