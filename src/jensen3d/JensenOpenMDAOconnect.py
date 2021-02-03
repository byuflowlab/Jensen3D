from __future__ import print_function, division, absolute_import
import time

import numpy as np
from scipy.integrate import quad

import openmdao.api as om
from plantenergy.GeneralWindFarmComponents import WindFrame

from jensen3d.JensenOpenMDAOconnect_extension import *
import _jensen, _jensen2



# Function Inputs:
#   X is an array containing the x-positions of all the turbines in the wind farm.
#   Y is an array containing the y-positions of all the turbines in the wind farm.
#   R0 is a scalar containing the radius of the 0th turbine scaled by "self.radius_multiplier".
#   bound_angle appears to be a scalar containing the angle at which the wake is spreading. Its
#       default value appears to be 20 degrees, but it's written in such a way that it can be
#       altered.
# Function Outputs:
#   f_theta = Array of all the cosine factors for each combination of turbines. For turbines
#       that aren't in another turbine's wake, that value of f_theta remains at zero.
def get_cosine_factor_original(X, Y, R0, bound_angle=20.0, relaxationFactor=1.0):

    n = np.size(X)
    bound_angle = bound_angle*np.pi/180.0           # convert bound_angle from degrees to radians
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/bound_angle                           # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))

    # Idea for relaxation factor requires new angle, gamma. Units in radians.
    gamma = (np.pi/2.0) - bound_angle

    # Calculate the cosine factor on the jth turbine from each ith turbine. Each row represents the cosine factor on
    # each turbine from the ith turbine, and each column represents the cosine factor on the jth turbine from each
    # ith turbine.
    for i in range(0, n):
        for j in range(0, n):

            # Only take action if the jth turbine is downstream (has greater x) than the ith turbine.
            if X[i] < X[j]:
                # z = R0/np.tan(bound_angle)               # distance from fulcrum to wake producing turbine
                # z = (R0 * relaxationFactor)/np.tan(bound_angle)
                z = (relaxationFactor * R0 * np.sin(gamma))/np.sin(bound_angle) # this eq. does the same thing as the
                #  equation direction above

                # angle in x-y plane from ith turbine to jth turbine. Measured positive counter-clockwise from positive
                # x-axis. 'z' included because the triangle actually starts at a distance 'z' in the negative
                # x-direction from the wake-producing turbine.
                # THETA ACTUALLY MEASURED BETWEEN THE FULCRUM OF THE WAKE AND THE DOWNSTREAM TURBINE.
                theta = np.arctan((Y[j] - Y[i]) / (np.abs(X[j] - X[i]) + z))

                # If theta is less than the bound angle, that means the jth turbine is within the ith turbine's wake.
                # else, f_theta[i, j] remains at zero.
                if -bound_angle < theta < bound_angle:

                    # f_theta[i][j] = (1. + np.cos((q*theta)/relaxationFactor))/2.    # cosine factor from Jensen's
                                                                                    # paper (1983, eq. 3).
                    f_theta[i][j] = (1. + np.cos(q*theta))/2.

    return f_theta

def add_jensen_params_IndepVarComps(om_group, model_options):

    ivc = om_group.add_subsystem('model_params', om.IndepVarComp(), promotes=['*'])

    ivc.add_discrete_output('model_params:alpha', 2.0,
                            desc='spread of cosine smoothing factor (multiple of sum of wake and '
                            'rotor radii)')

    if model_options['variant'] is 'Cosine' or model_options['variant'] is 'CosineNoOverlap':
        ivc.add_discrete_output('model_params:spread_angle', 2.0,
                                desc='spread of cosine smoothing factor (multiple of sum of wake and '
                                'rotor radii)')


# this component is to find the fraction of the all the rotors that are in the wakes of the other turbines
class JensenTopHat(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        self.add_input('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_input('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')

        # Unused but required for compatibility
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_input('hubHeight', np.zeros(nTurbines), units='m')
        self.add_input('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_input('Ct', np.zeros(nTurbines))

        self.add_discrete_input('model_params:alpha', val=0.1)

        self.add_input('wind_speed', val=8.0, units='m/s')
        self.add_input('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'rotorDiameter']
        self.declare_partials('*', depvars, method='fd', form='central')

        # Complex step not supported across fortran module
        self.set_check_partial_options('*', form='central', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        outputs['wtVelocity%i' % direction_id] = _jensen.jensen(inputs['turbineXw'],
                                                                inputs['turbineYw'],
                                                                inputs['rotorDiameter'],
                                                                discrete_inputs['model_params:alpha'],
                                                                inputs['wind_speed'],
                                                                inputs['axialInduction'])

class JensenCosine(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('options', types=dict, default=None, allow_none=True,
                             desc="Additional parameters for this component.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        options = opt['options']

        try:
            self.radius_multiplier = options['radius multiplier']
        except:
            self.radius_multiplier = 1.0

        #unused but required for compatibility
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_input('hubHeight', np.zeros(nTurbines), units='m')
        self.add_input('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_input('Ct', np.zeros(nTurbines))

        # used in this version of the Jensen model
        self.add_input('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_input('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_discrete_input('model_params:alpha', val=0.1)
        self.add_discrete_input('model_params:spread_angle', val=20.0, desc="spreading angle in degrees")
        self.add_input('wind_speed', val=8.0, units='m/s')
        self.add_input('axialInduction', val=np.zeros(nTurbines)+1./3.)

        # Spencer M's edit for WEC: add in xi (i.e., relaxation factor) as a parameter.
        self.add_discrete_input('model_params:wec_factor', val=1.0)
        # self.add_input('relaxationFactor', val=np.arange(3.0, 0.75, -0.25))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'turbineZ', 'rotorDiameter', 'wind_speed', 'axialInduction']
        self.declare_partials('*', depvars, method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        turbineZ = inputs['turbineZ']
        r = 0.5*inputs['rotorDiameter']
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']
        a = inputs['axialInduction']
        windSpeed = inputs['wind_speed']

        loss = np.zeros(nTurbines, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbines, dtype=inputs._data.dtype)

        # Save the relaxation factor from the params dictionary into a variable to be used in this function.
        relaxationFactor = discrete_inputs['model_params:wec_factor']

        f_theta = get_cosine_factor_original(turbineXw, turbineYw, R0=r[0]*self.radius_multiplier,
                                             bound_angle=bound_angle, relaxationFactor=relaxationFactor)
        # print f_theta

        # Calculate the hub velocity of the wind at the ith turbine downwind of the jth turbine.
        for i in range(nTurbines):
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineXw[i] - turbineXw[j]
                # if turbine j is upstream, calculate the deficit
                if dx > 0.0:

                    # calculate velocity deficit - looks like it's currently squaring the cosine factor.
                    loss[j] = 2.0*a[j]*f_theta[j][i]*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
                    # loss[j] = 2.0*a[j]*f_theta[j][i]*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula

                    loss[j] = loss[j]**2

            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1.-totalLoss)*windSpeed #effective hub velocity
            # print hubVelocity

        outputs['wtVelocity%i' % direction_id] = hubVelocity


class JensenCosineFortran(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('options', types=dict, default=None, allow_none=True,
                             desc="Additional parameters for this component.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        options = opt['options']

        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 1.0e-6
        # self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.direction_id = direction_id
        try:
            self.radius_multiplier = options['radius multiplier']
        except:
            self.radius_multiplier = 1.0
        
        try:
            self.use_ct_curve = options['use_ct_curve']
            self.ct_curve_ct = options['ct_curve_ct']
            self.ct_curve_wind_speed = options['ct_curve_wind_speed']
        except:
            print("ct curve options not found.")
            self.use_ct_curve = False
            self.ct_curve_ct = np.array([0.0])
            self.ct_curve_wind_speed = np.array([0.0])
                
        ct_max = 0.99
        if np.any(self.ct_curve_ct > 0.):
            if np.any(self.ct_curve_ct > ct_max):
                warnings.warn('Ct values must be <= 1, clipping provided values accordingly')
                self.ct_curve_ct = np.clip(self.ct_curve_ct, a_max=ct_max, a_min=None)

        #unused but required for compatibility
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_input('hubHeight', np.zeros(nTurbines), units='m')
        self.add_input('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_input('Ct', np.zeros(nTurbines))

        # used in this version of the Jensen model
        self.add_input('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_input('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_discrete_input('model_params:alpha', val=0.1)
        self.add_discrete_input('model_params:spread_angle', val=20.0, desc="spreading angle in degrees")
        self.add_input('wind_speed', val=8.0, units='m/s')
        self.add_input('axialInduction', val=np.zeros(nTurbines)+1./3.)

        # Spencer M's edit for WEC: add in xi (i.e., relaxation factor) as a parameter.
        self.add_discrete_input('model_params:wec_factor', val=1.0)
        # self.add_input('relaxationFactor', val=np.arange(3.0, 0.75, -0.25))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw']

        self.declare_partials('*', depvars, method='exact')

        # Complex step not supported across fortran module
        self.set_check_partial_options('*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        rotorDiameter = inputs['rotorDiameter']
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']
        windSpeed = inputs['wind_speed']
        use_ct_curve = self.use_ct_curve
        ct_curve_wind_speed = self.ct_curve_wind_speed
        ct_curve_ct = self.ct_curve_ct

        # Save the relaxation factor from the params dictionary into a variable to be used in this function.
        wec_factor = discrete_inputs['model_params:wec_factor']

        wtVelocity = _jensen2.jensenwake(turbineXw, turbineYw, rotorDiameter, alpha, bound_angle, ct_curve_ct, ct_curve_wind_speed, use_ct_curve, wec_factor, windSpeed)

        outputs['wtVelocity%i' % direction_id] = wtVelocity

    def compute_partials(self, inputs, partials, discrete_inputs):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        # x and y positions w.r.t. the wind dir. (wind dir. = +x)
        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']

        # turbine specs
        rotorDiameter = inputs['rotorDiameter']

        # air flow
        wind_speed = inputs['wind_speed']
        Ct = inputs['Ct']

        # wake model parameters
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']

        wec_factor = discrete_inputs['model_params:wec_factor']

        use_ct_curve = self.use_ct_curve
        ct_curve_wind_speed = self.ct_curve_wind_speed
        ct_curve_ct = self.ct_curve_ct
        
        # define jacobian size
        nTurbines = len(turbineXw)
        nDirs = nTurbines

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)

        turbineXwb, turbineYwb =  _jensen2.jensenwake_bv(turbineXw,turbineYw,
            rotorDiameter, alpha, bound_angle, ct_curve_ct, ct_curve_wind_speed, use_ct_curve,
            wec_factor, wind_speed, wtVelocityb)

        # print("this is a check")
        # print(np.shape(turbineXwb))
        # print(np.shape(turbineXw))
        # print(np.shape(turbineYwb))
        # print(np.shape(turbineYw))
        partials['wtVelocity%i' % direction_id, 'turbineXw'] = turbineXwb
        partials['wtVelocity%i' % direction_id, 'turbineYw'] = turbineYwb
        # partials['wtVelocity%i' % direction_id, 'rotorDiameter'] = rotorDiameterb

class Jensen(om.Group):
    #Group with all the components for the Jensen model

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('model_options', types=dict, default=None, allow_none=True,
                             desc="Additional parameters for this group.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        model_options = opt['model_options']

        try:
            model_options['variant']
        except:
            model_options = {'variant': 'Tophat'}

        # typical variants
        if model_options['variant'] is 'TopHat':
            self.add_subsystem('f_1', JensenTopHat(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

        elif (model_options['variant'] is 'Cosine'):
            self.add_subsystem('f_1', JensenCosine(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                 promotes=['*'])

        elif model_options['variant'] is 'CosineFortran':   # PJ's new Jensen code in FORTRAN.
            self.add_subsystem('f_1', JensenCosineFortran(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                     promotes=['*'])

        # non-typical variants for various research purposes
        else:
                #self.add_subsystem('f_2', effectiveVelocity(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

            #elif model_options['variant'] is 'Cosine':
                #self.add_subsystem('f_1', wakeOverlap(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])
                #self.add_subsystem('f_2', effectiveVelocityCosineOverlap(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

            if (model_options['variant'] is 'CosineNoOverlap_1R') or (model_options['variant'] is 'CosineNoOverlap_2R'):
                from JensenOpenMDAOconnect_extension import effectiveVelocityCosineNoOverlap
                self.add_subsystem('f_1', effectiveVelocityCosineNoOverlap(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                         promotes=['*'])

            elif model_options['variant'] is 'Conference':
                from JensenOpenMDAOconnect_extension import effectiveVelocityConference
                self.add_subsystem('f_1', effectiveVelocityConference(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

            elif (model_options['variant'] is 'CosineYaw_1R') or (model_options['variant'] is 'CosineYaw_2R'):
                from JensenOpenMDAOconnect_extension import JensenCosineYaw
                self.add_subsystem('f_1', JensenCosineYaw(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                         promotes=['*'])

            elif model_options['variant'] is 'CosineYawIntegral':
                from JensenOpenMDAOconnect_extension import JensenCosineYawIntegral
                self.add_subsystem('f_1', JensenCosineYawIntegral(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                         promotes=['*'])

            elif model_options['variant'] is 'CosineYaw':
                from JensenOpenMDAOconnect_extension import JensenCosineYaw
                self.add_subsystem('f_1', JensenCosineYaw(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
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

    # Tried inserting the WEC relaxation factor here to see if it would work, but still getting error.
    relaxationFactor = np.arange(3.0, 0.75, -0.25)

    # set model options
    # model_options = {'variant': 'Original'}
    # model_options = {'variant': 'CosineOverlap'}
    # model_options = {'variant': 'Cosine'}
    model_options = {'variant': 'CosineYaw_1R'}

    #setup problem
    prob = om.Problem()

    prob.model.add_subsystem('windframe', WindFrame(nTurbines=nTurbs), promotes=['*'])
    prob.model.add_subsystem('jensen', Jensen(nTurbines=nTurbs, model_options=model_options), promotes=['*'])

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
    print('start Jensen run')
    tic = time.time()
    prob.run_model()
    toc = time.time()

    #print the results
    print('Time to run: ', toc-tic)
    print('Hub Velocity at Each Turbine: ', prob['wtVelocity0'])
