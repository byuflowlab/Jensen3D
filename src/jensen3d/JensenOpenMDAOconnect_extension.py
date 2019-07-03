from __future__ import print_function, division, absolute_import
import time

import numpy as np
from scipy.integrate import quad

import openmdao.api as om
from plantenergy.GeneralWindFarmComponents import WindFrame

import _jensen


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
class Jensen_comp(om.ExplicitComponent):

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

        #unused but required for compatibility
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
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        outputs['wtVelocity%i' % direction_id] = _jensen.jensen(inputs['turbineXw'],
			inputs['turbineYw'], inputs['rotorDiameter'],
			discrete_inputs['model_params:alpha'], inputs['wind_speed'], inputs['axialInduction'])


class effectiveVelocityCosineOverlap(om.ExplicitComponent):

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
        self.add_input('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_input('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_discrete_input('model_params:alpha', val=0.1)
        self.add_discrete_input('model_params:spread_angle', val=2.0)
        self.add_input('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)
        self.add_input('overlap', val=np.empty([nTurbines, nTurbines]))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        self.declare_partials('*', '*', method='fd', form='central')
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
        spread_angle = discrete_inputs['model_params:spread_angle']
        a = inputs['axialInduction']
        windSpeed = inputs['wind_speed']
        overlap = inputs['overlap']

        loss = np.zeros(nTurbines, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbines, dtype=inputs._data.dtype)

        for i in range(nTurbines):
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineXw[i] - turbineXw[j]
                # if turbine j is upstream, calculate the deficit
                if dx > 0.0:
                    # determine cosine term
                    dy = turbineYw[i] - turbineYw[j]
                    dz = turbineZ[i] - turbineZ[j]
                    R = r[j]+dx*alpha
                    radiusLoc = np.sqrt(dy*dy+dz*dz)
                    rmax = spread_angle*(R + r[j])
                    cosFac = 0.5*(1.0 + np.cos(np.pi*radiusLoc/rmax))

                    loss[j] = overlap[i][j]*2.0*a[j]*(cosFac*r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
                    loss[j] = loss[j]**2

            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1-totalLoss)*windSpeed #effective hub velocity

        outputs['wtVelocity%i' % direction_id] = hubVelocity


class effectiveVelocityCosineNoOverlap(om.ExplicitComponent):

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

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'turbineZ', 'rotorDiameter', 'wind_speed', 'axialInduction']
        self.declare_partials('*', depvars, method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        r = 0.5*params['rotorDiameter']
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']
        a = params['axialInduction']
        windSpeed = params['wind_speed']

        loss = np.zeros(nTurbines, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbines, dtype=inputs._data.dtype)

        f_theta = get_cosine_factor_original(turbineXw, turbineYw, R0=r[0]*self.radius_multiplier, bound_angle=bound_angle)
        # print(f_theta)

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
            # print(hubVelocity)

        outputs['wtVelocity%i' % direction_id] = hubVelocity


class effectiveVelocityConference(om.ExplicitComponent):

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

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        #unused but required for compatibility
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_input('hubHeight', np.zeros(nTurbines), units='m')
        self.add_input('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_input('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_input('Ct', np.zeros(nTurbines))

        # used in Jensen model
        self.add_input('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_input('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_input('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_discrete_input('model_params:alpha', val=0.1)
        self.add_discrete_input('model_params:spread_angle', val=0.1)
        self.add_input('wind_speed', val=8.0, units='m/s')
        self.add_input('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'turbineZ', 'rotorDiameter', 'wind_speed', 'axialInduction']
        self.declare_partials('*', depvars, method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        x = inputs['turbineXw']
        y = inputs['turbineYw']
        rotorDiams = inputs['rotorDiameter']
        alpha = discrete_inputs['model_params:alpha']
        axialInd = inputs['axialInduction']
        effU_in = inputs['wind_speed']

        loss = np.zeros(nTurbines, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbines, dtype=inputs._data.dtype)

         # conference terms
        # boundAngle = 20       # before tuning to FLORIS
        # boundAngle = 7.7085     # tuned to FLORIS with offset
        # boundAngle = 12.7085     # tuned to FLORIS with offset
        # boundAngle =

        D = np.average(rotorDiams)
        a = np.average(axialInd)
        effU = np.average(effU_in)
        # alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
        # alpha = 0.01              # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
        # alpha = 0.084647            # Entrainment constant optimized to match FLORIS at 7*D
        # alpha = 0.399354
        # alpha = 0.216035
        R = D/2.
        # Ct = (4/3)*(1-1/3)
        # Uin = 8
        Uin = effU
        Uinf = np.ones(x.size, dtype=inputs._data.dtype)*Uin
        # a = (1-sqrt(1-Ct))/2          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
        # eta = 0.768
        # Cp = 4*a*((1-a)**2)*eta
        # rho = 1.1716
        n = np.size(x)
        # Area = np.pi*pow(R, 2)

        # commented out since this is now done elsewhere in the code
        # Phi = (90-wind)*np.pi/180           # define inflow direction relative to North = 0 CCW
        #
        # # adjust coordinates to wind direction
        # x = np.zeros(n)
        # y = np.zeros(n)
        # for i in range(0, n):
        #     # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        #     x[i] = X[i]*np.cos(-Phi)-Y[i]*np.sin(-Phi)
        #     y[i] = X[i]*np.sin(-Phi)+Y[i]*np.cos(-Phi)
        # # print x, y

        Ueff = np.zeros(n, dtype=inputs._data.dtype)     # initialize effective wind speeds array

        # find wake overlap
        Overlap_adjust = conferenceWakeOverlap(x, y, R)
        # Overlap_adjust = conferenceWakeOverlap_tune(x, y, R, boundAngle)

        # find upwind turbines and calc power for them
        # Pow = np.zeros(n, dtype=inputs._data.dtype)
        front = np.ones(n, dtype=inputs._data.dtype)
        for j in range(0, n):
            for i in range(0, n):
                if Overlap_adjust[i][j] != 0:
                    front[j] = 0

               # print j #Pow[j]

        # calc power for downstream turbines (actually now just calculate effective wind speed)

        xcopy = x-max(x)

        for j in range(0, n):

            q = np.argmin(xcopy)
            # print q, xcopy[q]
            if front[q] == 1:
               Ueff[q] = Uin

            elif front[q] == 0:
                G = 0
                for i in range(0, n):

                    if x[q] - x[i] > 0 and q != i:

                        z = abs(x[q] - x[i])

                        # V = Ueff[i]*(1-(2.0/3.0)*(R/(R+alpha*z))**2)
                        V = Uin*(1-Overlap_adjust[i][q]*(R/(R+alpha*z))**2) # conference method
                        # V = Uin*(1.-2.*a*(Overlap_adjust[i][q]*R/(R+alpha*z))**2) # corrected method (overlap_adjust is actualy f_cosine here)
                        # print V
                        G = G + (1.0-V/Uin)**2

                    # print 'G is:', G
                Ueff[q] = (1.-np.sqrt(G))*Uin
                # print Ueff[q]

            # commented since power is calculated elsewhere in code
            # Pow[q] = 0.5*rho*Area*Cp*Ueff[q]**3
                # print Pow[q]
            xcopy[q] = 1
        # Pow_tot = np.sum(Pow)

        # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
        outputs['wtVelocity%i' % direction_id] = Ueff


class JensenCosineYaw(om.ExplicitComponent):

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

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'turbineZ', 'rotorDiameter', 'wind_speed', 'axialInduction']
        self.declare_partials('*', depvars, method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        def get_wake_offset(dx, wake_spread_angle, yaw, Ct, R):
             # calculate initial wake angle
            # initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2 + 3.0*np.pi/180.
            initial_wake_angle = 0.5*Ct*np.sin(yaw)#*np.cos(yaw)**2

            # calculate distance from wake cone apex to wake producing turbine
            x1 = R/np.tan(wake_spread_angle)

            # calculate x position with cone apex as origin
            x = x1 + dx

            # calculate wake offset due to yaw
            deltaY = -initial_wake_angle*(x1**2)/x + x1*initial_wake_angle
            # print deltaY, initial_wake_angle, x1, Ct
            # return deltaY + 4.5
            return deltaY

        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        turbineZ = inputs['turbineZ']
        yaw = inputs['yaw%i' % direction_id]
        r = 0.5*inputs['rotorDiameter']
        Ct = inputs['Ct']
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']
        a = inputs['axialInduction']
        windSpeed = inputs['wind_speed']

        loss = np.zeros(nTurbines, dtype=inputs._data.dtype)
        hubVelocity = np.zeros(nTurbines, dtype=inputs._data.dtype)

        bound_angle *= np.pi/180.0                                      # convert bound angle to radians
        q = np.pi/bound_angle                                           # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))

        for i in range(nTurbines):
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineXw[i] - turbineXw[j]
                # if turbine j is upstream, calculate the deficit
                if dx > 0.0:
                    if self.radius_multiplier > 1.:
                        initial_wake_radius = r[i]+r[j]
                    else:
                        initial_wake_radius = r[j]

                    z = initial_wake_radius/np.tan(bound_angle)               # distance from fulcrum to wake producing turbine

                    deltaY = get_wake_offset(dx, bound_angle, yaw[j]*np.pi/180.0, Ct[j], initial_wake_radius)

                    # deltaY = 0.0 #get_wake_offset(dx, bound_angle, yaw[j], Ct[j], initial_wake_radius)

                    theta = np.arctan((turbineYw[j] - deltaY - turbineYw[i]) / (turbineXw[i] - turbineXw[j] + z))

                    if -bound_angle < theta < bound_angle:
                        f_theta = (1. + np.cos(q*theta))/2.
                    else:
                        f_theta = 0.0

                    # calculate velocity deficit
                    loss[j] = 2.0*a[j]*(f_theta*r[j]/(r[j]+alpha*dx))**2 #Jensen's formula

                    loss[j] = loss[j]**2

            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1.-totalLoss)*windSpeed #effective hub velocity
            # print hubVelocity

        outputs['wtVelocity%i' % direction_id] = hubVelocity


class JensenCosineYawIntegral(om.ExplicitComponent):

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

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        # Derivatives
        depvars = ['turbineXw', 'turbineYw', 'turbineZ', 'rotorDiameter', 'wind_speed', 'axialInduction']
        self.declare_partials('*', depvars, method='fd', form='central')
        self.set_check_partial_options('*', form='central')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']

        def get_wake_offset(dx, wake_spread_angle, yaw, Ct, initial_wake_radius):
            wake_spread_angle *= np.pi/180.0
            # wake_spread_angle *= .5

            yaw *= np.pi/180.0
             # calculate initial wake angle
            # initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2 + 3.0*np.pi/180.
            # initial_wake_angle = 2.0*np.pi/180.0 + 0.5*Ct*np.sin(yaw)#*np.cos(yaw)**2
            initial_wake_angle = 0.5*Ct*np.sin(yaw)#*np.cos(yaw)**2

            # print np.sin(yaw)

            # calculate distance from wake cone apex to wake producing turbine
            x1 = initial_wake_radius/np.tan(wake_spread_angle)

            # calculate x position with cone apex as origin
            x = x1 + dx

            # calculate wake offset due to yaw
            deltaY = -initial_wake_angle*(x1**2)/x + x1*initial_wake_angle
            # print deltaY, initial_wake_angle, 0.5*Ct*np.sin(yaw), Ct, np.sin(yaw), np.cos(yaw)**2
            # print deltaY, initial_wake_angle, x1, Ct
            # return deltaY + 18
            return deltaY

        def get_point_deficit(R, dx, bound_angle, alpha, initial_wake_radius, rotor_radius, axial_induction):

            # convert bound angle to radians
            bound_angle *= np.pi/180.0

            # distance from wake-cone fulcrum to wake producing turbine
            z = initial_wake_radius/np.tan(bound_angle)

            # angle from wake center line at fulcrum to radius of interest
            theta = np.arctan(R / (dx + z))

            if -bound_angle < theta < bound_angle:
                # cos term of the smooth Jensen (see Jensen1983 eq.(3)) based on percent of full wake angle
                f_theta = 0.5*(1. + np.cos(np.pi*theta/bound_angle))
            else:
                # no deficit outside the wake
                f_theta = 0.0

            # calculate velocity deficit
            deficit = 2.0*axial_induction*(f_theta*rotor_radius/(rotor_radius+alpha*dx))**2 #Jensen's formula

            return deficit

        def get_deficit_integral(R, dx, d, bound_angle, alpha, rotor_radius, initial_wake_radius, axial_induction):

            # if radius is very small, just set deficit to zero
            if R < 1E-12:
                # print "1"
                integration_angle = 0
            # if rotor overlaps the wake center then we have to account for full circular section of the wake
            elif (d < rotor_radius) and (R < abs(rotor_radius-d)):
                # print "2"
                integration_angle = 2.0*np.pi
            # if rotor does not overlap the wake center, then obtain arc angle from below equation derived from  the
            # geometry of two overlapping circles
            else:
                # print "3"
                integration_angle = 2.0*np.arccos((d**2+R**2-rotor_radius**2)/(2.*d*R))

            # get the deficit at the relevant radial location
            deficit = get_point_deficit(R, dx, bound_angle, alpha, initial_wake_radius, rotor_radius, axial_induction)
            # return the thing we are integrating
            return deficit*integration_angle*R

        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        turbineZ = inputs['turbineZ']
        yaw = inputs['yaw%i' % direction_id]
        r = 0.5*inputs['rotorDiameter']
        Ct = inputs['Ct']
        alpha = discrete_inputs['model_params:alpha']
        bound_angle = discrete_inputs['model_params:spread_angle']
        a = inputs['axialInduction']
        windSpeed = inputs['wind_speed']
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)

        for i in range(nTurbines):
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineXw[i] - turbineXw[j]
                dy = turbineYw[j] - turbineYw[i]
                # if turbine j is upstream, calculate the deficit
                if dx > 0.0:
                    # self.radius_multiplier = 1.0
                    if self.radius_multiplier > 1.:
                        initial_wake_radius = r[i]+r[j]
                    else:
                        initial_wake_radius = r[j]

                    # print dx/(2.*initial_wake_radius), dy/(2.*initial_wake_radius), 2.*initial_wake_radius, self.radius_multiplier

                    offset = get_wake_offset(dx, bound_angle, yaw[j], Ct[j], initial_wake_radius)

                    # deltaY = 0.0 #get_wake_offset(dx, bound_angle, yaw[j], Ct[j], initial_wake_radius)
                    d = abs(dy - offset)
                    # print d/(2.*r[i]), offset/(2.*r[i]), yaw
                    if d < 1E-6:
                        deficit, _ = quad(get_deficit_integral, 0.0, r[i], args=(dx, d, bound_angle, alpha, r[j], initial_wake_radius, a[j]))
                    elif d < r[i]:
                        deficit_0, _ = quad(get_deficit_integral, 0.0, r[i]-d, args=(dx, d, bound_angle, alpha, r[j], initial_wake_radius, a[j]))
                        deficit_1, _ = quad(get_deficit_integral, r[i]-d, r[i]+d, args=(dx, d, bound_angle, alpha, r[j], initial_wake_radius, a[j]))
                        deficit = deficit_0 + deficit_1
                    elif d >= r[i]:
                        deficit, _ = quad(get_deficit_integral, d-r[i], d+r[i], args=(dx, d, bound_angle, alpha, r[j], initial_wake_radius, a[j]))
                    # deficit = get_point_deficit(d, dx, bound_angle, alpha, initial_wake_radius, r[j], a[j])

                    deficit /= np.pi*r[i]**2

                    loss[j] = deficit**2

            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1.-totalLoss)*windSpeed #effective hub velocity
            # print hubVelocity

        outputs['wtVelocity%i' % direction_id] = hubVelocity


def conferenceWakeOverlap(X, Y, R):

    n = np.size(X)

    # theta = np.zeros((n, n), dtype=np.float)        # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing

    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R/np.tan(0.34906585)
                # print z
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                # print 'theta =', theta
                if -0.34906585 < theta < 0.34906585:
                    f_theta[i][j] = (1 + np.cos(9*theta))/2
                    # print f_theta

    # print z
    # print f_theta
    return f_theta


def conferenceWakeOverlap_tune(X, Y, R, boundAngle):

    n = np.size(X)
    boundAngle = boundAngle*np.pi/180.0
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/boundAngle                            # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))
    # print 'boundAngle = %s' %boundAngle, 'q = %s' %q
    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                # z = R/tan(0.34906585)
                z = R/np.tan(boundAngle)               # distance from fulcrum to wake producing turbine
                # print z
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                # print 'theta =', theta

                if -boundAngle < theta < boundAngle:

                    f_theta[i][j] = (1. + np.cos(q*theta))/2.
                    # print f_theta

    # print z
    # print f_theta
    return f_theta





def conferenceWakeOverlap_bk(X, Y, R):
    from math import atan, tan, cos
    n = np.size(X)

    # theta = np.zeros((n, n), dtype=np.float)        # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing

    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R/tan(0.34906585)
                theta = atan(Y[j] - Y[i]) / (X[j] - X[i] + z)
                if -0.34906585 < theta < 0.34906585:
                    f_theta[i][j] = (1 + cos(9*theta))/2
                    # print f_theta

    return f_theta


def jensen_bk(X, Y, wind, D):
    from math import sin, pi, cos
# X is an ndarray of the x-coordinates of all turbines in the plant
# Y is an ndarray of the y-coordinates of all turbines in the plant
# R is an integer representing a consistent turbine radius
# Uinf is the free stream velocity
# The cosine-bell method [N.O.Jensen] is  employed
# Assume no wake deflection. Set wake center equal to hub center
# Assume effective wind speed at rotor hub is the effective windspeed for the entire rotor

    alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
    # alpha = 0.01              # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
    R = D/2
    Uin = 8
    Uinf = np.ones(X.size)*Uin
    a = 0.333333333333          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
    eta = 0.768
    Cp = 4*a*((1-a)**2)*eta
    rho = 1.1716
    n = np.size(X)
    Area = pi*pow(R, 2)

    Phi = (90-wind)*pi/180           # define inflow direction relative to North = 0 CCW

    # adjust coordinates to wind direction
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        x[i] = X[i]*cos(-Phi)-Y[i]*sin(-Phi)
        y[i] = X[i]*sin(-Phi)+Y[i]*cos(-Phi)
    # print x, y
    Ueff = np.zeros(n)     # initialize effective wind speeds array

    # find wake overlap
    Overlap_adjust = conferenceWakeOverlap_bk(x, y, R)

    # find upwind turbines and calc power for them
    Pow = np.zeros(n)
    front = np.zeros(n)
    for j in range(0, n):
        for i in range(0, n):
            if Overlap_adjust[i][j] == 0:
                front[j] = 1

        if front[j] == 1:
            Ueff[j] = Uinf[j]
            Pow[j] = 0.5*rho*Area*Cp*Ueff[j]**3
            # print j #Pow[j]

    # calc power for downstream turbines
    for j in range(0, n):
        if front[j] == 0:
            k = 0
            q = 0

            for i in range(0, n):
                if X[j] - X[i] > 0:
                    q += 1
                    z = X[j] - X[i]
                    k += Overlap_adjust[i][j]*(R/(R+alpha*z))**2
                    # adjust_temp = Overlap_adjust[i][j]
            if q > 0:
                k = k/q

            Ueff[j] = Uinf[j]*(1 - 2*a*k)
            if Ueff[j] < 0:
                Ueff[j] = 0
                print(j)
            else:
                Pow[j] = 0.5*rho*Area*Cp*Ueff[j]**3

    Pow_tot = np.sum(Pow)

    # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
    return Pow


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
            model_options = {'variant': 'Original'}

        # typical variants
        if model_options['variant'] is 'TopHat':
            self.add_subsystem('f_1', JensenTopHat(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

        elif (model_options['variant'] is 'Cosine'):
            self.add_subsystem('f_1', JensenCosine(nTurbines, direction_id=direction_id, options=model_options),
                                promotes=['*'])
            #self.add_subsystem('f_2', effectiveVelocity(nTurbines, direction_id=direction_id), promotes=['*'])

        #elif model_options['variant'] is 'Cosine':
            #self.add_subsystem('f_1', wakeOverlap(nTurbines, direction_id=direction_id), promotes=['*'])
            #self.add_subsystem('f_2', effectiveVelocityCosineOverlap(nTurbines, direction_id=direction_id), promotes=['*'])

        elif (model_options['variant'] is 'CosineNoOverlap_1R') or (model_options['variant'] is 'CosineNoOverlap_2R'):
            self.add_subsystem('f_1', effectiveVelocityCosineNoOverlap(nTurbines, direction_id=direction_id, options=model_options),
                                promotes=['*'])

        elif model_options['variant'] is 'Conference':
            self.add_subsystem('f_1', effectiveVelocityConference(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

        elif (model_options['variant'] is 'CosineYaw_1R') or (model_options['variant'] is 'CosineYaw_2R'):
            self.add_subsystem('f_1', JensenCosineYaw(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                                promotes=['*'])

        elif model_options['variant'] is 'CosineYawIntegral':
            self.add_subsystem('f_1', JensenCosineYawIntegral(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
                                promotes=['*'])

        elif model_options['variant'] is 'CosineYaw':
            self.add_subsystem('f_1', JensenCosineYawIntegral(nTurbines=nTurbines, direction_id=direction_id, options=model_options),
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
