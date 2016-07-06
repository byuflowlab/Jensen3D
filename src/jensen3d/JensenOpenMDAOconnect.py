import numpy as np
from openmdao.api import Component, Group, Problem, IndepVarComp

from florisse.GeneralWindFarmComponents import WindFrame

import time


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
class wakeOverlap(Component):
    
    def __init__(self, nTurbines, direction_id=0):
        super(wakeOverlap, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')

        #unused but required for compatibility
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_param('Ct', np.zeros(nTurbines))


        self.add_param('model_params:alpha', val=0.1)

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
                        overlap_area = r[i]**2.*np.arccos((d**2.+r[i]**2.-R**2.)/(2.0*d*r[i]))+\
                                       R**2.*np.arccos((d**2.+R**2.-r[i]**2.)/(2.0*d*R))-\
                                       0.5*np.sqrt((-d+r[i]+R)*(d+r[i]-R)*(d-r[i]+R)*(d+r[i]+R))
                        overlap_fraction[i][j] = overlap_area/A
                
        # print "Overlap Fraction Matrix: ", overlap_fraction
        # print overlap_fraction
        unknowns['overlap'] = overlap_fraction


# solve for the effective wind velocity at each turbine
class effectiveVelocity(Component):

    def __init__(self, nTurbines, direction_id=0):
        super(effectiveVelocity, self).__init__()
        
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True
        
        self.nTurbines = nTurbines
        self.direction_id = direction_id

        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('model_params:alpha', val=0.1)
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)
        self.add_param('overlap', val=np.empty([nTurbines, nTurbines]))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

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
            loss[:] = 0.0
            for j in range(nTurbines):
                dx = turbineX[i]-turbineX[j]
                if dx > 0:
                    loss[j] = overlap[i][j]*2.0*a[j]*(r[j]/(r[j]+alpha*dx))**2 #Jensen's formula
                    loss[j] = loss[j]**2
            totalLoss = np.sqrt(np.sum(loss)) #square root of the sum of the squares
            hubVelocity[i] = (1-totalLoss)*windSpeed #effective hub velocity
        unknowns['wtVelocity%i' % direction_id] = hubVelocity


class effectiveVelocityCosineOverlap(Component):

    def __init__(self, nTurbines, direction_id=0):
        super(effectiveVelocityCosineOverlap, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('model_params:alpha', val=0.1)
        self.add_param('model_params:spread_angle', val=2.0)
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)
        self.add_param('overlap', val=np.empty([nTurbines, nTurbines]))

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

    def solve_nonlinear(self, params, unknowns, resids):

        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        r = 0.5*params['rotorDiameter']
        alpha = params['model_params:alpha']
        spread_angle = params['model_params:spread_angle']
        a = params['axialInduction']
        windSpeed = params['wind_speed']
        nTurbines = self.nTurbines
        direction_id = self.direction_id
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)
        overlap = params['overlap']

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
        unknowns['wtVelocity%i' % direction_id] = hubVelocity


class effectiveVelocityCosineNoOverlap(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(effectiveVelocityCosineNoOverlap, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.direction_id = direction_id
        if options is None:
            self.radius_multiplier = 1.0
        else:
            self.radius_multiplier = options['radius multiplier']


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


class effectiveVelocityConference(Component):

    def __init__(self, nTurbines, direction_id=0):
        super(effectiveVelocityConference, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        #unused but required for compatibility
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))
        self.add_param('Ct', np.zeros(nTurbines))

        # used in Jensen model
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('model_params:alpha', val=0.1)
        self.add_param('model_params:spread_angle', val=0.1)
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

    def solve_nonlinear(self, params, unknowns, resids):

        x = params['turbineXw']
        y = params['turbineYw']
        rotorDiams = params['rotorDiameter']
        alpha = params['model_params:alpha']
        axialInd = params['axialInduction']
        effU_in = params['wind_speed']
        nTurbines = self.nTurbines
        direction_id = self.direction_id
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)

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
        Uinf = np.ones(x.size)*Uin
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

        Ueff = np.zeros(n)     # initialize effective wind speeds array

        # find wake overlap
        Overlap_adjust = conferenceWakeOverlap(x, y, R)
        # Overlap_adjust = conferenceWakeOverlap_tune(x, y, R, boundAngle)

        # find upwind turbines and calc power for them
        # Pow = np.zeros(n)
        front = np.ones(n)
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
        unknowns['wtVelocity%i' % direction_id] = Ueff


class JensenCosineYaw(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(JensenCosineYaw, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.direction_id = direction_id
        if options is None:
            self.radius_multiplier = 1.0
        else:
            self.radius_multiplier = options['radius multiplier']

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

        def get_wake_offset(dx, wake_spread_angle, yaw, Ct, R):
             # calculate initial wake angle
            # initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2 + 3.0*np.pi/180.
            initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2

            # calculate distance from wake cone apex to wake producing turbine
            x1 = R/np.tan(wake_spread_angle)

            # calculate x position with cone apex as origin
            x = x1 + dx

            # calculate wake offset due to yaw
            deltaY = -initial_wake_angle*(x1**2)/x + x1*initial_wake_angle
            # print deltaY, initial_wake_angle, x1, Ct
            # return deltaY + 4.5
            return deltaY

        nTurbines = self.nTurbines
        direction_id = self.direction_id
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        yaw = params['yaw%i' % direction_id]
        r = 0.5*params['rotorDiameter']
        Ct = params['Ct']
        alpha = params['model_params:alpha']
        bound_angle = params['model_params:spread_angle']
        a = params['axialInduction']
        windSpeed = params['wind_speed']
        loss = np.zeros(nTurbines)
        hubVelocity = np.zeros(nTurbines)

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
        unknowns['wtVelocity%i' % direction_id] = hubVelocity

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
                print j
            else:
                Pow[j] = 0.5*rho*Area*Cp*Ueff[j]**3

    Pow_tot = np.sum(Pow)

    # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
    return Pow


class Jensen(Group):
    #Group with all the components for the Jensen model

    def __init__(self, nTurbs, direction_id=0, model_options=None):
        super(Jensen, self).__init__()

        try:
            model_options['variant']
        except:
            model_options = {'variant': 'Original'}

        if model_options['variant'] is 'Original':
            self.add('f_1', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
            self.add('f_2', effectiveVelocity(nTurbs, direction_id=direction_id), promotes=['*'])
        elif model_options['variant'] is 'Cosine':
            self.add('f_1', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
            self.add('f_2', effectiveVelocityCosineOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
        elif model_options['variant'] is 'CosineNoOverlap_1R' or model_options['variant'] is 'CosineNoOverlap_2R':
            self.add('f_1', effectiveVelocityCosineNoOverlap(nTurbs, direction_id=direction_id, options=model_options),
                     promotes=['*'])
        elif model_options['variant'] is 'Conference':
            self.add('f_1', effectiveVelocityConference(nTurbines=nTurbs, direction_id=direction_id), promotes=['*'])
        elif model_options['variant'] is 'CosineYaw_1R' or model_options['variant'] is 'CosineYaw_2R':
            self.add('f_1', JensenCosineYaw(nTurbines=nTurbs, direction_id=direction_id, options=model_options), promotes=['*'])


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
