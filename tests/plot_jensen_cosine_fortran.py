import numpy as np
from matplotlib import pyplot as plt
import openmdao.api as om
from plantenergy.OptimizationGroups import AEPGroup
from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps

def get_jensen_problem(): 
    # define turbine size
    rotor_diameter = 40.0  # (m)
    hub_height = 90.0

    # define turbine locations in global reference frame
    turbineX = np.array([0.0, 6.0, 20.0])*rotor_diameter/2
    turbineY = np.array([0.0, 0.0, 0.0])

    z_ref = 90.0  # m
    z_0 = 0.0

    # load performance characteristics
    cut_in_speed = 3.  # m/s
    cut_out_speed = 25.  # m/s
    rated_wind_speed = 11.4  # m/s
    rated_power = 5000.  # kW
    generator_efficiency = 0.944

    input_directory = "./inputfiles/"
    filename = input_directory + "NREL5MWCPCT_dict.txt"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"

    data = np.loadtxt(filename)

    ct_curve = np.zeros([data[:, 0].size, 2])
    ct_curve_wind_speed = data[:, 0]
    ct_curve_ct = data[:, 2]

    # cp_curve_cp = data['CP']
    # cp_curve_wind_speed = data['wind_speed']

    loc0 = np.where(data[:, 0] < 11.55)
    loc1 = np.where(data[:, 0] > 11.7)

    cp_curve_cp = np.hstack([data[:, 1][loc0], data[:, 1][loc1]])
    cp_curve_wind_speed = np.hstack([data[:, 0][loc0], data[:, 0][loc1]])

    # initialize input variable arrays
    nTurbines = turbineX.size
    rotorDiameter = np.zeros(nTurbines)
    axialInduction = np.zeros(nTurbines)
    Ct = np.zeros(nTurbines)
    Cp = np.zeros(nTurbines)
    generatorEfficiency = np.zeros(nTurbines)
    yaw = np.zeros(nTurbines)

    # define initial values
    for turbI in range(0, nTurbines):
        rotorDiameter[turbI] = rotor_diameter  # m
        axialInduction[turbI] = 1.0 / 3.0
        Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
        Cp[turbI] = 0.7737 / 0.944 * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
        generatorEfficiency[turbI] = 1.0
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    nDirections = 1
    wind_speed = 6.6 # m/s
    air_density = 1.1716  # kg/m^3
    wind_direction = 270.0  # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_frequency = 1.  # probability of wind in this direction at this speed

    # set up problem

    wake_model_options = {'nSamples': 0,
                            'nRotorPoints': 1,
                            'use_ct_curve': True,
                            'ct_curve_ct': ct_curve_ct,
                            'ct_curve_wind_speed': ct_curve_wind_speed,
                            'interp_type': 1,
                            'use_rotor_components': False,
                            'differentiable': False,
                            'verbose': False,
                            'variant': "CosineFortran"}

    prob = om.Problem(model=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=jensen_wrapper,
                                        wake_model_options=wake_model_options,
                                        params_IdepVar_func=add_jensen_params_IndepVarComps,
                                        cp_points=cp_curve_cp.size,
                                        params_IdepVar_args={'use_angle': False}))

    # initialize problem
    prob.setup(check=False)

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = np.array([wind_frequency])
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_wind_speed
    # prob['model_params:spread_angle'] = 20.0
    # prob['model_params:alpha'] = 0.1

    # assign values to turbine states
    prob['cut_in_speed'] = np.ones(nTurbines) * cut_in_speed
    prob['cut_out_speed'] = np.ones(nTurbines) * cut_out_speed
    prob['rated_power'] = np.ones(nTurbines) * rated_power
    prob['rated_wind_speed'] = np.ones(nTurbines) * rated_wind_speed
    prob['use_power_curve_definition'] = True
    prob['gen_params:CTcorrected'] = True
    prob['gen_params:CPcorrected'] = True

    # run the problem
    prob.run_model()

    return prob

def get_jensen_1983_xr6(prob, ax):

    dataxr6 = np.loadtxt("./inputfiles/jensen_cosine_xr6.txt", delimiter=",")

    theta_xr6 = dataxr6[:,0]

    vu_xr6 = dataxr6[:,1]

    # theta = np.linspace(np.min(theta_xr6), np.max(theta_xr6), 100)
    theta = np.copy(theta_xr6)
    vu = np.zeros_like(theta)
    vu2 = np.zeros_like(theta)

    r = prob['rotorDiameter'][0]/2.0
    z = r/np.tan(20.0*np.pi/180.0)
    dx = r * 6.0

    prob['model_params:alpha'] = 0.1
    prob['model_params:spread_angle'] = 20.0

    prob['turbineX'][1] = dx
    for i in np.arange(0, len(theta)):

        dy = np.tan(theta[i]*np.pi/180.0)*(dx+z)
        prob['turbineY'][1] = dy

        prob.run_model()

        vu[i] = prob['wtVelocity0'][1]/prob['windSpeeds'][0]
        vu2[i] = prob['wtVelocity0'][2]/prob['windSpeeds'][0]

    ax[0].plot(theta, vu, label='xr6-model')
    ax[0].plot(theta_xr6, vu_xr6, 'o', label="xr6-data")
    # ax[1].plot(theta, vu, label='xr6-model')
    ax[1].plot(theta, vu2, '-', label="xr6 3rd turb")

    return

def get_jensen_1983_xr10(prob, ax):

    dataxr10 = np.loadtxt("./inputfiles/jensen_cosine_xr10.txt", delimiter=",")

    theta_xr10 = dataxr10[:,0]

    vu_xr10 = dataxr10[:,1]

    # theta = np.linspace(np.min(theta_xr6), np.max(theta_xr6), 100)
    theta = np.copy(theta_xr10)
    vu = np.zeros_like(theta)
    vu2 = np.zeros_like(theta)

    r = prob['rotorDiameter'][0]/2.0
    z = r/np.tan(20.0*np.pi/180.0)
    dx = r * 10.0

    prob['model_params:alpha'] = 0.1
    prob['model_params:spread_angle'] = 20.0

    prob['turbineX'][1] = dx
    for i in np.arange(0, len(theta)):

        dy = np.tan(theta[i]*np.pi/180.0)*(dx+z)
        prob['turbineY'][1] = dy

        prob.run_model()

        vu[i] = prob['wtVelocity0'][1]/prob['windSpeeds'][0]
        vu2[i] = prob['wtVelocity0'][2]/prob['windSpeeds'][0]
    
    ax[0].plot(theta, vu, label='xr10-model')
    ax[0].plot(theta_xr10, vu_xr10, 'o', label="xr10-data")
    # ax[1].plot(theta, vu, label='xr10-model')
    ax[1].plot(theta, vu2, '-', label="xr10 3rd turb")
    
    return

def get_jensen_1983_xr16(prob, ax):

    dataxr16 = np.loadtxt("./inputfiles/jensen_cosine_xr16.txt", delimiter=",")

    theta_xr16 = dataxr16[:,0]

    vu_xr16 = dataxr16[:,1]

    # theta = np.linspace(np.min(theta_xr6), np.max(theta_xr6), 100)
    theta = np.copy(theta_xr16)
    vu = np.zeros_like(theta)
    vu2 = np.zeros_like(theta)

    r = prob['rotorDiameter'][0]/2.0
    z = r/np.tan(20.0*np.pi/180.0)
    dx = r * 16.0

    prob['model_params:alpha'] = 0.1
    prob['model_params:spread_angle'] = 20.0

    prob['turbineX'][1] = dx
    for i in np.arange(0, len(theta)):

        dy = np.tan(theta[i]*np.pi/180.0)*(dx+z)
        prob['turbineY'][1] = dy

        prob.run_model()

        vu[i] = prob['wtVelocity0'][1]/prob['windSpeeds'][0]
        vu2[i] = prob['wtVelocity0'][2]/prob['windSpeeds'][0]

    ax[0].plot(theta, vu, label='xr16-model')
    ax[0].plot(theta_xr16, vu_xr16, 'o', label="xr16-data")
    # ax[1].plot(theta, vu, label='xr16-model')
    ax[1].plot(theta, vu2, '-', label="xr16 3rd turb")
    return

if __name__ == "__main__":

    fig, ax = plt.subplots(2)
    prob = get_jensen_problem()
    get_jensen_1983_xr6(prob,ax)
    get_jensen_1983_xr10(prob,ax)
    get_jensen_1983_xr16(prob,ax)

    plt.legend()
    plt.show()
