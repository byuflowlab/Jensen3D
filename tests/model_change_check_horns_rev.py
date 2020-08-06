from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
# from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.utilities import sunflower_points, circumference_points
import time
import numpy as np
import matplotlib.pyplot as plt
from _porteagel_fortran import ct_to_axial_ind_func

from scipy.interpolate import UnivariateSpline

#import cProfile
#import sys

def niayifar_power_model(u):
    power_niayifar_model = 0.17819 * (u) ** 5. - 6.5198 * (u) ** 4. + \
                           90.623 * (u) ** 3. - 574.62 * (u) ** 2. + 1727.2 * (u) - 1975.
    return power_niayifar_model


if __name__ == "__main__":

    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    model = 1
    wake_model_version = 2016
    print(MODELS[model])

    sort_turbs = True
    wake_combination_method = 1 # can be [0:Linear freestreem superposition,
                                       #  1:Linear upstream velocity superposition,
                                       #  2:Sum of squares freestream superposition,
                                       #  3:Sum of squares upstream velocity superposition]

    ti_calculation_methods = np.array([9])  # can be [0:No added TI calculations,
                                        #1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
                                        #2:TI by Niayifar and Porte Agel 2016,
                                        #3:TI by Niayifar and Porte Agel 2016 with added soft max function,
                                        #4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
                                        #5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]
                                        #9 use JensenCosineFortran

    nRotorPoints = 1
    # location = 0.735
    # location = (0.7353288267358161 + 0.6398349246319044)/2.
    location = 0.69
    # location = 0.625
    # location = 0.73
    rotor_pnt_typ = 1  # can be [0: circumference points,
                       #         1: sunflower points\

    z_ref = 70.0
    z_0 = 0.0002
    # z_0 = 0.000
    TI = 0.077

    rotor_diameter = 80.0  # (m)
    hub_height = 70.0

    # k_calc = 0.022
    k_calc = 0.3837*TI + 0.003678

    ct_curve = np.loadtxt('./inputfiles/mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    ct_curve_ct = ct_curve[:, 1]
    ct_curve_wind_speed = ct_curve[:, 0]
    # ct_curve = np.loadtxt('./input_files/predicted_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    # filename = "../input_files/NREL5MWCPCT_dict.p"
    # # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    # import cPickle as pickle
    #
    # data = pickle.load(open(filename, "rb"))
    # ct_curve = np.zeros([data['wind_speed'].size, 2])
    # ct_curve_wind_speed = ct_curve['wind_speed']
    # ct_curve_ct = ct_curve['CT']

    air_density = 1.1716  # kg/m^3
    Ar = 0.25*np.pi*rotor_diameter**2
    cp_curve_wind_speed = ct_curve[:, 0]
    power_data = np.loadtxt('./inputfiles/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
    # cp_curve_cp = niayifar_power_model(cp_curve_wind_speed)/(0.5*air_density*cp_curve_wind_speed**3*Ar)
    cp_curve_cp = power_data[:, 1]*(1E6)/(0.5*air_density*power_data[:,0]**3*Ar)
    cp_curve_wind_speed = power_data[:,0]
    cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
    cp_curve_spline.set_smoothing_factor(.0001)
    # xs = np.linspace(0, 35, 1000)
    # plt.plot(xs, cp_curve_spline(xs))
    # plt.scatter(cp_curve_wind_speed, cp_curve_cp)
    # plt.show()
    # quit()
    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speeed': ct_curve_wind_speed,
                          'interp_type': 1}

    locations = np.loadtxt("./inputfiles/horns_rev_locations.txt", delimiter=",")
    turbineX = locations[:, 0]*rotor_diameter
    turbineY = locations[:, 1]*rotor_diameter
    nTurbines = turbineX.size

    nDirections = 200

    shear_exp = 0.15

    # plt.plot(turbineX, turbineY, 'o')
    # plt.show()
    # quit()

    normalized_power_NPA = np.zeros([10, nDirections])
    normalized_power_OURS = np.zeros([10, nDirections])


    for ti_calculation_method in ti_calculation_methods:
                                        # can be [0:No added TI calculations,
                                        #1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
                                        #2:TI by Niayifar and Porte Agel 2016,
                                        #3:TI by Niayifar and Porte Agel 2016 with added soft max function,
                                        #4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
                                        #5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]
        if ti_calculation_method == 0:
            calc_k_star = False
        else:
            calc_k_star = True
        if ti_calculation_method == 9:
            model = 2
            wake_model_options = {'nSamples': 0,
                                  'nRotorPoints': nRotorPoints,
                                  'use_ct_curve': True,
                                  'ct_curve_ct': ct_curve_ct,
                                  'ct_curve_wind_speeed': ct_curve_wind_speed,
                                  'interp_type': 1,
                                  'variant': "CosineFortran"}
        ######################### for MPI functionality #########################
        # from openmdao.core.mpi_wrap import MPI
        #
        # if MPI:  # pragma: no cover
        #     # if you called this script with 'mpirun', then use the petsc data passing
        #     from openmdao.core.petsc_impl import PetscImpl as impl
        #
        # else:
        #     # if you didn't use 'mpirun', then use the numpy data passing
        #     from openmdao.api import BasicImpl as impl
        #
        #
        # def print(*args):
        #     """ helper function to only print on rank 0 """
        #     if prob.root.comm.rank == 0:
        #         print(*args)
        #
        # 
        # prob = Problem(impl=impl)
        print(model)
        # continue
        size = 1  # number of processors (and number of wind directions to run)

        #########################################################################
        # define turbine size


        # define turbine locations in global reference frame
        # original example case
        # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
        # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

        # # Scaling grid case
        # nRows = 3  # number of rows and columns in grid
        # spacing = 3.5  # turbine grid spacing in diameters
        #
        # # Set up position arrays
        # points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
        # xpoints, ypoints = np.meshgrid(points, points)
        # turbineX = np.ndarray.flatten(xpoints)
        # turbineY = np.ndarray.flatten(ypoints)


        # turbineY[9] = rotor_diameter/2.0
        # np.savetxt('RoundFarm38Turb5DSpacing.txt', np.c_[turbineX+500.,turbineY+500.], header="TurbineX (m), TurbineY (m)")
        # locs = np.loadtxt('RoundFarm38Turb5DSpacing.txt')
        # x = locs[:, 0]/rotor_diameter
        # y = locs[:, 1]/rotor_diameter
        # plt.scatter(x, y)
        # set values for circular boundary constraint

        # initialize input variable arrays
        nTurbs = turbineX.size
        rotorDiameter = np.zeros(nTurbs)
        hubHeight = np.zeros(nTurbs)
        axialInduction = np.zeros(nTurbs)
        Ct = np.zeros(nTurbs)
        Cp = np.zeros(nTurbs)
        generatorEfficiency = np.zeros(nTurbs)
        yaw = np.zeros(nTurbs)
        minSpacing = 2.  # number of rotor diameters

        # print("N Turbines: ", nTurbs)

        # define initial values
        for turbI in range(0, nTurbs):
            rotorDiameter[turbI] = rotor_diameter  # m
            hubHeight[turbI] = hub_height  # m
            axialInduction[turbI] = 1.0 / 3.0
            # Ct[turbI] = 0.805 #4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
            # Ct[turbI] = 0.77 #4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
            Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])

            # print(Ct)
            # quit()
            # Cp[turbI] = (0.7737 / 0.944) * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)

            # Ct[turbI] = 0.803
            # axialInduction[turbI] = ct_to_axial_ind_func(Ct[turbI])
            # print(Ct)
            # quit()
            # Cp[turbI] = (0.7737 / 0.944) * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
            Cp[turbI] = 4.0 * axialInduction[turbI] * np.power((1. - axialInduction[turbI]), 2)

            # generatorEfficiency[turbI] = 0.944
            generatorEfficiency[turbI] = 1.0
            yaw[turbI] = 0.0  # deg.

        # Define flow properties
        windDirections = np.linspace(173., 353.0, nDirections) #np.linspace(-15., 15., nDirections) + 270.0
        windFrequencies = np.ones(1)


        # print(windDirections)
        # print(windFrequencies)
        # air_density = 1.1716  # kg/m^3
        # air_density = 1.225  # kg/m^3


        wind_speed = 8.0  # m/s
        windSpeeds = np.ones(size) * wind_speed

        if MODELS[model] == 'BPA':
            # initialize problem
            prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                                  minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                                  wake_model=gauss_wrapper, params_IdepVar_func=add_gauss_params_IndepVarComps,
                                                  params_IdepVar_args={'nRotorPoints': nRotorPoints}, wake_model_options=wake_model_options,
                                                  cp_points=cp_curve_cp.size, cp_curve_spline=cp_curve_spline))
        elif MODELS[model] == 'FLORIS':
            # initialize problem
            prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                                  minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                                  wake_model=floris_wrapper,
                                                  params_IdepVar_func=add_floris_params_IndepVarComps,
                                                  params_IdepVar_args={}))
        elif MODELS[model] == 'JENSEN':
            # initialize problem
            prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                                  minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
                                                  wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                                                  params_IdepVar_func=add_jensen_params_IndepVarComps,
                                                  params_IdepVar_args={}))
        else:
            ValueError('The %s model is not currently available. Please select BPA or FLORIS' %(MODELS[model]))

        tic = time.time()
        prob.setup(check=False)
        toc = time.time()

        # print the results
        print('Problem setup took %.03f sec.' % (toc - tic))

        tic = time.time()
        # time.sleep(10)
        # assign initial values to design variables
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        for direction_id in range(0, windDirections.size):
            prob['yaw%i' % direction_id] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['hubHeight'] = hubHeight
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['cut_in_speed'] = np.ones(nTurbines)*4.
        # prob['cut_in_speed'] = np.ones(nTurbines)*7.
        prob['rated_power'] = np.ones(nTurbines)*2000.
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_wind_speed'] = cp_curve_wind_speed
        prob['use_power_curve_definition'] = False
        # prob['cp_curve_spline'] = cp_curve_spline

        if MODELS[model] is 'BPA':
            # prob['generatorEfficiency'] = np.ones(nTurbines)
            prob['model_params:wake_combination_method'] = wake_combination_method
            prob['model_params:ti_calculation_method'] = ti_calculation_method
            prob['model_params:calc_k_star'] = calc_k_star
            prob['model_params:sort'] = sort_turbs
            prob['model_params:z_ref'] = z_ref
            prob['model_params:z_0'] = z_0
            prob['model_params:ky'] = k_calc
            prob['model_params:kz'] = k_calc
            prob['model_params:print_ti'] = True
            prob['model_params:wake_model_version'] = wake_model_version
            # prob['model_params:I'] = TI
            # prob['model_params:shear_exp'] = shear_exp
            if nRotorPoints > 1:
                if rotor_pnt_typ == 0:
                    prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = circumference_points(nRotorPoints, location=location)
                if rotor_pnt_typ == 1:
                    prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)
        if MODELS[model] is 'JENSEN':
            prob['model_params:alpha'] = 0.1
            # prob['model_params:spread_angle'] = 10.0

        # set options
        # prob['floris_params:FLORISoriginal'] = True
        # prob['floris_params:CPcorrected'] = False
        # prob['floris_params:CTcorrected'] = False

        # run the problem
        print('start %s run' %(MODELS[model]))
        tic = time.time()
        # cProfile.run('prob.run()')
        prob.run_model()
        toc = time.time()
        print("total run time for ti=%s was %0.3f sec. " % (ti_calculation_method, toc-tic))


        # for direction_id in range(0, windDirections.size):
            # print('yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
            # for direction_id in range(0, windDirections.size):
            # print( 'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
        #     print( 'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

        # print('turbine X positions in wind frame (m): %s' % prob['turbineX'])
        # print('turbine Y positions in wind frame (m): %s' % prob['turbineY'])
        # print('turbine hub wind velcities (m/s): %s' % prob['wtVelocity0'])
        # print('wind farm power in each direction (kW): %s' % prob['dirPowers'])
        # print('AEP (kWh): %s' % prob['AEP'])

        # max_power = niayifar_power_model(wind_speed) * nTurbines
        max_power_NPA = niayifar_power_model(max(prob['wtVelocity0']))*nTurbines
        max_power_OURS = max(prob['wtPower%i' % 0])*nTurbines
        # max_power = max(prob['wtPower0'])*nTurbines
        # max_power = np.interp(wind_speed, power_curve_data[:, 0],

        # normalized_power[ti_calculation_method, :] = prob['dirPowers']
        for dir in np.arange(0, nDirections):
            normalized_power_NPA[ti_calculation_method, dir] = np.sum(niayifar_power_model(prob['wtVelocity%i' % dir]))/max_power_NPA
            normalized_power_OURS[ti_calculation_method, dir] = prob['dirPowers'][dir]/max_power_OURS


        # fig, ax = plt.subplots()
        # for x, y in zip(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter):
        #     circle_end = plt.Circle((x,y), 0.5, facecolor='none', edgecolor='k', linestyle='-', label='Turbines')
        #     ax.add_artist(circle_end)
        # # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
        # # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
        #
        # for i in range(0, nTurbs):
        #     ax.plot([turbineX[i] / rotor_diameter, prob['turbineX'][i] / rotor_diameter],
        #             [turbineY[i] / rotor_diameter, prob['turbineY'][i] / rotor_diameter], '--k')
        # plt.axis('equal')
        # # plt.show()

        for q in np.linspace(0,180,10):
            # print(prob['wtVelocity%i' % q], prob['wtPower%i' % q])
            if np.any(prob['wtVelocity%i' % q] < np.any(prob['cut_in_speed'])):
                print('below cut in at direction %i' % q)
                print(prob['wtVelocity%i' % q][prob['wtVelocity%i' % q] < prob['cut_in_speed']], prob['wtPower%i' % q][prob['wtVelocity%i' % q] < prob['cut_in_speed']])



    tic = time.time()
    # print(max_power, normalized_power)

    data_directory = "./inputfiles/"
    power_data_les = np.loadtxt(data_directory+"power_by_direction_niayifar_les.txt", delimiter=",")
    power_data_model = np.loadtxt(data_directory+"power_by_direction_niayifar_model.txt", delimiter=",")

    # power_curve_data = np.loadtxt(data_directory + "power_curve_v80.txt", delimiter=",")


    #                       power_curve_data[:, 1]) * 1E3 * nTurbines
    # plt.scatter(power_curve_data[:, 0], power_curve_data[:, 1]*1E3)
    # plt.plot(power_curve_data[:, 0], niayifar_power_model(power_curve_data[:, 0]))
    # plt.show()
    # print(max_power, normalized_power)
    # print(velocity_data[:, 0])
    # quit()
    fig, ax = plt.subplots(1)

    ax.plot(power_data_les[:, 0], power_data_les[:, 1], label="Niayifar LES", c='r', marker='o')
    ax.plot(power_data_model[:, 0], power_data_model[:, 1], label="Niayifar Model", c='b')
    ax.plot(windDirections, normalized_power_OURS[0, :], "--g", label="Ours No local TI")
    ax.plot(windDirections, normalized_power_OURS[9, :], "--k", label="Ours Jensen Fortran")
    # ax.plot(windDirections, normalized_power[2, :]/max_power, "-k", label="NPA TI")
    # ax.plot(windDirections, normalized_power[3, :]/max_power, "--c", label="NPA TI w/SM")
    # ax.plot(windDirections, normalized_power_NPA[4, :], "k", label="NPA")
    # ax.plot(windDirections, normalized_power_NPA[5, :], "--k", label="NPA w/SM")
    ax.plot(windDirections, normalized_power_OURS[4, :], "c", label="Ours")
    ax.plot(windDirections, normalized_power_OURS[5, :], "--c", label="Ours w/SM")

    ax.set_ylim([0.4, 1.0])
    ax.set_xlabel('Wind Direction (deg)')
    ax.set_ylabel('Normalized Power ($P_i/P_0$)')
    # ax.set_ylim([0.0, 1.2])
    ax.legend(ncol=2, loc=4)#, bbox_to_anchor=(1., 0.15))

    plt.tight_layout()

    plt.savefig("power-by-directions.pdf", transparent=True)

    #
    # ax[1].legend(loc=2)

    toc = time.time()

    print("plot creation took %0.3f sec." % (toc-tic))

    plt.show()


    # np.savetxt("power_direction_res_%irotorpts_%imodel.txt" % (nRotorPoints, wake_model_version),
    #            np.c_[windDirections, normalized_power_OURS[0, :],
    #                  normalized_power_OURS[4, :], normalized_power_OURS[5, :]],
    #            header="wind direction, no loc ti, our power, our power w/SM (%i rotor pts, %i model)" % (
    #            nRotorPoints, wake_model_version))

    