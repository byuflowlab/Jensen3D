import unittest
import numpy as np

from openmdao.api import Group
from plantenergy.OptimizationGroups import AEPGroup

# from fusedwake.WindTurbine import WindTurbine
# from fusedwake.WindFarm import WindFarm
# from windIO.Plant import WTLayout, yaml

from openmdao.api import Problem

class test_jensen_tophat(unittest.TestCase):

    def setUp(self):
        try:
            from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

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
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.1                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        wake_model_options = None
        prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                                     params_IdepVar_func=add_jensen_params_IndepVarComps,
                                     params_IndepVar_args={'use_angle': False}))

        # initialize problem
        prob.setup(check=True)

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
        # prob['model_params:spread_angle'] = 20.0
        # prob['model_params:alpha'] = 0.1

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "jensen_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'],  np.array([8.1, 8.1, 6.74484, 6.74484, 6.616713, 6.616713]))

class test_jensencosine(unittest.TestCase):

    def setUp(self):
        try:
            from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

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
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.1                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        wake_model_options = {'variant': 'Cosine'}
        prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                                     params_IdepVar_func=add_jensen_params_IndepVarComps,
                                     params_IndepVar_args={'use_angle': False}))

        # initialize problem
        prob.setup(check=True)

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
        # prob['model_params:spread_angle'] = 20.0
        # prob['model_params:alpha'] = 0.1

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "jensen_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'],  np.array([8.1, 8.1, 6.982647, 6.982647, 6.886741, 6.879763]))

if __name__ == "__main__":
    unittest.main()