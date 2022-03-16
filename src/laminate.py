from threading import local
from lamina import Lamina
from material import Material
from utils import create_tensor_3D, T_z, to_epsilon, to_gamma, tensor_to_vec

import numpy as np
from typing import Union


class Laminate:

    def __init__(self, length=0, width=0):

        self._num_plys = 0
        self._layers = {'lamina': [],
                        'orientation': [],
                        'material': [],
                        'local_stress': [None],
                        'local_strain': [None],
                        'global_stress': [None],
                        'global_strain': [None],
                        'global_props': [None]}
        self._thickness = 0
        self._length = length
        self._width = width

    def __str__(self):

        desc = f'''
        - Layers: {self._num_plys}
        - Orientation:  {'/'.join([str(theta) for theta in self._layers['orientation']])}
        '''
        return desc

    def add_lamina(self, new_lamina: Lamina, orientation: float = 0) -> None:
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 

            Parameters:
                new_lamina (Lamina):         the constructed lamina object to be added to the laminate.
                orientation (numpy.ndarray): orientation vector of the fiber directions relative to the 
                                             principal axes in the z, y, x directions. 
        '''

        # Initialize new layer properties
        self._layers['lamina'].append(new_lamina)
        self._layers['orientation'].append(orientation)
        self._layers['material'].append(new_lamina.get_material())
        self._layers['local_stress'].append(None)
        self._layers['local_strain'].append(None)
        self._layers['global_stress'].append(None)
        self._layers['global_strain'].append(None)

        self._thickness += new_lamina.props.thickness
        self._num_plys += 1

        # Sets the orientation to calculate the transformed matrices
        new_lamina.set_orientation(orientation)

    def transformation_3D(self, tensor, rot_matrix, theta, theta_radians=False):
        '''
        Return the transformed 3D tensor. Shear outputs are in terms of epsilon.

            Parameters:
                tensor (numpy.ndarray):        Cauchy tensor
                rot_matrix (numpy.ndarray):    Rotation matrix
                theta (float):                 Angle of rotation
                radians (bool):                True if theta is given in radians 

            Returns:
                prime (numpy.ndarray):   Transformed matrix
        '''

        _tensor = tensor.copy()

        # Convert to radians and evaluate the rotation matrix
        _theta = theta if theta_radians else theta * np.pi/180
        _R = rot_matrix(_theta)

        # Transformation equation
        _prime = _R.dot(_tensor).dot(_R.T)

        return _prime

    def stress2strain(self, stress_tensor: np.ndarray) -> np.ndarray:
        '''
        Conversion from global stress tensor to global strain vector.

            Parameters:
                stress_tensor (numpy.ndarray):   Stress tensor 
                elasticity_mod (numpy.ndarray):  Young's modulus [E1, E2, E3]
                shear_mod (numpy.ndarray):       Shear modulus [G23, G13, G12]
                poissons_ratio (numpy.ndarray):  Poisson's ratio [v23, v13, v12]

            Returns:
                strain_vec (numpy.ndarray):  Strain vector [E_1, E_2, E_3, g_23, g_13, g_12]
        '''

        for i, lamina in enumerate(self._layers['lamina']):

            # Retrieve the selected layer's orientation
            theta_deg = self._layers['orientation'][i]
            self._layers['global_stress'][i] = tensor_to_vec(stress_tensor)

            # Transform global to local lamina stress
            local_lamina_stress = self.transformation_3D(
                stress_tensor, T_z, theta=theta_deg)

            # Store the layer's stress state
            self._layers['local_stress'][i] = tensor_to_vec(
                local_lamina_stress)

            # Convert local stress to local strain and store the value
            local_lamina_strain = lamina.stress2strain(local_lamina_stress)
            self._layers['local_strain'][i] = local_lamina_strain

            # Epsilon tensor of local lamina strain
            e_local = to_epsilon(create_tensor_3D(*local_lamina_strain))

            # Gamma values of global laminate strain
            e_global = to_gamma(self.transformation_3D(
                e_local, T_z, theta=-theta_deg))

            # Convert back to vector
            e_global = tensor_to_vec(e_global)

            # Store the global laminate strain
            self._layers['global_strain'][i] = e_global

        return e_global

    def strain2stress(self, strain_tensor: np.ndarray) -> np.ndarray:
        '''
        Conversion from strain tensor to stress vector. 
        Strain must be in terms of gamma so pre-mulitiply the epsilon values by 2 for state of strain.
        State of strain is a tensor in terms of epsilon.

            Parameters:
                strain_tensor (numpy.ndarray):   Strain tensor in terms of gamma
                elasticity_mod (numpy.ndarray):  Young's modulus [E1, E2, E3]
                shear_mod (numpy.ndarray):       Shear modulus [G23, G13, G12]
                poissons_ratio (numpy.ndarray):  Poisson's ratio [v23, v13, v12]

            Returns:
                stress_vec (numpy.ndarray):  Stress vector [s_1, s_2, s_3, t_23, t_13, t_12] 
        '''

        for i, lamina in enumerate(self._layers['lamina']):

            theta_deg = self._layers['orientation'][i]
            self._layers['global_strain'][i] = tensor_to_vec(strain_tensor)

            # Transform global to local lamina strain and store it
            strain_tensor = to_epsilon(strain_tensor)
            local_lamina_strain = self.transformation_3D(
                strain_tensor, T_z, theta=theta_deg)
            self._layers['local_strain'][i] = tensor_to_vec(
                local_lamina_strain)

            # Convert local strain to local stress and store the value
            local_lamina_strain = to_gamma(local_lamina_strain)
            local_lamina_stress = lamina.strain2stress(local_lamina_strain)
            self._layers['local_stress'][i] = local_lamina_stress

            # Global laminate stress
            local_lamina_stress = create_tensor_3D(*local_lamina_stress)
            sigma_global = self.transformation_3D(
                local_lamina_stress, T_z, theta=-theta_deg)
            sigma_global = tensor_to_vec(sigma_global)

            # Store the global laminate stress
            self._layers['global_stress'][i] = sigma_global

        return sigma_global

    def apply_stress(self, tensor_applied: np.ndarray) -> Union[np.ndarray, np.ndarray]:

        sigma = 0
        epsilon = 0
        return sigma, epsilon
