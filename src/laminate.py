from lamina import Lamina
from dataclasses import dataclass
from properties import StateProperties
from conversion import (
    create_tensor_3D,
    T_z,
    to_epsilon,
    to_gamma,
    tensor_to_vec,
    transformation_3D,
)

import numpy as np
from typing import List


@dataclass
class LaminateProperties:
    thickness: float = 0.0
    num_layers: int = 0
    length: float = 0.0
    width: float = 0.0


class Laminate:
    def __init__(self, length=0, width=0):

        self._num_plys = 0
        self.props = LaminateProperties(length=length, width=width)
        self._thickness = 0
        self._length = length
        self._width = width
        self.lamina: List[Lamina] = []
        self.global_state: List[StateProperties] = []

    def __str__(self):

        desc = f'''
        - Layers: {self._num_plys}
        - Orientation:  {'/'.join([str(theta) for theta in self._layers['orientation']])}
        '''
        return desc

    def add_lamina(self, new_lamina: Lamina, orientation_deg: float = 0) -> None:
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 

            Parameters:
                new_lamina (Lamina):         the constructed lamina object to be added to the laminate.
                orientation_deg (numpy.ndarray): orientation of the fiber directions in degrees relative to the 
                                                 principal axes in the z, y, x directions. 
        '''

        # Create a copy to allow a lamina to be reused multiple times in a laminate stack
        lamina_copy = new_lamina.copy()

        # Sets the orientation to calculate the transformed matrices
        lamina_copy.set_orientation(orientation_deg)

        self.lamina.append(lamina_copy)

        # Update laminate properties
        self.props.thickness += lamina_copy.props.thickness
        self.props.num_layers += 1

    def apply_stress(self, global_stress_tensor: np.ndarray) -> None:
        '''
        Assign global and local stress/strain state based on the given applied stress.

            Parameters:
                stress_tensor (numpy.ndarray):   Global stress tensor to be applied
        '''
        # Clear the current state
        temp_state = []

        # Iterate over each lamina in the stack
        for lamina in self.lamina:

            # Calculate the local stress and strain
            lamina.apply_stress(global_stress_tensor)

            # Calculate the global stress and strain vectors
            e_global = lamina.matrices.S_bar.dot(tensor_to_vec(global_stress_tensor))
            s_global = tensor_to_vec(global_stress_tensor)

            # Update laminate state properties list
            temp_state.append(StateProperties(s_global, e_global))

        self.global_state = temp_state

    def apply_strain(self, global_strain_tensor: np.ndarray) -> None:
        '''
        Assign global and local stress/strain state based on the given applied stress.

            Parameters:
                stress_tensor (numpy.ndarray):   Global stress tensor to be applied
        '''
        # Clear the current state
        temp_state = []

        # Iterate over each lamina in the stack
        for lamina in self.lamina:

            # Calculate the local stress and strain
            lamina.apply_strain(global_strain_tensor)

            # Calculate the global stress and strain vectors
            s_global = lamina.matrices.Q_bar.dot(tensor_to_vec(global_strain_tensor))
            e_global = tensor_to_vec(global_strain_tensor)

            # Update laminate state properties list
            temp_state.append(StateProperties(s_global, e_global))

        self.global_state = temp_state

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

        # Iterate over each lamina in the stack
        for lamina in self.lamina:

            # Retrieve the selected layer's orientation
            theta_deg = lamina.props.orientation

            lamina.apply_stress(stress_tensor)
            # print(lamina.matrices.S_bar.dot(tensor_to_vec(stress_tensor)))
            # Transform global to local lamina stress
            s_local = transformation_3D(stress_tensor, T_z, theta=theta_deg)

            # Convert local stress to local strain and store the value
            local_lamina_strain = lamina.stress2strain(s_local)

            # Epsilon tensor of local lamina strain
            e_local = to_epsilon(create_tensor_3D(*local_lamina_strain))

            # Gamma values of global laminate strain
            e_global = to_gamma(transformation_3D(e_local, T_z, theta=-theta_deg))

            # Convert global stress and strain to vector
            e_global = tensor_to_vec(e_global)
            s_global = tensor_to_vec(stress_tensor)

            # Store the global stress and strain state
            self.global_state.append(StateProperties(s_global, e_global))

            # Store the local lamina stress and strain state
            print(StateProperties(tensor_to_vec(s_local), local_lamina_strain))
            # lamina.local_state = StateProperties(s_local, local_lamina_strain)

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
                strain_tensor, T_z, theta=theta_deg
            )
            self._layers['local_strain'][i] = tensor_to_vec(local_lamina_strain)

            # Convert local strain to local stress and store the value
            local_lamina_strain = to_gamma(local_lamina_strain)
            local_lamina_stress = lamina.strain2stress(local_lamina_strain)
            self._layers['local_stress'][i] = local_lamina_stress

            # Global laminate stress
            local_lamina_stress = create_tensor_3D(*local_lamina_stress)
            sigma_global = self.transformation_3D(
                local_lamina_stress, T_z, theta=-theta_deg
            )
            sigma_global = tensor_to_vec(sigma_global)

            # Store the global laminate stress
            self._layers['global_stress'][i] = sigma_global

        return sigma_global
