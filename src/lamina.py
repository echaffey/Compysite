from material import Material

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, List
from properties import StateProperties
from compositeMaterial import CompositeMaterial
from conversion import (
    to_epsilon,
    to_gamma,
    tensor_to_vec,
    T_z,
    transformation_3D,
    ConversionMatrices,
)
import copy


@dataclass
class LaminaProperties:
    material: Material = None
    material_fiber: Material = None
    material_matrix: Material = None
    Vol_f: float = 0.0
    Vol_m: float = 1.0
    thickness: float = 0.0
    orientation: float = 0.0


class Lamina(CompositeMaterial):
    def __init__(
        self,
        mat_fiber: Material = None,
        mat_matrix: Material = None,
        mat_composite: Material = None,
        Vol_fiber: float = 0.0,
        Vol_matrix: float = 1.0,
        thickness: float = 0.0,
        array_geometry: int = 1,
    ):
        '''
        Create a single lamina using known fiber and matrix materials or assigned with a predetermined composite material.

        Args:
            mat_fiber (MaterialProperties, optional): Fiber material object. Defaults to None.
            mat_matrix (MaterialProperties, optional): Matrix material object. Defaults to None.
            mat_composite (MaterialProperties, optional): Composite material object. Defaults to None.
            Vol_fiber (float, optional): Fiber volume fraction. Defaults to 0.0.
            Vol_matrix (float, optional): Matrix volume fraction. Defaults to 1.0.
            thickness (float, optional): Lamina thickness. Defaults to 0.0.
            array_geometry (int, optional): Matrix array geometry constant.  1 = Hexagonal array, 2 = Square array. Defaults to 1.
        '''

        if not Vol_fiber:
            Vol_fiber = 1 - Vol_matrix

        # Create the composite from the fiber and matrix materials if a composite is not given
        # Alternatively, if only a matrix is given, its a uniform material
        if mat_composite is None:
            if (mat_fiber is not None) & (mat_matrix is not None):

                # Create composite from the fiber and matrix materials
                material = self._create_composite(
                    mat_fiber, mat_matrix, Vol_fiber, array_geometry
                )

            elif mat_matrix is not None:
                material = mat_matrix

            else:
                material = mat_composite
                print('You must create at least one material that is not a fiber.')
        else:
            material = mat_composite

        # Set the lamina properties to the passed in values
        self.props = LaminaProperties(
            material, mat_fiber, mat_matrix, Vol_fiber, Vol_matrix, thickness, 0,
        )

        # Initialize the stress and strain states
        self.local_state: StateProperties = StateProperties()

        # Initialize the compliance and stiffness matrices with default orientation at 0 degrees
        self.matrices = ConversionMatrices(material)

    def copy(self):
        return copy.deepcopy(self)

    def set_orientation(self, theta_deg: float = 0.0) -> None:
        '''
        Sets the orientation of the lamina within the laminate stack.

        Args:
            theta_deg (float, optional): Lamina orientation in degrees. Defaults to 0.0
        '''

        # Store orientation in radians
        self.props.orientation = theta_deg * np.pi / 180

        # Updates transformation matrices with new orientation
        self.matrices.update_orientation(self.props.orientation)

    def apply_stress(self, global_stress_tensor: np.ndarray) -> None:
        '''
        Updates the local stress and strain state from a given globally applied stress.

        Args:
            stress_tensor (np.ndarray): Global stress tensor being applied.
        '''

        # Global lamina orientation
        theta_rad = self.props.orientation

        # Convert global to local stress
        local_stress_tensor = transformation_3D(
            global_stress_tensor, T_z, theta_rad, theta_radians=True
        )

        local_stress = tensor_to_vec(local_stress_tensor)

        # Create compliance matrix
        S = self.matrices.S

        local_strain = S.dot(local_stress)

        self.local_state = StateProperties(local_stress, local_strain)

    def apply_strain(self, global_strain_matrix: np.ndarray) -> None:
        '''
        Updates the local stress and strain state from a given globally applied stress.

        Args:
            stress_tensor (np.ndarray): Global stress tensor being applied.
        '''

        # Global lamina orientation
        theta_rad = self.props.orientation

        global_strain_matrix = to_epsilon(global_strain_matrix)

        # Convert global to local stress
        local_strain_tensor = transformation_3D(
            global_strain_matrix, T_z, theta_rad, theta_radians=True
        )

        local_strain = to_gamma(local_strain_tensor)
        local_strain = tensor_to_vec(local_strain)

        C = self.matrices.C
        local_stress = C.dot(local_strain)

        self.local_state = StateProperties(local_stress, local_strain)

    def apply_2D_boundary_conditions(
        self,
        stress_tensor: np.ndarray,
        direction: int = 1,
        additional_strain: list = [],
    ) -> np.ndarray:
        '''
        Calculate the resulting total strain from a boundary condition applied to a material experiencing applied stresses. Allows for
        the inclusion of non-mechanical strains to be applied within the boundary conditions. 

        Parameters:
            stress_tensor (numpy.ndarray): Stress tensor representing all of the applied stresses.
            E (numpy.ndarray): Vector of the effective composite elastic modulii in the principal directions [E1, E2, E3]
            v (numpy.ndarray): Vector of the effective composite Poisson's ratios in the principal directions [v23, v13, v12]
            G (numpy.ndarray): Vector of the effective composite shear modulii in the principal directions [G23, G13, G12]
            direction (int): The direction or directions in which the composite is constrained, optional. Defaults to the longitudinal direction.
            additional_strain (list): List of additional strain vectors to be applied alongside the mechanical strain, optional. 

        Returns:
            total_strain (numpy.ndarray): Vector containing the total normal and shear strain values. Shear is reported in terms of gamma. [e1, e2, e3, g23, g13, g12]

        '''

        # Ensure that the dimensions of the thermal and moisture vectors are the correct length
        for i, strain in enumerate(additional_strain):
            if len(strain) < 6:
                additional_strain[i] = np.append(strain, np.zeros(6 - len(strain)))

        # Get the composite compliance matrix
        S = self.matrices.S

        # Create applied stress vector [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        stress_vec = tensor_to_vec(stress_tensor)

        # Create vector representing boundary conditions where 1=applied stress, 0=no applied stress
        bc = np.zeros_like(stress_vec)

        for i, v in enumerate(stress_vec):
            if v != 0:
                bc[i] = 1

        # Set the constrained direction stress value to 1 in the stress vector.
        # This is so that the compliance matrix value at that point is preserved.
        # Example:
        # vec = [0, 125e6, 0, 0, 0, 0] -> original stress vector
        # vec = [0, 125e6, 1, 0, 0, 0] -> constrained in direction 3 so sigma_3's value is preserved
        stress_vec[direction - 1] = 1

        # Slice just the row pertaining to the unknown stress (constrained direction) and multiply it by the stress vector
        # This calculates the mechanical strain at the applied stress direction and leaves the unknown compliance value
        # From above example, only the row related to direction 2 (125e6) is preserved
        # S = [0, 125e6*S_22, 1*S_23, 0, 0, 0]
        _S = S[direction - 1, :] * stress_vec

        # Factor in additional strains experienced by the composite (thermal, hydro, etc)
        net_strain = 0
        for add_strain in additional_strain:
            net_strain += add_strain[direction - 1]

        # Solve for the stress in the constrained direction as a result of all acting strains
        # From above example with added thermal strain:
        # epsilon_3 = [0*S_13 + sigma_2*S_23 + sigma_3*S_33] + [alpha*dT]
        #         0 = [0 + 125e6*S_22  + S_23*sigma_3]       + [alpha*dT]
        # -alpha*dT = 125e6*S_22 + S_23*sigma_3
        #   sigma_3 = -(125e6*S_22 + alpha*dT)/S_23

        sigma_c = -(_S.dot(bc) + net_strain) / _S[direction - 1]

        # Put the solved for stress back into the stress vector
        stress_vec[direction - 1] = sigma_c

        # Solve for the strain values
        # epsilon = S * sigma
        _total_strain = S.dot(stress_vec)

        # Add in non-mechanical strains and then set total strain to zero in the constrained direction
        # because constraints don't allow for changes in dimension which means that strain is zero
        for add_strain in additional_strain:
            _total_strain += add_strain

        _total_strain[direction - 1] = 0

        # All stresses acting on the system
        _total_stress = stress_vec

        return _total_stress, _total_strain

    def get_lamina_properties(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:

        E, v, G = self.props.material.get_properties()

        return E, v, G

    def get_lamina_expansion_properties(
        self,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:

        alpha, beta = self.props.material.get_expansion_properties()

        return alpha, beta

    def get_material(self) -> Material:

        return self.props.material

    def plot_compliance(self, range_theta_rad):

        fig, (ax1, ax2) = plt.subplots(1, 2)

        S11, S22, S12, S66, S26, S16 = [], [], [], [], [], []
        for theta in range_theta_rad:
            S11.append(self.compliance_matrix(theta)[0, 0])
            S22.append(self.compliance_matrix(theta)[1, 1])
            S12.append(self.compliance_matrix(theta)[0, 1])
            S66.append(self.compliance_matrix(theta)[-1, -1])
            S26.append(self.compliance_matrix(theta)[1, -1])
            S16.append(self.compliance_matrix(theta)[0, -1])

        theta_range = range_theta_rad * 180 / np.pi
        ax1.plot(theta_range, S11)
        ax1.plot(theta_range, S22)
        ax1.plot(theta_range, S12)
        ax1.plot(theta_range, S66)
        ax2.plot(theta_range, S26)
        ax2.plot(theta_range, S16)

        ax1.legend(['S11', 'S22', 'S12', 'S66'])
        ax2.legend(['S26', 'S16'])
        plt.show()
