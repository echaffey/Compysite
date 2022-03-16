from material import Material

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from properties import LaminaProperties, StateProperties, ConversionMatrices, MaterialProperties


class Lamina:

    def __init__(self,
                 mat_fiber: MaterialProperties = None,
                 mat_matrix: MaterialProperties = None,
                 mat_composite: MaterialProperties = None,
                 Vol_fiber: float = 0.0, Vol_matrix: float = 1.0,
                 thickness: float = 0.0, array_geometry: int = 1):
        '''Create a single lamina using known fiber and matrix materials or assigned with a predetermined composite material.

        Parameters:
            mat_fiber (Material):  [Optional] Fiber material object.
            mat_matrix (Material):  [Optional] Matrix material object. 
            mat_composite (Material):  [Optional] Composite material object.
            Vol_fiber (float):  [Optional] Fiber volume fraction.
            Vol_matrix (float):  [Optional] Matrix volume fraction.
            array_geometry (int):  [Optional] Matrix array geometry constant.  1 = Hexagonal array, 2 = Square array.
        '''

        # Create the composite from the fiber and matrix materials if a composite is not given
        # Alternatively, if only a matrix is given, its a uniform material
        if mat_composite is None:
            if (mat_fiber is not None) & (mat_matrix is not None):

                # -------------------
                # Need to add beta calculations
                # -------------------

                # Create composite from the fiber and matrix materials
                material = self._create_composite(
                    mat_fiber, mat_matrix, array_geometry)

            elif mat_matrix is not None:
                material = mat_matrix

            else:
                material = mat_composite
                print('You must create at least one material that is not a fiber.')
        else:
            material = mat_composite

        self.props = LaminaProperties(
            material.props, mat_fiber, mat_matrix, Vol_fiber, Vol_matrix, thickness, 0)
        self.state = StateProperties(0, 0)
        self.matrices = ConversionMatrices(self.props.material)

    def set_orientation(self, theta_deg: float = 0) -> None:

        # Store orientation in radians
        self.props.orientation = theta_deg*np.pi/180

        # Updates transformation matrices with new orientation
        self.matrices.update_orientation(self.props.orientation)

    def _halpin_tsai(self, M_f, M_m, V_f, array_geometry=1) -> float:
        '''
        Calculates the Halpin-Tsai prediction for the in-plane and transverse elastic or shear modulus of the composite material.

            Parameters:
                M_f  (float): Elastic/shear modulus of the fiber material
                M_m  (float): Elastic/shear modulus of the matrix material
                V_f  (float): Fiber volume fraction
                xi   (int):   Geometric constant where 1=Hexagonal array, 2=Square array
            Returns:
                M_12    (float): In-plane elastic/shear modulus
        '''

        xi = array_geometry

        # Halpin Tsai finds transverse/in-plane modulus so only E_2 or G_13=G_12 is used
        M_f = M_f[1]
        M_m = M_m[1]

        # Ratio of fiber modulus to matrix modulus
        _M = M_f/M_m

        # Proportionality constant based on the array geometry
        _n = (_M - 1)/(_M + xi)

        # Transverse modulus
        _M_2 = M_m * (1 + xi*_n*V_f)/(1 - _n*V_f)

        return _M_2

    def _composite_shear_mod(self, mat_fiber, mat_matrix, array_geometry=1) -> np.ndarray:
        '''
        Calculates the equivalent composite shear modulus in the 3 principal directions.
            *This assumes that the transverse directions are isotropic*

            Parameters:
                E_m (np.ndarray, float): Elastic modulus of the matrix material
                v_f (np.ndarray, float): Poisson's ratio of the fiber material
                v_m (np.ndarray, float): Poisson's ratio of the matrix material
                G_f (np.ndarray, float): Shear modulus of the fiber material
                Vol_f (float):           Volume fraction of the fiber
                Vol_m (float):           Volume fraction of the matrix [Optional]

            Returns:
                G_3D (np.ndarray): 3x1 array of the Shear modulus in the principal directions
        '''

        # This needs value checking
        # -------------------
        _, _, G_f = mat_fiber.get_properties()
        E_m, v_m, G_m = mat_matrix.get_properties()

        Vol_f = self._Vol_f
        Vol_m = 1 - Vol_f
        # ----------------------

        xi = array_geometry

        # Create directional components if an isotropic value is given and
        # calculate the shear modulus for the matrix in each of the principal directions
        if np.sum(G_m) == 0:
            G_m = E_m/(2*(1+v_m))

        # Calculate the in-plane shear modulus using Halpin-Tsai formula
        G_12 = self._halpin_tsai(G_f, G_m, Vol_f, xi)

        # Set the poisson's ratio and shear modulus in the 23 direction
        v_m_23 = v_m[0]
        G_m_23 = G_m[0]
        G_f_23 = G_f[0]

        # Calculate the out of plane shear
        n_23 = (3 - 4*v_m_23 + G_m_23/G_f_23)/(4*(1 - v_m_23))
        _G_23 = G_m_23 * (Vol_f + n_23*Vol_m) / \
            (n_23*Vol_m + Vol_f*(G_m_23/G_f_23))

        # Composite is assume to be orthotropic
        _G_13 = G_12

        # Return an array containing the composite shear modulus
        return np.array([_G_23, _G_13, G_12])

    def _composite_poisson_ratio(self, E_c, G_c, mat_fiber, mat_matrix) -> np.ndarray:
        '''
        Calculates the equivalent composite Poisson's ratio in the 3 principal directions.
        *This assumes that the transverse directions are isotropic*

        Parameters:
            E (np.ndarray):          Elastic modulus of the composite
            v_f (np.ndarray, float): Poisson's ratio of the fiber
            v_m (np.ndarray, float): Poisson's ratio of the matrix
            G (np.ndarray):          Shear modulus of the composite
            Vol_f (float):           Volume fraction of the fiber

        Returns:
            v (np.ndarray): Equivalent composite Poisson's ratio in the principal directions
        '''

        E = E_c
        G = G_c

        _, v_f, _ = mat_fiber.get_properties()
        _, v_m, _ = mat_matrix.get_properties()

        # Calculate the matrix volume fraction
        Vol_f = self._Vol_f
        Vol_m = 1 - Vol_f

        # Rule of mixtures
        _v = v_f*Vol_f + v_m*Vol_m

        # Calculate Poisson's on the 23 plane
        _v[0] = E[2]/(2*G[0]) - 1

        return _v

    def _composite_elastic_mod(self, mat_fiber, mat_matrix, array_geometry=1) -> np.ndarray:
        '''
        Calculates the equivalent composite elastic modulus in the 3 principal directions. 
        *This assumes that the transverse directions are isotropic*

            Parameters:
                E_f (float): Elastic modulus of the fiber material
                E_m (float): Elastic modulus of the matrix material
                V_f (float): Volume fraction of the fiber
                V_m (float): Volume fraction of the matrix [Optional]
                xi    (int): Geometric array factor (1=hexagonal array, 2=square array)

            Returns:
                E_3D (np.ndarray): 3x1 array of the Elastic Modulus in the principal directions
        '''

        E_f, _, _ = mat_fiber.get_properties()
        E_m, _, _ = mat_matrix.get_properties()

        Vol_f = self._Vol_f
        Vol_m = 1 - Vol_f

        # Rule of mixtures
        _E_1 = E_f[0] * Vol_f + E_m[0] * Vol_m

        # Halpin-Tsai for transverse directions
        _E_2 = self._halpin_tsai(E_f, E_m, Vol_f, array_geometry)

        # Material assumed to be orthotropic
        _E_3 = _E_2

        return np.array([_E_1, _E_2, _E_3])

    def _composite_thermal_expansion(self, mat_fiber, mat_matrix) -> np.ndarray:
        '''
        alpha_f, alpha_m, E_f, E_m, v_f, v_m, G_f, Vol_f, xi=1
        '''

        # Calculate matrix volume fraction
        Vol_m = 1 - self._Vol_f

        # Calculate the effective composite material properties
        _E, _v, _ = self.get_lamina_properties()
        E_f, v_f, _ = mat_fiber.get_properties()
        E_m, v_m, _ = mat_matrix.get_properties()
        alpha_f, _ = mat_fiber.get_expansion_properties()
        alpha_m, _ = mat_matrix.get_expansion_properties()

        # Calculate the effective thermal expansion constant of the composite
        _alpha_1 = 1/_E[0] * (alpha_f[0]*E_f[0] *
                              self._Vol_f + alpha_m[0]*E_m[0]*Vol_m)

        if self._Vol_f > 0.25:
            _alpha_2 = alpha_f[1]*self._Vol_f + (1 + v_m[1])*alpha_m[1]*Vol_m
        else:
            _alpha_2 = (1 + v_f[1])*alpha_f[1]*self._Vol_f + \
                (1+v_m[1])*alpha_m[1]*Vol_m-_alpha_1*_v[2]

        _alpha_3 = _alpha_2

        return np.array([_alpha_1, _alpha_2, _alpha_3])

    def _create_composite(self, mat_fiber, mat_matrix, array_geometry=1) -> Material:
        '''
        Returns the effective material properties (elastic modulus, Poisson's ratio and shear modulus) for the composite material.

        Parameters:
            E_f (numpy.ndarray, float): Elastic modulus of the fiber.
            E_m (numpy.ndarray, float): Elastic modulus of the matrix.
            v_f (numpy.ndarray, float): Poisson's ratio of the fiber.
            v_m (numpy.ndarray, float): Poisson's ratio of the matrix.
            G_f (numpy.ndarray, float): Shear modulus of the fiber.
            Vol_f (float):              Volume fraction of the fiber.
            xi (int):                   Halpin-Tsai geometric constant. (1 = hexagonal array, 2 = square array)

        Returns:
            properties (numpy.ndarray):   3 row matrix containing the elastic modulus, Poisson's ratio and shear modulus, respectively
        '''

        xi = array_geometry

        # Calculate the composite elastic modulus
        _E = self._composite_elastic_mod(
            mat_fiber, mat_matrix, array_geometry=xi)

        # Calculate the composite shear modulus
        _G = self._composite_shear_mod(
            mat_fiber, mat_matrix, array_geometry=xi)

        # Calculate the composite Poisson's ratio
        _v = self._composite_poisson_ratio(_E, _G, mat_fiber, mat_matrix)

        # Calculate the composite thermal expansion ratio
        _alpha = self._composite_thermal_expansion(mat_fiber, mat_matrix)

        # Return the created composite
        return Material(_E, _v, _G, _alpha)

    def get_lamina_properties(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:

        E, v, G = self.props.material.get_properties()

        return E, v, G

    def get_lamina_expansion_properties(self) -> Union[np.ndarray, np.ndarray, np.ndarray]:

        alpha, beta = self.props.material.get_expansion_properties()

        return alpha, beta

    def get_material(self) -> Material:

        return self.props.material

    def stress2strain(self, stress_tensor: np.ndarray) -> np.ndarray:
        '''
        Conversion from stress tensor to strain vector.

            Parameters:
                stress_tensor (numpy.ndarray):   Stress tensor 
                elasticity_mod (numpy.ndarray):  Young's modulus [E1, E2, E3]
                shear_mod (numpy.ndarray):       Shear modulus [G23, G13, G12]
                poissons_ratio (numpy.ndarray):  Poisson's ratio [v23, v13, v12]

            Returns:
                strain_vec (numpy.ndarray):  Strain vector [E_1, E_2, E_3, g_23, g_13, g_12]
        '''

        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(stress_tensor), stress_tensor[1, 2],
                        stress_tensor[0, 2], stress_tensor[0, 1]])

        # Create compliance matrix
        _S = self.matrices.S

        _strain_vec = _S.dot(_vec)

        return _strain_vec

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

        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(strain_tensor), strain_tensor[1, 2],
                        strain_tensor[0, 2], strain_tensor[0, 1]])

        # Create stiffness matrix
        _C = self.matrices.C

        stress_vec = _C.dot(_vec)

        return stress_vec

    def apply_2D_boundary_conditions(self, stress_tensor: np.ndarray, direction: int = 1, additional_strain: list = []) -> np.ndarray:
        '''
        Calculate the resulting total strain from a boundary condition applied to a material experiencing applied stresses. Allows for
        the inclusion of non-mechanical strains to be applied within the boundary conditions. 

        Parameters:
            stress_tensor (numpy.ndarray): Stress tensor representing all of the applied stresses.
            E             (numpy.ndarray): Vector of the effective composite elastic modulii in the principal directions [E1, E2, E3]
            v             (numpy.ndarray): Vector of the effective composite Poisson's ratios in the principal directions [v23, v13, v12]
            G             (numpy.ndarray): Vector of the effective composite shear modulii in the principal directions [G23, G13, G12]
            direction               (int): The direction or directions in which the composite is constrained, optional. Defaults to the longitudinal direction.
            additional_strain      (list): List of additional strain vectors to be applied alongside the mechanical strain, optional. 

        Returns:
            total_strain (numpy.ndarray): Vector containing the total normal and shear strain values. Shear is reported in terms of gamma. [e1, e2, e3, g23, g13, g12]

        '''

        # Ensure that the dimensions of the themal and moisture vectors are the correct length
        for i, strain in enumerate(additional_strain):
            if len(strain) < 6:
                additional_strain[i] = np.append(
                    strain, np.zeros(6-len(strain)))

        # Get the composite compliance matrix
        S = self.matrices.S

        # Create applied stress vector [sigma_1, sigma_2, sigma_3, tau_23, tau_13, tau_12]
        _vec = np.array([*np.diagonal(stress_tensor), stress_tensor[1,
                        2], stress_tensor[0, 2], stress_tensor[0, 1]])

        # Create vector representing boundary conditions where 1=applied stress, 0=no applied stress
        bc = np.zeros_like(_vec)
        for i, v in enumerate(_vec):
            if v != 0:
                bc[i] = 1

        # Set the constrained direction stress value to 1 in the stress vector.
        # This is so that the compliance matrix value at that point is preserved.
        # Example:
        # vec = [0, 125e6, 0, 0, 0, 0] -> original stress vector
        # vec = [0, 125e6, 1, 0, 0, 0] -> constrained in direction 3 so sigma_3's value is preserved
        _vec[direction-1] = 1

        # Slice just the row pertaining to the unknown stress (constrained direction) and multiply it by the stress vector
        # This calculates the mechanical strain at the applied stress direction and leaves the unknown compliance value
        # From above example, only the row related to direction 2 (125e6) is preserved
        # S = [0, 125e6*S_22, 1*S_23, 0, 0, 0]
        _S = S[direction-1, :]*_vec

        # Factor in additional strains experienced by the composite (thermal, hydro, etc)
        net_strain = 0
        for add_strain in additional_strain:
            net_strain += add_strain[direction-1]

        # Solve for the stress in the constrained direction as a result of all acting strains
        # From above example with added thermal strain:
        # epsilon_3 = [0*S_13 + sigma_2*S_23 + sigma_3*S_33] + [alpha*dT]
        #         0 = [0 + 125e6*S_22  + S_23*sigma_3]       + [alpha*dT]
        # -alpha*dT = 125e6*S_22 + S_23*sigma_3
        #   sigma_3 = -(125e6*S_22 + alpha*dT)/S_23

        sigma_c = -(_S.dot(bc) + net_strain)/_S[direction-1]

        # Put the solved for stress back into the stress vector
        _vec[direction-1] = sigma_c

        # Solve for the strain values
        # epsilon = S * sigma
        _total_strain = S.dot(_vec)

        # Add in non-mechanical strains and then set total strain to zero in the constrained direction
        # because constraints don't allow for changes in dimension which means that strain is zero
        for add_strain in additional_strain:
            _total_strain += add_strain

        _total_strain[direction-1] = 0

        # All stresses acting on the system
        _total_stress = _vec

        return _total_stress, _total_strain

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

        theta_range = range_theta_rad * 180/np.pi
        ax1.plot(theta_range, S11)
        ax1.plot(theta_range, S22)
        ax1.plot(theta_range, S12)
        ax1.plot(theta_range, S66)
        ax2.plot(theta_range, S26)
        ax2.plot(theta_range, S16)

        ax1.legend(['S11', 'S22', 'S12', 'S66'])
        ax2.legend(['S26', 'S16'])
        plt.show()
