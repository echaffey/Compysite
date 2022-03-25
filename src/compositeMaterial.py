import numpy as np
from material import Material


class CompositeMaterial:
    '''
    Provides the fundamental calculations and formulas necessary to create an 
    equivalent composite from the matrix and fiber components
    '''

    def _halpin_tsai(
        self, M_f: float, M_m: float, V_f: float, array_geometry: int = 1
    ) -> float:
        '''
        Calculates the Halpin-Tsai prediction for the in-plane and transverse 
        elastic or shear modulus of the composite material.

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
        _M = M_f / M_m

        # Proportionality constant based on the array geometry
        _n = (_M - 1) / (_M + xi)

        # Transverse modulus
        _M_2 = M_m * (1 + xi * _n * V_f) / (1 - _n * V_f)

        return _M_2

    def _composite_shear_mod(
        self,
        mat_fiber: Material,
        mat_matrix: Material,
        Vol_f: float,
        array_geometry: int = 1,
    ) -> np.ndarray:
        '''
        Calculates the equivalent composite shear modulus in the 3 principal directions.
            *This assumes that the transverse directions are isotropic*

        Args:
            mat_fiber (Material): Fiber material object.
            mat_matrix (Material): Matrix material object
            array_geometry (int, optional): Geometric constant where 
                                            1=Hexagonal array, 
                                            2=Square array. Defaults to 1.

        Returns:
            G_c (np.ndarray): Composite shear modulus vector. [G_23, G_13, G_12]
        '''

        # This needs value checking
        # -------------------
        _, _, G_f = mat_fiber.get_properties()
        E_m, v_m, G_m = mat_matrix.get_properties()

        Vol_m = 1 - Vol_f
        # ----------------------

        xi = array_geometry

        # Create directional components if an isotropic value is given and
        # calculate the shear modulus for the matrix in each of the principal directions
        if np.sum(G_m) == 0:
            G_m = E_m / (2 * (1 + v_m))

        # Calculate the in-plane shear modulus using Halpin-Tsai formula
        G_12 = self._halpin_tsai(G_f, G_m, Vol_f, xi)

        # Set the poisson's ratio and shear modulus in the 23 direction
        v_m_23 = v_m[0]
        G_m_23 = G_m[0]
        G_f_23 = G_f[0]

        # Calculate the out of plane shear
        n_23 = (3 - 4 * v_m_23 + G_m_23 / G_f_23) / (4 * (1 - v_m_23))
        _G_23 = (
            G_m_23 * (Vol_f + n_23 * Vol_m) / (n_23 * Vol_m + Vol_f * (G_m_23 / G_f_23))
        )

        # Composite is assume to be orthotropic
        _G_13 = G_12

        # Return an array containing the composite shear modulus
        return np.array([_G_23, _G_13, G_12])

    def _composite_poisson_ratio(
        self, E_c, G_c, mat_fiber, mat_matrix, Vol_f
    ) -> np.ndarray:
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
        Vol_m = 1 - Vol_f

        # Rule of mixtures
        _v = v_f * Vol_f + v_m * Vol_m

        # Calculate Poisson's on the 23 plane
        _v[0] = E[2] / (2 * G[0]) - 1

        return _v

    def _composite_elastic_mod(
        self, mat_fiber, mat_matrix, Vol_f, array_geometry=1
    ) -> np.ndarray:
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

        Vol_m = 1 - Vol_f

        # Rule of mixtures
        _E_1 = E_f[0] * Vol_f + E_m[0] * Vol_m

        # Halpin-Tsai for transverse directions
        _E_2 = self._halpin_tsai(E_f, E_m, Vol_f, array_geometry)

        # Material assumed to be orthotropic
        _E_3 = _E_2

        return np.array([_E_1, _E_2, _E_3])

    def _composite_thermal_expansion(
        self, mat_fiber, mat_matrix, Vol_f, E_c, v_c
    ) -> np.ndarray:
        '''
        alpha_f, alpha_m, E_f, E_m, v_f, v_m, G_f, Vol_f, xi=1
        '''

        # Calculate matrix volume fraction
        Vol_m = 1 - Vol_f

        # Calculate the effective composite material properties
        _E, _v, = E_c, v_c
        E_f, v_f, _ = mat_fiber.get_properties()
        E_m, v_m, _ = mat_matrix.get_properties()
        alpha_f, _ = mat_fiber.get_expansion_properties()
        alpha_m, _ = mat_matrix.get_expansion_properties()

        # Calculate the effective thermal expansion constant of the composite
        _alpha_1 = (
            1 / _E[0] * (alpha_f[0] * E_f[0] * Vol_f + alpha_m[0] * E_m[0] * Vol_m)
        )

        if Vol_f > 0.25:
            _alpha_2 = alpha_f[1] * Vol_f + (1 + v_m[1]) * alpha_m[1] * Vol_m
        else:
            _alpha_2 = (
                (1 + v_f[1]) * alpha_f[1] * Vol_f
                + (1 + v_m[1]) * alpha_m[1] * Vol_m
                - _alpha_1 * _v[2]
            )

        _alpha_3 = _alpha_2

        print(_alpha_1)
        return np.array([_alpha_1, _alpha_2, _alpha_3])

    def _create_composite(
        self, mat_fiber, mat_matrix, Vol_f, array_geometry=1
    ) -> Material:
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
            mat_fiber, mat_matrix, Vol_f, array_geometry=xi
        )

        # Calculate the composite shear modulus
        _G = self._composite_shear_mod(mat_fiber, mat_matrix, Vol_f, array_geometry=xi)

        # Calculate the composite Poisson's ratio
        _v = self._composite_poisson_ratio(_E, _G, mat_fiber, mat_matrix, Vol_f)

        # Calculate the composite thermal expansion ratio
        _alpha = self._composite_thermal_expansion(mat_fiber, mat_matrix, Vol_f, _E, _v)

        # Return the created composite
        return Material(_E, _v, _G, _alpha)
