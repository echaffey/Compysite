
import numpy as np

from material import Material

class Lamina:
    
    
    def __init__(self, mat_fiber=None, mat_matrix=None, mat_composite=None, Vol_fiber=0, Vol_matrix=0, orientation=None, thickness=0, array_geometry=1):
        '''Create a single lamina using known fiber and matrix materials or assigned with a predetermined composite material.
        
        Parameters:
            mat_fiber (Material):  [Optional] Fiber material object.
            mat_matrix (Material):  [Optional] Matrix material object. 
            mat_composite (Material):  [Optional] Composite material object.
            Vol_fiber (float):  [Optional] Fiber volume fraction.
            Vol_matrix (float):  [Optional] Matrix volume fraction.
            orientation (numpy.ndarray):  [Optional] Orientation vector in the z, y, x directions. 
            array_geometry (int):  [Optional] Matrix array geometry constant.  1 = Hexagonal array, 2 = Square array.
        '''
        
        self._material = None
        self._material_fiber = mat_fiber
        self._material_matrix = mat_matrix
        
        self._Vol_f = Vol_fiber
        self._Vol_m = Vol_matrix
        self.thickness = thickness
        
        if orientation is not None: 
            self._orientation = orientation
        else:
            self._orientation = np.zeros(3)
        
        # Create the composite from the fiber and matrix materials if a composite is not given
        # Alternatively, if only a matrix is given, its a uniform material
        if mat_composite is None:
            if (mat_fiber is not None) & (mat_matrix is not None):
                
                # -------------------
                # Need to add alpha and beta calculations
                # -------------------
                
                # Create composite from the fiber and matrix materials
                E_c, v_c, G_c = self._create_composite(mat_fiber, mat_matrix, array_geometry)
                self._material = Material(E_c, v_c, G_c)
                
            elif mat_matrix is not None:
                self._material = mat_matrix
            else:
                self._material = mat_composite
                print('You must create at least one material that is not a fiber.')
        
       
    def halpin_tsai(self, M_f, M_m, V_f, array_geometry=1):
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
    
    
    def composite_shear_mod(self, mat_fiber, mat_matrix, array_geometry=1):
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
        G_12 = self.halpin_tsai(G_f, G_m, Vol_f, xi)
        
        # Set the poisson's ratio and shear modulus in the 23 direction
        v_m_23 = v_m[0]
        G_m_23 = G_m[0]
        G_f_23 = G_f[0]
        
        # Calculate the out of plane shear
        n_23 = (3 - 4*v_m_23 + G_m_23/G_f_23)/(4*(1 - v_m_23))
        _G_23 = G_m_23 * (Vol_f + n_23*Vol_m)/(n_23*Vol_m + Vol_f*(G_m_23/G_f_23))
        
        # Composite is assume to be orthotropic
        _G_13 = G_12
        
        # Return an array containing the composite shear modulus
        return np.array([_G_23, _G_13, G_12])
    
    
    def composite_poisson_ratio(self, E_c, G_c, mat_fiber, mat_matrix):
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
    
        
    def composite_elastic_mod(self, mat_fiber, mat_matrix, array_geometry=1):
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
        
        # if self._Vol_f + self._Vol_m != 1:
        #     print('Sum of the volume fractions of the materials does not equal 1.')
        #     exit
            
        # elif self._Vol_f > 0 & self._Vol_m == 0:
        #     Vol_f = self._Vol_f
        #     self._Vol_m = Vol_m = 1 - Vol_f
            
        # elif self._Vol_m > 0 & self._Vol_f == 0:
        #     Vol_m = self._Vol_m
        #     self._Vol_f = Vol_f = 1 - Vol_f
        
        Vol_f = self._Vol_f
        Vol_m = 1 - Vol_f
        
        # Rule of mixtures
        _E_1 = E_f[0] * Vol_f + E_m[0] * Vol_m
        
        # Halpin-Tsai for transverse directions
        _E_2 = self.halpin_tsai(E_f, E_m, Vol_f, array_geometry)
        
        # Material assumed to be orthotropic
        _E_3 = _E_2
        
        return np.array([_E_1, _E_2, _E_3])
    
    
    def _create_composite(self, mat_fiber, mat_matrix, array_geometry=1):
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
        
        E_f, v_f, G_f = mat_fiber.get_properties()
        E_m, v_m, _ = mat_matrix.get_properties()
        
        Vol_f = self._Vol_f
        
        # -----------------------------------------------------
        # CHANGE THIS BACK TO ACCEPT THE PROPERTIES AS ARGUMENTS
        # -----------------------------------------------------
        
        # Calculate the composite elastic modulus
        _E = self.composite_elastic_mod(mat_fiber, mat_matrix, array_geometry=xi)
        
        # Calculate the composite shear modulus
        _G = self.composite_shear_mod(mat_fiber, mat_matrix, array_geometry=xi)

        # Calculate the composite Poisson's ratio
        _v = self.composite_poisson_ratio(_E, _G, mat_fiber, mat_matrix)
        
        return np.array([_E, _v, _G])
    
    
    def get_lamina_properties(self):
        
        E, v, G = self._material.get_properties()
        return E, v, G