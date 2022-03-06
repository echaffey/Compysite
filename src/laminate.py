from lamina import Lamina
from material import Material
from utils import create_tensor_3D, T_z, to_epsilon, to_gamma

import numpy as np
from typing import Union



class Laminate:
    
    
    def __init__(self, length=0, width=0):
        
        self._num_plys = 0
        self._layers = {'lamina':[], 
                        'orientation':[],
                        'material':[None],
                        'stress':[None],
                        'strain':[None],
                        'global_stress':[None],
                        'global_strain':[None]}
        self._thickness = 0
        self._length = length
        self._width = width
    
    
    def add_lamina(self, new_lamina: Lamina, orientation: float):
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 
        
            Parameters:
                new_lamina (Lamina):         the constructed lamina object to be added to the laminate.
                orientation (numpy.ndarray): orientation vector of the fiber directions relative to the 
                                             principal axes in the z, y, x directions. 
        '''
        
        self._layers['lamina'].append(new_lamina)
        self._layers['orientation'].append(orientation)
        
        self._thickness += new_lamina.thickness
        self._num_plys += 1


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
        _theta   = theta if theta_radians else theta * np.pi/180
        _R       = rot_matrix(_theta)
        
        # Transformation equation
        _prime = _R.dot(_tensor).dot(_R.T)
        
        return _prime
    
    
    def principal_stress_3D(stress_tensor) -> Union[np.ndarray, np.ndarray]:
        '''
        Returns the three principal stresses of a given tensor and their corresponding direction vectors.
        
            Parameters:
                stress_tensor (numpy.ndarray):  Stress tensor
            Returns:
                p_val (numpy.ndarray):  Array of ordered principal stress in descending value 
                p_vec (numpy.ndarray):  Array of corresponding direction vectors
        '''
        _stress_tensor = stress_tensor.copy()
        
        # Principal stresses and thier corresponding vectors
        _e_val, _e_vec = np.linalg.eig(_stress_tensor)
        
        # Sort the principal stresses in ascending order
        _p3, _p2, _p1 = np.sort(_e_val)
        
        # Correlate the sorted stresses with their vectors
        _e_val_l = _e_val.tolist()
        _p1_index, _p2_index, _p3_index = _e_val_l.index(_p1), _e_val_l.index(_p2), _e_val_l.index(_p3)
        _p1_vec, _p2_vec, _p3_vec = _e_vec[:,_p1_index], _e_vec[:,_p2_index], _e_vec[:,_p3_index]
        
        # Assembble two vectors containing the principal stresses and their dirctional vectors
        _p_val, _p_vec = np.array([_p1,_p2,_p3]), np.array([_p1_vec, _p2_vec, _p3_vec])
        
        return _p_val, _p_vec


    def stress2strain(stress_tensor, lamina: Lamina, theta_deg: float=0) -> np.ndarray:
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
        
        _stress_tensor = stress_tensor.copy()
        
        # Convert to radians
        theta_rad = theta_deg * np.pi/180
        
        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(_stress_tensor), _stress_tensor[1,2], _stress_tensor[0,2], _stress_tensor[0,1]])

        # Create compliance matrix
        _S = lamina.compliance_matrix(theta_rad=theta_rad)

        _strain_vec = _S.dot(_vec)

        return _strain_vec
    
    
    def stress2strain_global_temp(self, stress_tensor) -> np.ndarray:
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
            
            # Transform global to local lamina stress
            local_lamina_stress = self.transformation_3D(stress_tensor, T_z, theta=theta_deg)
            
            # Store the layer's stress state
            if i > len(self._layers['stress']):
                self._layers['stress'].append(local_lamina_stress)
            else:
                self._layers['stress'][i] = local_lamina_stress
            
            # Convert local stress to local strain and store the value
            local_lamina_strain = lamina.stress2strain(local_lamina_stress)
            if i > len(self._layers['strain']):
                self._layers['strain'].append(local_lamina_strain)
            else:
                self._layers['strain'][i] = local_lamina_strain
            
            # Epsilon tensor of local lamina strain
            e_local = to_epsilon(create_tensor_3D(*local_lamina_strain))
            
            # Gamma values of global lamina strain
            e_global = to_gamma(self.transformation_3D(e_local, T_z, theta=-theta_deg))
            if i > len(self._layers['global_strain']):
                self._layers['global_strain'].append(e_global)
            else:
                self._layers['global_strain'][i] = e_global
    
    
    def stress2strain_global(stress_tensor, lamina: Lamina) -> np.ndarray:
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
        
        _stress_tensor = stress_tensor.copy()
        
        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(_stress_tensor), _stress_tensor[1,2], _stress_tensor[0,2], _stress_tensor[0,1]])

        # Transformed compliance matrix
        _S = lamina.S_bar

        _strain_vec = _S.dot(_vec)

        return _strain_vec


    def strain2stress(strain_tensor, lamina: Lamina) -> np.ndarray:
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
        _strain_tensor = strain_tensor.copy()
        
        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(_strain_tensor), _strain_tensor[1,2], _strain_tensor[0,2], _strain_tensor[0,1]])
        
        # Create stiffness matrix
        _C = lamina.C
        
        _stress_vec = _C.dot(_vec)
        
        return _stress_vec
    
    
    def strain2stress_global(strain_tensor, lamina: Lamina) -> np.ndarray:
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
        _strain_tensor = strain_tensor.copy()
        
        # Unpack tensor into a 6x1 column vector
        _vec = np.array([*np.diag(_strain_tensor), _strain_tensor[1,2], _strain_tensor[0,2], _strain_tensor[0,1]])
        
        # Create stiffness matrix
        _C = lamina.Q_bar
        
        _stress_vec = _C.dot(_vec)
        
        return _stress_vec
    
    
    def apply_stress(self, tensor_applied):
        
        sigma = 0
        epsilon = 0
        return sigma, epsilon
    
    def global_compliance_matrix(self):
        # S_bar
        pass
    
    def global_stiffness_matrix(self):
        # Q_bar
        # inverse compliance
        pass
    