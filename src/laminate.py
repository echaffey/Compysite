
import numpy as np
from lamina import Lamina
from utils import create_tensor_3D, transformation_3D, T_z


class Laminate:
    
    
    def __init__(self, length, width):
        
        self._num_plys = 0
        self._list_plys = []
        self._thickness = 0
        self._length = 0
        self._width = 0
    
    
    def add_lamina(self, new_lamina:Lamina, orientation):
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 
        '''
        
        self._list_plys.append((new_lamina, orientation))
        self._thickness += new_lamina.thickness
        
    
    def transformation_3D(tensor, rot_matrix, theta, theta_radians=False):
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
    
if __name__ == '__main__':
    
    s = create_tensor_3D(400, 100, 0, 0, 0, -200)
    print(transformation_3D(s, T_z, 26.57))
    # print(s)