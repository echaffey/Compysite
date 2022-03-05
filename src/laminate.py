
import numpy as np
from lamina import Lamina
from material import Material
from utils import create_tensor_3D, transformation_3D, T_z, stress2strain


class Laminate:
    
    
    def __init__(self, length=0, width=0):
        
        self._num_plys = 0
        self._list_plys = []
        self._thickness = 0
        self._length = 0
        self._width = 0
    
    
    def add_lamina(self, new_lamina:Lamina, orientation):
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 
        
            Parameters:
                new_lamina (Lamina):         the constructed lamina object to be added to the laminate.
                orientation (numpy.ndarray): orientation vector of the fiber directions relative to the 
                                             principal axes in the z, y, x directions. 
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
    
    lam = Laminate()
    
    sigma = create_tensor_3D(100, 10, 0, 0, 0, -5)
    s_prime = transformation_3D(sigma, T_z, 55)
    
    # print(s_prime)
    
    V_f = 0.61
    xi = 1
    
    E_f = np.array([233, 23.1, 23.1])*1e9
    v_f = np.array([0.4, 0.2, 0.2])
    G_f = np.array([8.27, 8.96, 8.96])*1e9
    alpha_f = np.array([-0.54, 10.10, 10.10])*1e-6
    
    E_m = np.ones(3)*4.62e9
    v_m = np.ones(3)*0.360
    alpha_m = np.ones(3)*41.4e-6
    
    fiber = Material(E_f, v_f, G_f, alpha_f, name='fiber')
    matrix = Material(E_m, v_m, alpha=alpha_m, name='matrix')
    
    layer_1 = Lamina(fiber, matrix, Vol_fiber=V_f, array_geometry=xi)
    
    E, v, G = layer_1.get_lamina_properties()
    
    E = np.array([14, 3.5, 3.5])*1e9
    v = np.array([0, 0.4, 0.4])
    G = np.array([0, 4.2, 4.2])*1e9
    
    comp = Material(E, v, G, name='comp')
    layer_2 = Lamina(mat_composite=comp, Vol_fiber=1)
    
    theta = 60
    c = np.cos(theta*np.pi/180)
    s = np.sin(theta*np.pi/180)
    
    T = np.array([[c**2, s**2, 2*c*s], [s**2, c**2, -2*c*s], [-c*s, c*s, c**2-s**2]])
    sig = np.array([-3.5, 7, -1.4])*1e6
    sig_1 = T.dot(sig)
    S = layer_2.S_red
    # print(layer_2.S_red.dot(sig_1)*1e6)
    
    print(S.dot(T).dot(sig))
    print(np.linalg.inv(T).dot(S.dot(T).dot(sig)))