
import numpy as np


'''Rotation matrix about the x-axis'''
def R_x(theta_rad): return np.array([[1, 0, 0],
                                [0, np.cos(theta_rad), -np.sin(theta_rad)],
                                [0, np.sin(theta_rad),  np.cos(theta_rad)]])


'''Rotation matrix about the y-axis'''
def R_y(theta_rad): return np.array([[np.cos(theta_rad),  0,  np.sin(theta_rad)],
                                [0, 1, 0],
                                [-np.sin(theta_rad), 0,  np.cos(theta_rad)]])


'''Rotation matrix about the z-axis'''
def R_z(theta_rad): return np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                                [np.sin(theta_rad),   np.cos(theta_rad), 0],
                                [0, 0, 1]])


'''Transformation matrix about the x-axis'''
def T_x(theta_rad): return np.array([[1, 0, 0],
                                        [0, np.cos(theta_rad), np.cos(np.pi/2-theta_rad)],
                                        [0, np.cos(theta_rad+np.pi/2),  np.cos(theta_rad)]])


'''Transformation matrix about the y-axis'''
def T_y(theta_rad): return np.array([[np.cos(theta_rad),  0,  np.cos(theta_rad+np.pi/2)],
                                        [0, 1, 0],
                                        [np.cos(np.pi/2-theta_rad), 0,  np.cos(theta_rad)]])


'''Transformation matrix about the z-axis'''
def T_z(theta_rad): return np.array([[np.cos(theta_rad), np.cos(np.pi/2-theta_rad), 0],
                                        [np.cos(theta_rad+np.pi/2), np.cos(theta_rad), 0],
                                        [0, 0, 1]])


"""Create a 3D tensor given the x, y, z, xy, xz, yz values"""
def create_tensor_3D(_11, _22, _33, _23=0, _13=0, _12=0): return np.array([[_11, _12, _13],[_12, _22, _23],[_13, _23, _33]])


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

def compliance_matrix_3D(E, v, G):
    '''
    Returns the orthotropic compliance matrix.
    
    Parameters:
        E (np.ndarray): Vector of the elastic moduli for each of the principal directions [E1, E2, E3]
        v (np.ndarray): Vector of Poisson's ratio for each of the principal directions [v23, v13, v12]
        G (np.ndarray): Vector of the shear moduli for each fo the principal directions [G23, G13, G12]
        
    Returns:
        S (np.ndarray): Compliance matrix describing the material in the 3 principal directions
    '''
    
    # Check to see if general or directional values are given and convert to vector if needed        
    # E, v, G = type_check(E, v, G)
        
    # Unpack the Poisson's ratio values
    _v23, _v13, _v12 = v
    
    # Create the 3x3 linear-elastic stress relationship
    _norm = np.ones((3,3))*(1/E)
    _n  = np.eye(3)
    
    # Relationships between elastic modulii and Poison's ratio
    _n[0,1] = -E[1]/E[0]*_v12
    _n[1,0] = -_v12
    _n[0,2] = -E[2]/E[0]*_v13
    _n[2,0] = -_v13
    _n[1,2] = -E[1]/E[2]*_v23
    _n[2,1] = -_v23

    # Create the 3x3 shear relationship
    _shear = np.eye(3) / G

    # Combine all into compliance matrix
    _S = np.zeros((6,6))
    _S[:3,:3] = _n * _norm
    _S[3:,3:] = _shear
    
    return _S

def stress2strain(stress_tensor, elasticity_mod, poissons_ratio, shear_mod) -> np.ndarray:
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
    
    # Unpack tensor into a 6x1 column vector
    _vec = np.array([*np.diag(_stress_tensor), _stress_tensor[1,2], _stress_tensor[0,2], _stress_tensor[0,1]])

    # Create compliance matrix
    _S = compliance_matrix_3D(elasticity_mod, poissons_ratio, shear_mod)

    _strain_vec = _S.dot(_vec)

    return _strain_vec