
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