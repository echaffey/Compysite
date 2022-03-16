from typing import Union
import numpy as np


def R_x(theta_rad):
    '''Rotation matrix about the x-axis'''
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0, np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )


def R_y(theta_rad):
    '''Rotation matrix about the y-axis'''
    return np.array(
        [
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)],
        ]
    )


def R_z(theta_rad):
    '''Rotation matrix about the z-axis'''
    return np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1],
        ]
    )


def T_x(theta_rad):
    '''Transformation matrix about the x-axis'''
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_rad), np.cos(np.pi / 2 - theta_rad)],
            [0, np.cos(theta_rad + np.pi / 2), np.cos(theta_rad)],
        ]
    )


def T_y(theta_rad):
    '''Transformation matrix about the y-axis'''
    return np.array(
        [
            [np.cos(theta_rad), 0, np.cos(theta_rad + np.pi / 2)],
            [0, 1, 0],
            [np.cos(np.pi / 2 - theta_rad), 0, np.cos(theta_rad)],
        ]
    )


def T_z(theta_rad):
    '''Transformation matrix about the z-axis'''
    return np.array(
        [
            [np.cos(theta_rad), np.cos(np.pi / 2 - theta_rad), 0],
            [np.cos(theta_rad + np.pi / 2), np.cos(theta_rad), 0],
            [0, 0, 1],
        ]
    )


def create_tensor_3D(_11, _22, _33, _23=0, _13=0, _12=0):
    """Create a 3D tensor given the x, y, z, xy, xz, yz values"""
    return np.array([[_11, _12, _13], [_12, _22, _23], [_13, _23, _33]])


def tensor_to_vec(tensor):
    '''Create a vector from a given 3D tensor'''
    return np.array([*np.diag(tensor), tensor[1, 2], tensor[0, 2], tensor[0, 1]])


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
    _theta = theta if theta_radians else theta * np.pi / 180
    _R = rot_matrix(_theta)

    # Transformation equation
    _prime = _R.dot(_tensor).dot(_R.T)

    return _prime


def principal_angle_2D(tensor: np.ndarray) -> float:
    '''
    Returns the principal angle of the tensor in degrees.

        Parameters:
            tensor (numpy.ndarray):  Cauchy tensor
        Returns:
            theta_p (numpy.ndarray):  Principal angle measure between coordinate axes and principal axes in degrees
    '''
    _tensor = tensor.copy()

    _x = _tensor[0, 0]
    _y = _tensor[1, 1]
    _xy = _tensor[0, 1]

    _theta_p = 0.5 * np.arctan(2 * _xy / (_x - _y))

    return _theta_p * 180 / np.pi


def principal_stress_3D(stress_tensor: np.ndarray) -> Union[np.ndarray, np.ndarray]:
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
    _p1_index, _p2_index, _p3_index = (
        _e_val_l.index(_p1),
        _e_val_l.index(_p2),
        _e_val_l.index(_p3),
    )
    _p1_vec, _p2_vec, _p3_vec = (
        _e_vec[:, _p1_index],
        _e_vec[:, _p2_index],
        _e_vec[:, _p3_index],
    )

    # Assembble two vectors containing the principal stresses and their dirctional vectors
    _p_val, _p_vec = np.array([_p1, _p2, _p3]), np.array([_p1_vec, _p2_vec, _p3_vec])

    return _p_val, _p_vec


def to_gamma(strain_tensor) -> np.ndarray:
    '''
    Converts a given strain tensor into a matrix with shear strain in terms of gamma.

    Parameters:
        strain_tensor (np.ndarray): Strain tensor in terms of epsilon.

    Returns:
        gamma_matrix (np.ndarray):  Strain matrix in terms of gamma.
    '''
    _strain_tensor = strain_tensor.copy()

    _gamma_matrix = _strain_tensor + (_strain_tensor - _strain_tensor * np.eye(3))

    return _gamma_matrix


def to_epsilon(strain_matrix) -> np.ndarray:
    '''
    Converts a given strain matrix into a strain tensor with shear strain in terms of gamma.

    Parameters:
        strain_matrix (np.ndarray): Strain matrix in terms of gamma.

    Returns:
        epsilon_tensor (np.ndarray):  Strain tensor in terms of epsilon.
    '''
    _strain_matrix = strain_matrix.copy()

    _epsilon_tensor = _strain_matrix * np.eye(3) + 0.5 * (
        _strain_matrix - _strain_matrix * np.eye(3)
    )

    return _epsilon_tensor
