import numpy as np
from dataclasses import dataclass, fields

from material import Material


@dataclass
class ConversionMatrices:
    S: np.ndarray
    S_reduced: np.ndarray
    S_bar: np.ndarray
    S_bar_reduced: np.ndarray

    C: np.ndarray
    C_reduced: np.ndarray
    Q_bar: np.ndarray
    Q_bar_reduced: np.ndarray

    def __init__(self, mat: Material):

        self.mat = mat

        self.S = self.compliance_matrix(mat, theta_rad=0)
        self.S_reduced = self._reduced_compliance_matrix()
        self.S_bar = self.compliance_matrix(mat, theta_rad=0)
        self.S_bar_reduced = self._transformed_compliance_matrix_2D()

        self.C = np.linalg.inv(self.S)
        self.C_reduced = np.linalg.inv(self.S_reduced)
        self.Q_bar = np.linalg.inv(self.S_bar)
        self.Q_bar_reduced = np.linalg.inv(self.S_bar_reduced)

        self.T_2D = self.transformation_matrix_2D(theta_rad=0)
        self.T_3D = self.transformation_matrix_3D(theta_rad=0)

    def update_orientation(self, theta_rad: float):
        '''
        Updates the conversion matrices which rely on the orientation of the lamina.

        Args:
            theta_rad (float): Orientation in radians.
        '''
        self.S_bar = self.compliance_matrix(self.mat, theta_rad)
        self.S_bar_reduced = self._transformed_compliance_matrix_2D(theta_rad)
        self.Q_bar = np.linalg.inv(self.S_bar)
        self.Q_bar_reduced = np.linalg.inv(self.S_bar_reduced)
        self.T_2D = self.transformation_matrix_2D(theta_rad)
        self.T_3D = self.transformation_matrix_3D(theta_rad)

    def compliance_matrix(self, mat: Material, theta_rad: float = 0) -> np.ndarray:
        '''
        Returns the orthotropic compliance matrix.

        Parameters:
            E (np.ndarray): Vector of the elastic moduli for each of the principal directions [E1, E2, E3]
            v (np.ndarray): Vector of Poisson's ratio for each of the principal directions [v23, v13, v12]
            G (np.ndarray): Vector of the shear moduli for each fo the principal directions [G23, G13, G12]

        Returns:
            S (np.ndarray): Compliance matrix describing the material in the 3 principal directions
        '''

        E, v, G = mat.get_properties()

        # Unpack the Poisson's ratio values
        _v23, _v13, _v12 = v

        # Create the 3x3 linear-elastic stress relationship
        _norm = np.ones((3, 3)) * (1 / E)
        _n = np.eye(3)

        # Relationships between elastic modulii and Poison's ratio
        _n[0, 1] = -E[1] / E[0] * _v12
        _n[1, 0] = -_v12
        _n[0, 2] = -E[2] / E[0] * _v13
        _n[2, 0] = -_v13
        _n[1, 2] = -E[1] / E[2] * _v23
        _n[2, 1] = -_v23

        # Create the 3x3 shear relationship
        _shear = np.eye(3) / G

        # Combine all into compliance matrix
        _S = np.zeros((6, 6))
        _S[:3, :3] = _n * _norm
        _S[3:, 3:] = _shear

        # Transformation matrix (defaults to identity if no rotation)
        T = self.transformation_matrix_3D(theta_rad=theta_rad)

        _S = T.T.dot(_S).dot(T)

        return _S

    def _reduced_compliance_matrix(self) -> np.ndarray:
        '''
        Returns the planar compliance matrix.

        Parameters:
            E (np.ndarray): Vector of the elastic moduli for each of the principal directions
            v (np.ndarray): Vector of Poisson's ratio for each of the principal directions [v23, v13, v12]
            G (np.ndarray): Vector of the shear moduli for each fo the principal directions [G23, G13, G12]

        Returns:
            S (np.ndarray): Planar (reduced) compliance matrix 
        '''

        S = self.S

        _S_r = np.zeros((3, 3))
        _S_r[:2, :2] = S[:2, :2]
        _S_r[2, 2] = S[-1, -1]

        return _S_r

    def transformation_matrix_2D(self, theta_rad: float = 0) -> np.ndarray:

        c = np.cos(theta_rad)
        s = np.sin(theta_rad)

        T = np.array(
            [
                [c ** 2, s ** 2, 2 * c * s],
                [s ** 2, c ** 2, -2 * c * s],
                [-c * s, c * s, c ** 2 - s ** 2],
            ]
        )

        return T

    def transformation_matrix_3D(self, theta_rad: float = 0) -> np.ndarray:

        c = np.cos(theta_rad)
        s = np.sin(theta_rad)

        T = np.array(
            [
                [c ** 2, s ** 2, 0, 0, 0, 2 * c * s],
                [s ** 2, c ** 2, 0, 0, 0, -2 * c * s],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [-c * s, c * s, 0, 0, 0, c ** 2 - s ** 2],
            ]
        )

        return T

    def _transformed_compliance_matrix_2D(self, theta_rad: float = 0) -> np.ndarray:
        '''
        Calculates the 2D transformed compliance matrix.

        Args:
            theta_rad (float, optional): Rotation angle measure in radians. Defaults to 0.

        Returns:
            np.ndarray: The transformed 2D compliance matrix evaluated at theta_rad
        '''
        T = self.transformation_matrix_2D(theta_rad)

        S = self.S_reduced
        S_bar_reduced = T.T.dot(S).dot(T)

        return S_bar_reduced


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

