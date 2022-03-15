import numpy as np
from dataclasses import dataclass, fields
from abc import ABC

class Properties(ABC):
    ...

@dataclass
class MaterialProperties(Properties):
    E: np.ndarray = None
    v: np.ndarray = None
    G: np.ndarray = None
    alpha: np.ndarray = None
    beta: np.ndarray = None
    name: str = ''

    
@dataclass
class LaminaProperties(Properties):
    material: MaterialProperties = None
    material_fiber: MaterialProperties = None
    material_matrix: MaterialProperties = None
    Vol_f: float = 0.0
    Vol_m: float = 1.0
    thickness: float = 0.0
    orientation: float = 0.0
    

@dataclass
class StateProperties(Properties):
    stress: np.ndarray
    strain: np.ndarray
    

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
    
    
    def __init__(self, props: MaterialProperties):
        
        self.props = props
        
        S = self.S = self.compliance_matrix(props, theta_rad=0)
        S_reduced = self.S_reduced = self._reduced_compliance_matrix()
        S_bar = self.compliance_matrix(props, theta_rad=0)
        S_bar_reduced = self._transformed_compliance_matrix_2D()
        
        C = np.linalg.inv(S)
        C_reduced = np.linalg.inv(S_reduced)
        Q_bar = np.linalg.inv(S_bar)
        Q_bar_reduced = np.linalg.inv(S_bar_reduced)
        
        
    def update_orientation(self, theta_rad):
        S_bar = self.compliance_matrix(self.props, theta_rad)
        S_bar_reduced = self._transformed_compliance_matrix_2D(theta_rad)
        Q_bar = np.linalg.inv(S_bar)
        Q_bar_reduced = np.linalg.inv(S_bar_reduced)
    
    
    def compliance_matrix(self, props: MaterialProperties, theta_rad: float=0) -> np.ndarray:
        '''
        Returns the orthotropic compliance matrix.
        
        Parameters:
            E (np.ndarray): Vector of the elastic moduli for each of the principal directions [E1, E2, E3]
            v (np.ndarray): Vector of Poisson's ratio for each of the principal directions [v23, v13, v12]
            G (np.ndarray): Vector of the shear moduli for each fo the principal directions [G23, G13, G12]
            
        Returns:
            S (np.ndarray): Compliance matrix describing the material in the 3 principal directions
        '''
            
        E, v, G = props.E, props.v, props.G
            
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
        
        _S_r = np.zeros((3,3))
        _S_r[:2, :2] = S[:2, :2]
        _S_r[2,2] = S[-1, -1]
        
        return _S_r
    
    
    def transformation_matrix_2D(self, theta_rad: float=0) -> np.ndarray:
        
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        T = np.array([[c**2, s**2, 2*c*s], 
                    [s**2, c**2, -2*c*s], 
                    [-c*s, c*s, c**2-s**2]])
        
        return T
    
    
    def transformation_matrix_3D(self, theta_rad: float=0) -> np.ndarray:
        
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        T = np.array([[c**2, s**2, 0, 0, 0, 2*c*s], 
                    [s**2, c**2, 0, 0, 0, -2*c*s],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0], 
                    [-c*s, c*s, 0, 0, 0, c**2-s**2]])
        
        return T
    
    
    def _transformed_compliance_matrix_2D(self, theta_rad: float=0) -> np.ndarray:
        
        
        T = self.transformation_matrix_2D(theta_rad)
        
        S = self.S_reduced
        S_bar_reduced = T.T.dot(S).dot(T) 
        
        return S_bar_reduced
    
    
def type_check(properties: Properties) -> Properties:
        '''
        Create vectors for variables that are passed in as single values
        '''
        p = properties
        
        for field in fields(properties):
            arg = getattr(p, field.name)
            
            if isinstance(arg, (float, int)):
                setattr(p, field.name, np.ones(3)*arg)
                
            elif isinstance(arg, str):
                pass
            
            elif arg is None:
                setattr(p, field.name, np.zeros(3))
            
            else:
                pass
        
        return p
