import numpy as np
from dataclasses import dataclass
from properties import MaterialProperties, type_check

class Material:
    
    def __init__(self, E=None, v=None, G=None, alpha=None, beta=None, name=''):
        
        props = MaterialProperties(E, v, G, alpha, beta, name)
        props = type_check(props)
        
        if np.sum(props.G) == 0:
            props.G = props.E/(2*(1+props.v))
        
        self.props = props
        
    def __str__(self):
        desc = f'''
        - Material Properties - 
            Name: {self.props.name}
            
            E1:   {(self.props.E[0]*1e-9).round(3) if self.props.E is not None else '-'} GPa
            E2:   {(self.props.E[1]*1e-9).round(3) if self.props.E is not None else '-'} GPa
            E3:   {(self.props.E[2]*1e-9).round(3) if self.props.E is not None else '-'} GPa
            
            v23:  {self.props.v[0] if self.props.v is not None else '-'}
            v13:  {self.props.v[1] if self.props.v is not None else '-'}
            v12:  {self.props.v[2] if self.props.v is not None else '-'}
            
            G23:  {(self.props.G[0]*1e-9).round(3) if self.props.G is not None else '-'} GPa
            G13:  {(self.props.G[1]*1e-9).round(3) if self.props.G is not None else '-'} GPa
            G12:  {(self.props.G[2]*1e-9).round(3) if self.props.G is not None else '-'} GPa
            
            a1:   {(self.props.alpha[0]*1e6).round(3) if self.props.alpha is not None else '-'} * 1e-6 1/C
            a2:   {(self.props.alpha[1]*1e6).round(3) if self.props.alpha is not None else '-'} * 1e-6 1/C
            a3:   {(self.props.alpha[2]*1e6).round(3) if self.props.alpha is not None else '-'} * 1e-6 1/C
            
            b1:   {(self.props.beta[0]*1e6).round(3) if self.props.beta is not None else '-'} * 1e-6 
            b2:   {(self.props.beta[1]*1e6).round(3) if self.props.beta is not None else '-'} * 1e-6
            b3:   {(self.props.beta[2]*1e6).round(3) if self.props.beta is not None else '-'} * 1e-6
        
        '''
        return desc
        
    def get_properties(self):
        
        return self.props.E, self.props.v, self.props.G
    
    
    def poisson_tensor(self):
        
        v23, v13, v12 = self._v
        E1, E2, E3 = self._E
        
        v21 = v12*E2/E1
        v31 = v13*E3/E1
        v32 = v23*E3/E2
        
        return np.array([[0, v21, v31],
                         [v12, 0, v32],
                         [v13, v23, 0]])
        
        
class CompositeMaterial:
    pass