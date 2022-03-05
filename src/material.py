import numpy as np

class Material:
    
    def __init__(self, E=None, v=None, G=None, alpha=None, beta=None, name=''):
        
        E, v, G = self._type_check(E, v, G)
        
        if np.sum(G) == 0:
            G = E/(2*(1+v))
        
        self._E = E
        self._v = v
        self._G = G
        self._alpha = alpha
        self._beta = beta
        self._name = name
        
    def __str__(self):
        desc = f'''
        - Material Properties - 
            Name: {self._name}
            
            E1:   {(self._E[0]*1e-9).round(3) if self._E is not None else '-'} GPa
            E2:   {(self._E[1]*1e-9).round(3) if self._E is not None else '-'} GPa
            E3:   {(self._E[2]*1e-9).round(3) if self._E is not None else '-'} GPa
            
            v23:  {self._v[0] if self._v is not None else '-'}
            v13:  {self._v[1] if self._v is not None else '-'}
            v12:  {self._v[2] if self._v is not None else '-'}
            
            G23:  {(self._G[0]*1e-9).round(3) if self._G is not None else '-'} GPa
            G13:  {(self._G[1]*1e-9).round(3) if self._G is not None else '-'} GPa
            G12:  {(self._G[2]*1e-9).round(3) if self._G is not None else '-'} GPa
            
            a1:   {(self._alpha[0]*1e6).round(3) if self._alpha is not None else '-'} * 1e-6 1/C
            a2:   {(self._alpha[1]*1e6).round(3) if self._alpha is not None else '-'} * 1e-6 1/C
            a3:   {(self._alpha[2]*1e6).round(3) if self._alpha is not None else '-'} * 1e-6 1/C
            
            b1:   {(self._beta[0]*1e6).round(3) if self._beta is not None else '-'} * 1e-6 
            b2:   {(self._beta[1]*1e6).round(3) if self._beta is not None else '-'} * 1e-6
            b3:   {(self._beta[2]*1e6).round(3) if self._beta is not None else '-'} * 1e-6
        
        '''
        return desc
        
    def get_properties(self):
        
        return self._E, self._v, self._G
    
    
    def set_elastic_modulus(self, new_E):
        
        self._E = self._type_check(new_E)
    
        
    def set_poisson_ratio(self, new_v):
        
        self._v = self._type_check(new_v)
    
        
    def set_shear_modulus(self, new_G):
        
        self._G = self._type_check(new_G)
        
    
    def _type_check(self, *args):
        '''
        Create vectors for variables that are passed in as single values
        '''
        _out = None
        
        for arg in args:
            if isinstance(arg, (float, int)):
                if _out is None:
                    _out = np.ones(3)*arg
                else:
                    _out = np.vstack([_out, np.ones(3)*arg])
            elif arg is None:
                if _out is None:
                    _out = np.zeros(3)
                else:
                    _out = np.vstack([_out, np.zeros(3)])
            else:
                if _out is None:
                    _out = np.array(arg)
                else:
                    _out = np.vstack([_out, arg])
        
        return _out
    
    def poisson_tensor(self):
        
        v23, v13, v12 = self._v
        E1, E2, E3 = self._E
        
        v21 = v12*E2/E1
        v31 = v13*E3/E1
        v32 = v23*E3/E2
        
        return np.array([[0, v21, v31],
                         [v12, 0, v32],
                         [v13, v23, 0]])