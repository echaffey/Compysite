import Compysite as comp

import numpy as np

if __name__ == '__main__':
    E = np.array([100, 20, 20])*1e9
    v = np.array([0.4, 0.18, 0.18])
    G = np.array([4, 5, 5])*1e9
    
    mat_1 = comp.Material(E, v, G, name='test')
    layer_1 = comp.Lamina(mat_composite=mat_1, deg_orientation=-30)

    T = layer_1.transformation_matrix()
    S_bar = layer_1.Q_bar
    
    print(S_bar*1e-9)
