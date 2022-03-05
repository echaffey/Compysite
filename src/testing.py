import Compysite as comp

import numpy as np
import matplotlib.pyplot as plt

from utils import create_tensor_3D, transformation_3D, T_z, to_epsilon, to_gamma, tensor_to_vec

if __name__ == '__main__':
    # E = np.array([100, 20, 20])*1e9
    # v = np.array([0.4, 0.18, 0.18])
    # G = np.array([4, 5, 5])*1e9
    
    # mat_1 = comp.Material(E, v, G, name='test')
    # layer_1 = comp.Lamina(mat_composite=mat_1, deg_orientation=60)
    
    # # Applied global stresses
    # sigma_applied = create_tensor_3D(2, -3, 0, 0, 0, 4)

    # # Global principal stress
    # p, _ = comp.Laminate.principal_stress_3D(sigma_applied)
    # sigma_p = create_tensor_3D(*p)
    # s_1, s_2, s_3 = p
    
    # # Global max shear
    # shear_max = (s_1 - s_3)/2
    
    # # Global principal strain
    # e_p = comp.Laminate.stress2strain(sigma_p, layer_1)
    # e_xyz = comp.Laminate.stress2strain(sigma_applied, layer_1)
    
    # T = layer_1.transformation_matrix()
    # S_bar = layer_1.Q_bar
    
    # print(sigma_p)
    # print(e_xyz*1e9)
    
    E = np.array([14, 3.5, 3.5])
    v = np.array([0.5, 0.4, 0.4])
    G = np.array([3, 4.2, 4.2])
    
    mat_1 = comp.Material(E, v, G, name='test')
    layer_1 = comp.Lamina(mat_composite=mat_1, deg_orientation=60)
    
    # Create global applied stress
    sigma_xyz = create_tensor_3D(-3.5, 7, 0, 0, 0, -1.4)
    
    # Transform to local stress
    sigma_123 = transformation_3D(sigma_xyz, T_z, 60)
    
    # Convert local stress to local strain
    e_123 = comp.Laminate.stress2strain(sigma_123, layer_1)
    
    # Convert to epsilon tensor (MUST DO)
    e_123 = to_epsilon(create_tensor_3D(*e_123))
    
    # Transform local to global strain and convert back to gamma (MUST DO)
    e_xyz = to_gamma(transformation_3D(e_123, T_z, -60))
    
    # CORRECT ACCORDING TO NOTES
    print(tensor_to_vec(e_xyz))
    print(comp.Laminate.stress2strain_global(sigma_xyz,layer_1))
    print(comp.Laminate.strain2stress_global(e_xyz, layer_1).round(4))
    
    # 1. assign material properties
    # 2. Create material
    # 3. Create lamina layer
    # 4. Create global stresses
    # 5. Solve for global strains
    # 6. Convert global to local
    # 7. 
