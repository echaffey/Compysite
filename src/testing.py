from venv import create
from cv2 import add
from Compysite import Material, Lamina, Laminate

import numpy as np
import matplotlib.pyplot as plt

from utils import create_tensor_3D, transformation_3D, T_z, to_epsilon, \
                  to_gamma, tensor_to_vec, principal_stress_3D, \
                  principal_angle_2D

if __name__ == '__main__':
   
    E = np.array([14, 3.5, 3.5])
    v = np.array([0.5, 0.4, 0.4])
    G = np.array([3, 4.2, 4.2])
    
    # Vol_f = 0.61
    # E_f = np.array([233, 23.1, 23.1])*1e9
    # v_f = np.array([0.4, 0.2, 0.2])
    # G_f = np.array([8.27, 8.96, 8.96])*1e9
    # alpha_f = np.array([-0.54, 10.10, 10.10])*1e-6

    # E_m = np.ones(3)*4.62e9
    # v_m = np.ones(3)*0.360
    # alpha_m = np.ones(3)*41.4e-6
    
    # E = np.array([50, 15.2, 15.2])*1e9
    # v = np.array([0.428, 0.254, 0.254])
    # G = np.array([3.28, 4.7, 4.7])*1e9

    # alpha = np.array([6.34, 23.3, 23.3])*1e-6
    # beta = np.array([434, 6320, 6320, 0, 0, 0])*1e-6

    # l, w, h = np.ones(3)*75e-3

    # dT = 100
    
    # sigma = create_tensor_3D(0, 0, 0)
    # e_thermal = alpha*dT
    
    lam = Laminate()
    # mat_f = Material(E_f, v_f, G_f, alpha_f)
    # mat_m = Material(E_m, v_m, alpha=alpha_m)
    # layer_1 = Lamina(mat_fiber=mat_f, mat_matrix=mat_m, Vol_fiber=Vol_f)
    
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1, 0)
    
    # lam.add_lamina(layer_1, 60)
    # print(layer_1.apply_2D_boundary_conditions(sigma, 1, additional_strain=[e_thermal]))
    
    # mat_1 = Material(E, v, G, name='test')
    # layer_1 = Lamina(mat_composite=mat_1)
    # lam.add_lamina(layer_1, 60)
    # lam.add_lamina(layer_1, 30)
    
    # Create global applied stress
    # sigma_xyz = create_tensor_3D(-3.5, 7, 0, 0, 0, -1.4)
    
    # # Transform to local stress
    # sigma_123 = transformation_3D(sigma_xyz, T_z, 60)
 
    # # Convert local stress to local strain
    # e_123 = layer_1.stress2strain(sigma_123)

    # # Convert to epsilon tensor (MUST DO)
    # e_123 = to_epsilon(create_tensor_3D(*e_123))
    # # e_123 = create_tensor_3D(*e_123)

    # # Transform local to global strain and convert back to gamma (MUST DO)
    # e_xyz = to_gamma(transformation_3D(e_123, T_z, -60))

    # # CORRECT ACCORDING TO NOTES
    # print(tensor_to_vec(e_xyz))

    # abc = lam.stress2strain(sigma_xyz)

    # d = lam.strain2stress(create_tensor_3D(*abc))

    
    # 1. assign material properties
    # 2. Create material
    # 3. Create lamina layer
    # 4. Create global stresses
    # 5. Solve for global strains
    # 6. Convert global to local
    # 7. 
    
    # theta_range = np.linspace(-np.pi/2, np.pi/2, 100)
    # layer_1.plot_compliance(theta_range)
    
    # sigma = np.array([2, -3, 0, 0, 0, 4])
    # s = create_tensor_3D(*sigma)
    
    # (s1, s2, s3), _ = principal_stress_3D(s)
    
    # print(s1, s2, s3)
    
    # print(principal_angle_2D(s))
    
    T = layer_1.transformation_matrix_3D(60*np.pi/180)
    
    T_inv = np.linalg.inv(T)
    
    alpha = np.array([8.6, 22.1, 0, 0, 0, 0])*1e-6
    
    print(T_inv.dot(alpha))

