from Compysite import Material, Lamina, Laminate

import numpy as np
import matplotlib.pyplot as plt

from utils import create_tensor_3D, transformation_3D, T_z, to_epsilon, to_gamma, tensor_to_vec

if __name__ == '__main__':
   
    E = np.array([14, 3.5, 3.5])
    v = np.array([0.5, 0.4, 0.4])
    G = np.array([3, 4.2, 4.2])
    
    lam = Laminate()
    mat_1 = Material(E, v, G, name='test')
    layer_1 = Lamina(mat_composite=mat_1)
    lam.add_lamina(layer_1, 60)
    
    # Create global applied stress
    sigma_xyz = create_tensor_3D(-3.5, 7, 0, 0, 0, -1.4)
    
    # Transform to local stress
    sigma_123 = transformation_3D(sigma_xyz, T_z, 60)
 
    # Convert local stress to local strain
    e_123 = layer_1.stress2strain(sigma_123)

    # Convert to epsilon tensor (MUST DO)
    e_123 = to_epsilon(create_tensor_3D(*e_123))
    # e_123 = create_tensor_3D(*e_123)

    # Transform local to global strain and convert back to gamma (MUST DO)
    e_xyz = to_gamma(transformation_3D(e_123, T_z, -60))

    # CORRECT ACCORDING TO NOTES
    # print(tensor_to_vec(e_xyz))

    abc = lam.stress2strain(sigma_xyz)

    d = lam.strain2stress(create_tensor_3D(*abc))

    
    # 1. assign material properties
    # 2. Create material
    # 3. Create lamina layer
    # 4. Create global stresses
    # 5. Solve for global strains
    # 6. Convert global to local
    # 7. 
    
    theta_range = np.linspace(-np.pi/2, np.pi/2, 100)
    # layer_1.plot_compliance(theta_range)
