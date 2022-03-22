from Compysite import Material, Lamina, Laminate
import numpy as np
import matplotlib.pyplot as plt
from conversion import create_tensor_3D, tensor_to_vec


def validation_6():
    E = np.array([100, 20, 20])
    v = np.array([0.40, 0.18, 0.18])
    G = np.array([4, 5, 5])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)

    lam.add_lamina(layer_1, 45)
    # lam.add_lamina(layer_1, -30)

    sigma = create_tensor_3D(30, 15, 5)

    lam.apply_stress(sigma)

    print(lam.global_state)


def validation_4():
    E = np.array([163, 14.1, 14.1])
    v = np.array([0.45, 0.24, 0.24])
    G = np.array([3.6, 4.8, 4.8])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1)

    epsilon = create_tensor_3D(4.0e-4, -3.5e-3, 1.2e-3, 0, 0, -6e-4)

    lam.apply_strain(epsilon)
    print(lam.lamina[0].local_state.stress * 1e3)


def validation_3():
    E = np.array([163, 14.1, 14.1]) * 1e9
    v = np.array([0.45, 0.24, 0.24])
    G = np.array([3.6, 4.8, 4.8]) * 1e9

    sigma = create_tensor_3D(50, -50, -5, 0, 0, -3) * 1e6

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)

    lam.add_lamina(layer_1)
    lam.apply_stress(sigma)

    print(lam.lamina[0].local_state.strain.round(4))
    # print(lam.global_state[0].strain.round(4))
    # print(lam.lamina[0].props.orientation)


def main():
    E = np.array([14, 3.5, 3.5])
    v = np.array([0.5, 0.4, 0.4])
    G = np.array([3, 4.2, 4.2])

    lam = Laminate()

    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1, 45)

    sigma = create_tensor_3D(30, 15, 5)

    # lam.apply_stress(sigma)
    lam.apply_stress(sigma)
    lam.apply_stress(sigma)
    print(lam.global_state)

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

    # T = layer_1.matrices.transformation_matrix_3D(60 * np.pi / 180)

    # T_inv = np.linalg.inv(T)

    # alpha = np.array([8.6, 22.1, 0, 0, 0, 0]) * 1e-6

    # print(T_inv.dot(alpha))


if __name__ == '__main__':
    # main()
    validation_3()
    # validation_4()
    # validation_6()
