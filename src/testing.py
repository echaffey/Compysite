from Compysite import Material, Lamina, Laminate
import numpy as np
import matplotlib.pyplot as plt
from conversion import create_tensor_3D, tensor_to_vec


def validation_1():
    E_f = np.array([233, 23.1, 23.1])
    v_f = np.array([0.40, 0.20, 0.20])
    G_f = np.array([8.27, 8.96, 8.96])
    alpha_f = np.array([-0.54, 10.10, 10.10])
    mat_f = Material(E_f, v_f, G_f, alpha_f)
    V_f = 0.61

    E_m = 4.62
    v_m = 0.36
    G_m = 0
    alpha_m = 41.4
    mat_m = Material(E_m, v_m, G_m, alpha_m)

    layer_1 = Lamina(mat_fiber=mat_f, mat_matrix=mat_m, Vol_fiber=V_f, array_geometry=2)

    E, v, G = layer_1.get_lamina_properties()
    a, b = layer_1.get_lamina_expansion_properties()

    print(E)
    print(v)
    print(G)
    print(a)


def validation_2():
    E = np.array([19.2, 1.56, 1.56])
    v = np.array([0.59, 0.24, 0.24])
    G = np.array([0.49, 0.82, 0.82])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)

    print(layer_1.matrices.S)
    print(layer_1.matrices.C)


def validation_3():
    E = np.array([163, 14.1, 14.1]) * 1e9
    v = np.array([0.45, 0.24, 0.24])
    G = np.array([3.6, 4.8, 4.8]) * 1e9
    alpha = np.array([-0.018, 24.3, 24.3, 0, 0, 0]) * 1e-6
    beta = np.array([150, 4870, 4870, 0, 0, 0]) * 1e-6

    sigma = create_tensor_3D(50, -50, -5, 0, 0, -3) * 1e6

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1)

    layer_1.apply_stress(sigma)
    e_thermal = alpha * 10
    e_moisture = beta * 0.6

    print((layer_1.local_state.strain + e_thermal + e_moisture))


def validation_4():
    E = np.array([163, 14.1, 14.1]) * 1e9
    v = np.array([0.45, 0.24, 0.24])
    G = np.array([3.6, 4.8, 4.8]) * 1e9
    alpha = np.array([-0.018, 24.3, 24.3, 0, 0, 0]) * 1e-6
    beta = np.array([150, 4870, 4870, 0, 0, 0]) * 1e-6

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1)

    epsilon = np.array([4.0e-4, -3.5e-3, 1.2e-3, 0, 0, -6e-4])
    e_thermal = alpha * -30
    e_moisture = beta * 0.6
    e_total = create_tensor_3D(*(epsilon - e_thermal - e_moisture))

    layer_1.apply_strain(e_total)
    print(layer_1.local_state.stress * 1e-6)


def validation_5():
    E = np.array([163, 14.1, 14.1]) * 1e9
    v = np.array([0.45, 0.24, 0.24])
    G = np.array([3.6, 4.8, 4.8]) * 1e9
    alpha = np.array([-0.018, 24.3, 24.3, 0, 0, 0]) * 1e-6
    beta = np.array([150, 4870, 4870, 0, 0, 0]) * 1e-6

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1)

    epsilon = create_tensor_3D(4.0e-4, -3.5e-3, 1.2e-3, 0, 0, -6e-4)

    layer_1.apply_strain(epsilon)

    print(layer_1.matrices.C_reduced.dot(np.array([4.0e-4, -3.5e-3, -6e-4])) * 1e-6)
    print(layer_1.local_state.stress * 1e-6)


def validation_6():
    E = np.array([100, 20, 20])
    v = np.array([0.40, 0.18, 0.18])
    G = np.array([4, 5, 5])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)

    lam.add_lamina(layer_1, 0)
    lam.add_lamina(layer_1, 90)
    lam.add_lamina(layer_1, 90)
    lam.add_lamina(layer_1, 0)

    sigma = create_tensor_3D(30, 15, 5)

    lam.apply_stress(sigma)

    z = np.linspace(0, lam.props.num_layers * 100, 400)
    s = np.ones(100)

    fig = plt.figure()

    stress = []

    for lamina in lam.lamina:
        stress = np.append(stress, s * lamina.local_state.strain[0])

    plt.xlim(0, max(stress) * 1.15)
    plt.plot(stress, z)

    plt.show()
    plt.close(fig)


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


if __name__ == '__main__':
    main()
    # validation_1()
    # validation_2()
    # validation_3()
    # validation_4()
    # validation_5()
    # validation_6()
