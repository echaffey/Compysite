from Compysite import Material, Lamina, Laminate
import numpy as np
import matplotlib.pyplot as plt
from conversion import create_tensor_3D
from conversion import to_epsilon
from conversion import tensor_to_vec


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

    print(layer_1.matrices.S_reduced)
    print(layer_1.matrices.C_reduced)


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
    layer_2 = Lamina(mat_composite=mat)

    lam.add_lamina(layer_1, 45)
    lam.add_lamina(layer_2, -30)

    print(lam.get_lamina(1).matrices.T_2D)
    print(lam.get_lamina(2).matrices.T_2D)


def validation_7():
    E = np.array([181, 10.3, 10.3]) * 1e9
    v = np.array([0, 0.28, 0.28])
    G = np.array([1, 7.17, 7.17]) * 1e9

    NM = np.array([1000, 1000, 0, 0, 0, 0])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat, thickness=5e-3)

    lam.add_lamina(layer_1, 0)
    lam.add_lamina(layer_1, 30)
    lam.add_lamina(layer_1, -45)

    lam.apply_load(NM)

    state = lam.get_state_at_height(-2.5e-3, 2)

    print(state.strain)

    # z_height = np.linspace(lam._z[0], lam._z[-1], 100)
    # stresses = np.zeros((3, 100))

    # for i, z in enumerate(z_height):
    #     state = lam.get_state_at_height(z)
    #     stresses[0, i] = state.strain[0]
    #     stresses[1, i] = state.strain[1]
    #     stresses[2, i] = state.strain[2]

    # plt.plot(stresses[0, :], z_height)
    # plt.plot(stresses[1, :], z_height)
    # plt.plot(stresses[2, :], z_height)

    # for i, _ in enumerate(lam._z):
    #     plt.hlines(lam._z[i], -4e-6, 6e-6, linewidth=1)
    # plt.show()


def validation_8():
    E = np.array([138, 9, 9])
    v = np.array([0, 0.3, 0.3])
    G = np.array([1, 6.9, 6.9])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat, thickness=0.25)

    lam.add_lamina(layer_1, 45)
    lam.add_lamina(layer_1, -45)
    lam.add_lamina(layer_1, -45)
    lam.add_lamina(layer_1, 45)

    print(lam.ABD_matrix()[3:, 3:].round(2))
    # print(lam.lamina[1].matrices.Q_bar_reduced * 1e-9)


def testing():
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


def notes_p_56():

    E = np.array([181, 10.3, 10.3]) * 1e9
    v = 0.28
    G = 7.17e9

    alpha = np.array([-0.018, 24.3, 24.3]) * 1e-6
    beta = np.array([146, 4770, 4770]) * 1e-6

    NM = np.array([1000, 1000, 0, 0, 0, 0])

    lam = Laminate()
    mat = Material(E, v, G, alpha, beta)
    layer_1 = Lamina(mat_composite=mat, thickness=5e-3)

    lam.add_lamina(layer_1, 0)
    lam.add_lamina(layer_1, 30)
    lam.add_lamina(layer_1, -45)

    sigma = create_tensor_3D(1000 / 5e-3, 1000 / 5e-3, 0)

    lam.apply_load(NM)
    # lam.apply_stress(sigma)

    # NOT WORKING BELOW
    strain_top_30 = lam.get_state_at_height(-2.5e-3)

    strain = create_tensor_3D(*strain_top_30)
    lam.lamina[1].apply_strain(strain)

    print(lam.lamina[0].matrices.T_2D.dot(strain_top_30))

    # print(strain_top_30)


def test_2D():
    E = np.array([180, 20, 20])
    v = np.array([0, 0.3, 0.3])
    G = np.array([1, 5, 5])

    lam = Laminate()
    mat = Material(E, v, G)
    layer_1 = Lamina(mat_composite=mat)
    lam.add_lamina(layer_1, 60)

    sigma = create_tensor_3D(50, 10, -10)

    lam.apply_stress(sigma)


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


def web_problem():
    # https://courses.washington.edu/mengr450/CLT_Summary.pdf
    E_1 = np.array([25, 1.5, 1.5]) * 1e6
    v_1 = np.array([0, 0.3, 0.3])
    G_1 = np.array([1, 1.9, 1.9]) * 1e6
    a_1 = np.array([-0.5, 15, 0, 0, 0, 0]) * 1e-6

    E_2 = np.array([8, 2.3, 2.3]) * 1e6
    v_2 = np.array([0, 0.28, 0.28])
    G_2 = np.array([1, 1.1, 1.1]) * 1e6
    a_2 = np.array([3.7, 14, 0, 0, 0, 0]) * 1e-6

    lam = Laminate()

    mat_1 = Material(E_1, v_1, G_1, a_1)
    mat_2 = Material(E_2, v_2, G_2, a_2)

    layer_1 = Lamina(mat_composite=mat_1, thickness=0.005)
    layer_2 = Lamina(mat_composite=mat_2, thickness=0.005)

    lam.add_lamina(layer_1, 0)
    lam.add_lamina(layer_2, 30)
    lam.add_lamina(layer_1, 90)
    lam.add_lamina(layer_2, -30)

    NM = np.array([520, 377, 64.4, -4, 0.22, -0.0854])
    # lam.apply_load(NM)

    dT = -275
    # print(lam.lamina[1].get_lamina_expansion_properties())

    # ABD = lam.ABD_matrix()


if __name__ == '__main__':
    # main()
    # validation_1()
    # validation_2()
    # validation_3()
    # validation_4()
    # validation_5()
    # validation_6()
    # validation_7()
    validation_8()
    # notes_p_56()
    # test_2D()
    # web_problem()

