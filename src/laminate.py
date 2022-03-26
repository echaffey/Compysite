import numpy as np
from typing import List
from dataclasses import dataclass

from lamina import Lamina
from properties import StateProperties
from conversion import tensor_to_vec


class Laminate:
    def __init__(self, length: int = 0, width: int = 0, symmetric: bool = True):

        self.num_layers: int = 0
        self.thickness: int = 0
        self.length: int = length
        self.width: int = width
        self._z: np.array = None
        self._symmetric: bool = symmetric
        self.lamina: List[Lamina] = []
        self.global_state: List[StateProperties] = []
        self.mid_plane_state: StateProperties = StateProperties()
        self._ABD: np.ndarray = None

    def __str__(self):

        desc = f'''
        - Layers: {self._num_layers}
        - Orientation:  {'/'.join([str(round(l.props.orientation*180/np.pi)) for l in self.lamina])}
        '''
        return desc

    def add_lamina(self, new_lamina: Lamina, orientation_deg: float = 0) -> None:
        '''
        Adds a new lamina layer to the laminate stack.  Updates the dimensions of the laminate and 
        recalculates the net directional stresses acting on the laminate. 

            Parameters:
                new_lamina (Lamina):         the constructed lamina object to be added to the laminate.
                orientation_deg (numpy.ndarray): orientation of the fiber directions in degrees relative to the 
                                                 principal axes in the z, y, x directions. 
        '''

        # Create a copy to allow a lamina to be reused multiple times in a laminate stack
        lamina_copy = new_lamina.copy()

        # Sets the orientation to calculate the transformed matrices
        lamina_copy.set_orientation(orientation_deg)

        self.lamina.append(lamina_copy)

        # Update laminate properties
        self.thickness += lamina_copy.props.thickness
        self.num_layers += 1

        # Determine layer heights
        self.calc_heights()

        # Construct the ABD matrix
        self._ABD = self.ABD_matrix()

    def calc_heights(self):

        # Create an array to keep track of the superior and inferior layer heights
        h = self.thickness / 2
        z = np.zeros(self.num_layers + 1)
        z[0] = -h

        for idx, lamina in enumerate(self.lamina, start=1):
            z[idx] = z[idx - 1] + lamina.props.thickness

        self._z = z.copy()

    def apply_stress(self, global_stress_tensor: np.ndarray) -> None:
        '''
        Assign global and local stress/strain state based on the given applied stress.

            Parameters:
                stress_tensor (numpy.ndarray):   Global stress tensor to be applied
        '''
        # Clear the current state
        temp_state = []

        # Iterate over each lamina in the stack
        for lamina in self.lamina:

            # Calculate the local stress and strain
            lamina.apply_stress(global_stress_tensor)

            # Calculate the global stress and strain vectors
            e_global = lamina.matrices.S_bar.dot(tensor_to_vec(global_stress_tensor))
            s_global = tensor_to_vec(global_stress_tensor)

            # Update laminate state properties list
            temp_state.append(StateProperties(s_global, e_global))

        self.global_state = temp_state

    def apply_strain(self, global_strain_tensor: np.ndarray) -> None:
        '''
        Assign global and local stress/strain state based on the given applied stress.

            Parameters:
                stress_tensor (numpy.ndarray):   Global stress tensor to be applied
        '''
        # Clear the current state
        temp_state = []

        # Iterate over each lamina in the stack
        for lamina in self.lamina:

            # Calculate the local stress and strain
            lamina.apply_strain(global_strain_tensor)

            # Calculate the global stress and strain vectors
            s_global = lamina.matrices.Q_bar.dot(tensor_to_vec(global_strain_tensor))
            e_global = tensor_to_vec(global_strain_tensor)

            # Update laminate state properties list
            temp_state.append(StateProperties(s_global, e_global))

        self.global_state = temp_state

    def apply_load(self, NM_matrix: np.ndarray) -> None:

        # Calculate the midplane strains due to the appllied loads and moments
        self.mid_plane_state.strain = np.linalg.inv(self._ABD).dot(NM_matrix)

    def get_state_at_height(self, z: int):

        e = self.mid_plane_state.strain[:3]
        k = self.mid_plane_state.strain[3:]

        strain = e + z * k

        return strain

    def get_lamina(self, layer_num: int = None) -> Lamina:
        '''
        Returns the lamina object at the given layer or the collection of all lamina in the laminate stack.

        Args:
            layer_num (int, optional): Layer to return. Index is 1 based. Defaults to None.

        Returns:
            Lamina: Selected lamina object.
        '''
        if not layer_num:
            return self.lamina

        return self.lamina[layer_num - 1]

    def ABD_matrix(self) -> np.ndarray:

        A = None
        B = None
        D = None

        for i, lamina in enumerate(self.lamina, start=1):
            # A matrix is working
            if A is not None:
                A += lamina.matrices.Q_bar_reduced * lamina.props.thickness
            else:
                A = lamina.matrices.Q_bar_reduced * lamina.props.thickness

            # B is working
            if B is not None:
                B += (
                    0.5
                    * lamina.matrices.Q_bar_reduced
                    * (self._z[i] ** 2 - self._z[i - 1] ** 2)
                )
            else:
                B = (
                    0.5
                    * lamina.matrices.Q_bar_reduced
                    * (self._z[i] ** 2 - self._z[i - 1] ** 2)
                )

            if D is not None:
                D += (
                    (1 / 3)
                    * lamina.matrices.Q_bar_reduced
                    * (self._z[i] ** 3 - self._z[i - 1] ** 3)
                )
            else:
                D = (
                    (1 / 3)
                    * lamina.matrices.Q_bar_reduced
                    * (self._z[i] ** 3 - self._z[i - 1] ** 3)
                )

        ABD = np.zeros((6, 6))
        ABD[:3, :3] = A
        ABD[3:, :3] = ABD[:3, 3:] = B
        ABD[3:, 3:] = D

        return ABD

