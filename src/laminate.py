import numpy as np
from typing import List
from dataclasses import dataclass

from lamina import Lamina
from properties import StateProperties
from conversion import tensor_to_vec


@dataclass
class LaminateProperties:
    thickness: float = 0.0
    num_layers: int = 0
    length: float = 0.0
    width: float = 0.0


class Laminate:
    def __init__(self, length: int = 0, width: int = 0, symmetric: bool = True):

        self._num_plys = 0
        self.props = LaminateProperties(length=length, width=width)
        self._thickness = 0
        self._length = length
        self._width = width
        self._symmetric = symmetric
        self.lamina: List[Lamina] = []
        self.global_state: List[StateProperties] = []

    def __str__(self):

        desc = f'''
        - Layers: {self.props.num_layers}
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
        self.props.thickness += lamina_copy.props.thickness
        self.props.num_layers += 1

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

        for lamina in self.lamina:
            # A matrix is working
            if A is not None:
                A += lamina.matrices.Q_bar_reduced * lamina.props.thickness
            else:
                A = lamina.matrices.Q_bar_reduced * lamina.props.thickness

            # B not working
            # B matrix needs square of superior and inferior layer heights
            if B is not None:
                B += 0.5 * lamina.matrices.Q_bar_reduced * lamina.props.thickness ** 2
            else:
                B = 0.5 * lamina.matrices.Q_bar_reduced * lamina.props.thickness ** 2

        return A

