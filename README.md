# Compysite

<i> Work in Progress </i>

Compysite is a python library designed to bring a simple, object-oriented approach to engineering analysis of fiber reinforced composite materials.  This library makes use of Classical Laminate Theory (CLT) for orthotropic composites. 

### Functionality

This package is written in Python and utilizes numpy. It allows for the creation of materials, lamina and laminate stacks for engineering analysis using Classical Lamination Theory (CLT).  Compysite allows for the application of loads, strains, stresses, as well as thermal and moisture effects. 

### Requirements

This package requires python version 3.70 or later, numpy and matplotlib.  
```
pip install numpy matplotlib
```

### Run Compysite with Python



To create a composite material from its constituent fiber and matrix materials:
```python
import numpy as np
from Compysite import Material, Lamina, Laminate

# Fiber material properties in the [23, 13, 12] directions
E_f = np.array([233, 23.1, 23.1])*1e9
v_f = np.array([0.40, 0.20, 0.20])
G_f = np.array([8.27, 8.96, 8.96])*1e9
alpha_f = np.array([-0.54, 10.10, 10.10])*1e-6
V_f = 0.61   # Fiber volume fraction

# Matrix material properties can also be given as an isotropic material
E_m = 4.62
v_m = 0.36
G_m = 0
alpha_m = 41.4

# Create the materials from their respective properties
mat_f = Material(E_f, v_f, G_f, alpha_f)
mat_m = Material(E_m, v_m, G_m, alpha_m)

# Assemble the fiber and matrix materials into a composite lamina
layer = Lamina(mat_fiber=mat_f, mat_matrix=mat_m, Vol_fiber=V_f)
```

The created lamina object now gives access to the composite material (a ```Material``` object) as well as its underlying compliance (```S```) and stiffness (```C```) matrices in both three and two (```S_reduced```, ```C_reduced```) dimensions.

```python
E, v, G = layer.get_material_properties()
alpha, beta = layer.get_expansion_properties()

S = layer.matrices.S
C = layer.matrices.C
```

To create a composite stack, individual lamina can be added with a given orientation:
```python
E = np.array([181, 10.3, 10.3]) * 1e9
v = np.array([0, 0.28, 0.28])
G = np.array([1, 7.17, 7.17]) * 1e9

lam = Laminate()
mat = Material(E, v, G)
layer = Lamina(mat_composite=mat, thickness=5e-3)

# Create a symmetric [0/30/30/0] laminate stack
lam.add_lamina(layer, 0)
lam.add_lamina(layer, 30)
lam.add_lamina(layer, 30)
lam.add_lamina(layer, 0)
```

Loads, stresses, strains can now all be applied to the composite to calculate the global and local effects on the laminate:

```python
# import a helper function from the included conversion module
from conversion import create_tensor_3D

# create a stress tensor
sigma = create_tensor_3D(50, -50, -5, 0, 0, -3) * 1e6

lam.apply_stress(sigma)

# Get the global laminate stress/strain state
print(lam.global_state)

# Get individual lamina stress/strain state (layer 2, 30 degree orientation here)
print(lam.get_lamina(2).local_state)
```
