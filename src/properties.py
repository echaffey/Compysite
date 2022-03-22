import numpy as np
from dataclasses import dataclass, fields


@dataclass
class StateProperties:
    stress: np.ndarray = None
    strain: np.ndarray = None


def type_check(properties):
    '''
    Create vectors for variables that are passed in as single values

    Args:
        properties (Properties): Lamina or material properties object to check.

    Returns:
        Properties: Properties object with all attributes being vectors of the appropriate length.
    '''
    p = properties

    for field in fields(properties):
        arg = getattr(p, field.name)

        if isinstance(arg, (float, int)):
            setattr(p, field.name, np.ones(3) * arg)

        elif arg is None:
            setattr(p, field.name, np.zeros(3))

        else:
            pass

    return p
