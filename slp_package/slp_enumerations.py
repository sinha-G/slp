from enum import Enum

# Enumeration of the inputs in the np arrays containing the input data
class InputDataKey(Enum):
    JSTICK_X = 0
    JSTICK_Y = 1
    CSTICK_X = 2
    CSTICK_Y = 3
    TRIGGER_LOGICAL = 4
    Z = 5
    A = 6
    B = 7
    X_or_Y = 8
    
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3   