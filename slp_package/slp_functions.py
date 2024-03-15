def one_hot_encode_flags(flag_enum, bitmask):
    """
    One-hot encode the set flags from the given bitmask using the specified IntFlag enumeration.

    :param flag_enum: An IntFlag enumeration class with defined flags.
    :param bitmask: The integer bitmask representing the set flags.
    :return: List of integers representing the one-hot encoded flags.
    """
    # Initialize a list to store the one-hot encoded values
    one_hot_encoded = [0] * len(flag_enum)

    # Iterate through the flags in the enumeration
    for i, flag in enumerate(flag_enum):
        if bitmask & flag:
            one_hot_encoded[i] = 1

    return one_hot_encoded