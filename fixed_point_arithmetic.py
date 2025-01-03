
def fixed_point_mult(x, y, l):
    """ Performs multiplication in fixed point arithmetic between two decimal numbers x and y
    
    Parameters
    ----------
    x : float
        Decimal number
    y : float
        Decimal number
    l : int
        Bit length of fractional part of decimal numbers. Determines the scaling factor f^l
    """

    # Scale x and y by using the scaling factor
    scaling_factor = 2**l
    scaled_x = int(x * scaling_factor)
    scaled_y = int(y * scaling_factor)

    # Perform multiplication on the scaled numbers
    z = scaled_x * scaled_y

    # Truncate the resulting product
    truncated_z = z // scaling_factor

    # Convert the number back to decimal/float and return
    real_z = truncated_z / scaling_factor
    return real_z