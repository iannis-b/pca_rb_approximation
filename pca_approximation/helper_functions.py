""" Old functions, stored here for reference.

"""

import numpy as np
from crbs.crbs_molecular import crbs_molecular


def construct_training_dataset(model='s6', temp_min=200, temp_max=350, n_temp=50,
                                           p_atm_min=0.01, p_atm_max=1.5, n_p_atm=50,
                                           xi_min = -5, xi_max = 5, n_xi = 100):
    """ Calculate the input for the PCA.

    Parameters
    ----------
    temp_min, temp_max: float
       The minimum and maximum temperaure
    n_temp: int
       Number of temperature points
    p_atm_min, p_atm_max: float
       The minimum and maximum pressure
    n_p_atm: int
       Number of pressure points
    xi_min, xi_max: float
       The minimum and maximum xi for calculation
    n_xi: int
       The number of xi values to calculate
    model: string
       One of 's6', 's7'. Defines the model to use. Default 's6'

    Returns
    -------
    X: array [n_y, n_xi]
       Spontaneous scattering for every y and xi.
    y: array [n_y]
       y values
    """

    sptsigs = []  # Store the signals
    ys = []  # Store the values of y

    Ps = np.linspace(p_atm_min, p_atm_max, n_p_atm)
    Ts = np.linspace(temp_min, temp_max, n_temp)

    for T in Ts:
        for P in Ps:
            _, sptsig, _, _, y = crbs_molecular(
                         lamda=1064.0, temp=T, p_atm=P, model=model, xi_min=xi_min, xi_max=xi_max, n_xi=n_xi)
            sptsigs.append(sptsig)
            ys.append(y)

    X = np.array(sptsigs)
    ys = np.array(ys)
    xi = np.linspace(xi_min, xi_max, n_xi)

    return X, ys, xi


def save_training_dataset(filename, X, ys, xi):
    """
    Save training dataset in a text file.

    Parameters
    ----------
    filename: str
       The filename to store the data.
    X: matrix
       A set of Rayeleigh-Brillouin spectrum
    ys:
       The corresponding values of y for each spectrum.
    xi:
       The xi values for each point of the profiles.
    """
    header_txt = "Training dataset for PCA approximation of Rayleigh-Brillouin spectrum.\n" \
                 "First row contains the xi values. The first value is just a placeholder, and should be ignored.\n" \
                 "Then, each row contains the y value and corresponding RB spectrum."

    ys = np.insert(ys, 0, -99)  # insert the value -99 at the 0 position of array y.

    spectrum_matrix = np.vstack([xi, X])
    stored_data = np.vstack([ys, spectrum_matrix.T]).T

    np.savetxt(filename, stored_data, delimiter=',', header=header_txt)