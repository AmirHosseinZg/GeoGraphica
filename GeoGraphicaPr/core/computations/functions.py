from GeoGraphicaPr.Plotter.core.computations import Constant, EGM96_data
from mpmath import mp, mpf, sqrt, factorial
import numpy as np
import threading
import os

# Load EGM96 data and constants
EGM96_data_dictionary = EGM96_data.data
constants = Constant.Constants()

EOTVOS = constants.EOTVOS
Gm = constants.Gm()
A = (constants.get_a())
Nmax = constants.Nmax()
PRECISION = constants.PRECISION()
f = (constants.get_f())
e2 = (constants.get_e2())
G = (constants.get_G())
p = (constants.get_p())

# Set the precision for mpmath operations
mp.dps = PRECISION

# Initialize a dictionary to store computed Legendre polynomial data
legendre_data = {}

# Part two of Txx function calculation
part_two = mpf(0)

# os.cpu_count() returns the number of cpu cores ( Determine number of CPU cores)
number_of_threads = os.cpu_count()
chunk_size = (Nmax + 1) // number_of_threads

# Initialize a single lock for thread-safe access to data
threading_lock = threading.Lock()

counter = 1


def convert_seconds(seconds):
    """
    Convert a given number of seconds into days, hours, minutes, and seconds.

    Args:
        seconds (int): The total number of seconds to be converted.

    Returns:
        str: A string in the format 'X days, X hours, X minutes, X seconds'.
    """
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


def legendre_data_existence(n, m):
    """
    Check if the Legendre polynomial data for given indices n, m exists.

    Parameters:
    - n (int): Degree of the polynomial.
    - m (int): Order of the polynomial.

    Returns:
    - bool: True if data exists, False otherwise.
    """
    return (n in legendre_data) and (m in legendre_data[n])


def retrieve_legendre_data(n, m):
    """
    Retrieve the stored Legendre polynomial data for given indices n, m.

    Parameters:
    - n (int): Degree of the polynomial.
    - m (int): Order of the polynomial.

    Returns:
    - mpf: The value of the stored Legendre polynomial.
    """
    return legendre_data[n][m]


def C_nm(n, m):
    """
    Retrieve the C_nm coefficient from EGM96 data.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - float: C_nm coefficient.
    """
    return EGM96_data_dictionary[n][m][0]


def S_nm(n, m):
    """
    Retrieve the S_nm coefficient from EGM96 data.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - float: S_nm coefficient.
    """
    return EGM96_data_dictionary[n][m][1]


def a_nm(n, m):
    """
    Compute the a_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed a_nm coefficient.
    """
    if abs(m) == 0 or abs(m) == 1:
        return 70
    elif 2 <= abs(m) <= n:
        sqrt_term_1 = pow(n, 2) - pow((abs(m) - 1), 2)
        sqrt_term_2 = n - abs(m) + 2
        if sqrt_term_1 < 0 or sqrt_term_2 < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in a_nm function")

        if abs(m) == 2:
            result = ((sqrt(2) / 4) * (sqrt(sqrt_term_1)) *
                      (sqrt(n + abs(m))) * (sqrt(sqrt_term_2)))
        else:
            result = ((1 / 4) * (sqrt(sqrt_term_1)) *
                      (sqrt(n + abs(m))) * (sqrt(sqrt_term_2)))
    else:
        raise ValueError(f"Invalid argument for a_nm() with n={n}, m={m}")

    return result


def b_nm(n, m):
    """
    Compute the b_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed b_nm coefficient.
    """
    if abs(m) == 0 or abs(m) == 1:
        result = ((n + abs(m) + 1) * (n + abs(m) + 2)) / (2 * (abs(m) + 1))
    elif 2 <= abs(m) <= n:
        sqrt_term = pow(n, 2) + pow(m, 2) + (3 * n) + 2
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in b_nm function")
        result = sqrt_term / 2
    else:
        raise ValueError(f"Invalid argument for b_nm() with n={n}, m={m}")

    return result


def c_nm(n, m):
    """
    Compute the c_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed c_nm coefficient.
    """
    if m == n:
        return 0
    sqrt_term_1 = pow(n, 2) - pow((abs(m) + 1), 2)
    sqrt_term_2 = n + abs(m) + 2
    if sqrt_term_1 < 0 or sqrt_term_2 < 0:
        raise ValueError(f"Invalid sqrt input at n={n}, m={m} in c_nm function")

    if abs(m) == 0:
        result = ((sqrt(2) / 4) * (sqrt(sqrt_term_1)) *
                  (sqrt(n - abs(m))) * (sqrt(sqrt_term_2)))
    elif abs(m) == 1 or 2 <= abs(m) <= n:
        result = ((1 / 4) * (sqrt(sqrt_term_1)) *
                  (sqrt(n - abs(m))) * (sqrt(sqrt_term_2)))
    else:
        raise ValueError(f"Invalid argument for c_nm() with n={n}, m={m}")

    return result


def d_nm(n, m):
    """
    Compute the d_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed d_nm coefficient.
    """
    if abs(m) == 1:
        return 0
    else:
        sqrt_product_terms = ((2 * n + 1) / (2 * n - 1)) * (pow(n, 2) - (pow(abs(m) - 1, 2))) * (n + abs(m)) * (
                n + abs(m) - 2)
        if sqrt_product_terms < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in d_nm function")

        if abs(m) == 2:
            result = sqrt(2 * sqrt_product_terms) * ((-1 / 4) * (m / abs(m)))
        elif 2 < abs(m) <= n:
            result = sqrt(sqrt_product_terms) * ((-1 / 4) * (m / abs(m)))
        else:
            raise ValueError(f"Invalid argument for d_nm() with n={n}, m={m}")
    return result


def g_nm(n, m):
    """
    Compute the g_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed g_nm coefficient.
    """
    if abs(m) == 1:
        sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (n + 1) * (n - 1)
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in g_nm function")
        result = (1 / 4) * (m / abs(m)) * sqrt(sqrt_term) * (n + 2)
    elif 2 <= abs(m) <= n:
        sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (n + abs(m)) * (n - abs(m))
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in g_nm function")
        result = (m / 2) * sqrt(sqrt_term)
    else:
        raise ValueError(f"Invalid argument for g_nm() with n={n}, m={m}")
    return result


def h_nm(n, m):
    """
    Compute the h_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed h_nm coefficient.
    """
    if abs(m) == 1:
        sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (n - 3) * (n - 2) * (n - 1) * (n + 2)
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in h_nm function")
        result = (1 / 4) * (m / abs(m)) * sqrt(sqrt_term)
    elif 2 <= abs(m) <= n:
        sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (pow(n, 2) - pow(abs(m) + 1, 2)) * (n - abs(m)) * (n - abs(m) - 2)
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in h_nm function")
        result = (1 / 4) * (m / abs(m)) * sqrt(sqrt_term)
    else:
        raise ValueError(f"Invalid argument for h_nm() with n={n}, m={m}")
    return result


def beta_nm(n, m):
    """
    Compute the beta_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed beta_nm coefficient.
    """
    if m == 0:
        return 0
    else:
        sqrt_term = (n + abs(m)) * (n - abs(m) + 1)
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in beta_nm function")

        if abs(m) == 1:
            result = ((n + 2) / 2) * sqrt(2 * sqrt_term)
        elif 1 < abs(m) <= n:
            result = ((n + 2) / 2) * sqrt(sqrt_term)
        else:
            raise ValueError(f"Invalid argument for beta_nm() with n={n}, m={m}")
    return result


def gama_nm(n, m):
    """
    Compute the gama_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed gama_nm coefficient.
    """
    if m == 0:
        result = (-1) * (n + 2) * (sqrt((n * (n + 1)) / 2))
    elif 1 <= abs(m) <= n:
        sqrt_term = (n - abs(m)) * (n + abs(m) + 1)
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in gama_nm function")
        result = (-1) * ((n + 2) / 2) * (sqrt(sqrt_term))
    else:
        raise ValueError(f"Invalid argument for gama_nm() with n={n}, m={m}")
    return result


def miu_nm(n, m):
    """
    Compute the miu_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed miu_nm coefficient.
    """
    sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (n + abs(m)) * (n + abs(m) - 1)
    if sqrt_term < 0:
        raise ValueError(f"Invalid sqrt input at n={n}, m={m} in miu_nm function")
    if abs(m) == 1:
        result = (-1) * (m / abs(m)) * ((n + 2) / 2) * (sqrt(2 * sqrt_term))
    else:
        result = (-1) * (m / abs(m)) * ((n + 2) / 2) * (sqrt(sqrt_term))
    return result


def v_nm(n, m):
    """
    Compute the v_nm coefficient for Legendre polynomial.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.

    Returns:
    - The computed v_nm coefficient.
    """
    sqrt_term = ((2 * n + 1) / (2 * n - 1)) * (n - abs(m)) * (n - abs(m) - 1)
    if sqrt_term < 0:
        raise ValueError(f"Invalid sqrt input at n={n}, m={m} in v_nm function")
    result = (-1) * (m / abs(m)) * ((n + 2) / 2) * (sqrt(sqrt_term))
    return result


def legendre_recurrence(n, m, x):
    """
    Compute the associated Legendre polynomial P_n^m(x) using recurrence relation.

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.
    - x (mpf): Argument of the polynomial.

    Returns:
    - mpf: The value of P_n^m(x).
    """
    pmm = mp.mpf(1.0)
    if m > 0:
        somx2 = sqrt((1 - x) * (1 + x))
        fact = mp.mpf(1.0)
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2

    if n == m:
        return pmm

    pmmp1 = x * (2 * m + 1) * pmm
    if n == m + 1:
        return pmmp1

    for i in range(m + 2, n + 1):
        pmm, pmmp1 = pmmp1, ((2 * i - 1) * x * pmmp1 - (i + m - 1) * pmm) / (i - m)

    return pmmp1


def normal_pnm(n, m, t):
    """
    Compute the normalized associated Legendre polynomial P_n^m(t).

    Parameters:
    - n (int): Degree index.
    - m (int): Order index.
    - t (mpf): Argument of the polynomial.

    Returns:
    - mpf: The normalized value of P_n^m(t).
    """
    n, m = mpf(n), mpf(m)

    if abs(m) > n:
        return 0  # if m > n then (n-abs(m))! is not defined ( negative factorial not defined)

    delta_m0 = mpf(1 if m == 0 else 0)

    normalization_factor = sqrt((2 * n + 1) * (2 - delta_m0) * (factorial(n - abs(m)) / factorial(n + abs(m))))

    legendre_value = mpf(legendre_recurrence(int(n), int(abs(m)), t))

    result = normalization_factor * ((-1) ** int(m)) * legendre_value

    return result


def compute_Txx_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Txx function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread Started processing compute_Txx_chunk from n={lower_bound} to n={upper_bound}")
    for n in range(lower_bound, upper_bound):
        for m in range(0, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients a, b, c and convert to Decimal
                a = mpf(float(a_nm(n, m)))
                b = mpf(float(b_nm(n, m)))
                c = mpf(float(c_nm(n, m)))

                legendre_m_2 = retrieve_legendre_data(n, m - 2)
                legendre_m = retrieve_legendre_data(n, m)
                legendre_m_2_plus = retrieve_legendre_data(n, m + 2)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * cos_term)) + mpf((S * sin_term))) * (
                        mpf(a * legendre_m_2) + (mpf((b - (n + 1) * (n + 2))) * legendre_m) + (
                        c * legendre_m_2_plus))

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")
    print(f"Thread finished processing compute_Txx_chunk from n={lower_bound} to n={upper_bound}")


def compute_Txy_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Txy function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread Started processing compute_Txy_chunk from n={lower_bound} to n={upper_bound}")

    for n in range(lower_bound, upper_bound):
        for m in range(1, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients d, g, h and convert to Decimal
                d = mpf(float(d_nm(n, m)))
                g = mpf(float(g_nm(n, m)))
                h = mpf(float(h_nm(n, m)))

                legendre_m_2 = retrieve_legendre_data(n - 1, m - 2)
                legendre_m = retrieve_legendre_data(n - 1, m)
                legendre_m_2_plus = retrieve_legendre_data(n - 1, m + 2)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * sin_term)) - mpf((S * cos_term))) * (
                        mpf(d * legendre_m_2) + mpf(g * legendre_m) + (
                        h * legendre_m_2_plus))

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")

    print(f"Thread finished processing compute_Txy_chunk from n={lower_bound} to n={upper_bound}")


def compute_Txz_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Txz function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread Started processing compute_Txz_chunk from n={lower_bound} to n={upper_bound}")

    for n in range(lower_bound, upper_bound):
        for m in range(0, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients d, g, h and convert to Decimal
                beta = mpf(float(beta_nm(n, m)))
                gama = mpf(float(gama_nm(n, m)))

                legendre_m_1 = retrieve_legendre_data(n, m - 1)
                legendre_m_1_plus = retrieve_legendre_data(n, m + 1)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * cos_term)) + mpf((S * sin_term))) * (
                        mpf(beta * legendre_m_1) + mpf(gama * legendre_m_1_plus)
                )

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")

    print(f"Thread finished processing compute_Txz_chunk from n={lower_bound} to n={upper_bound}")


def compute_Tyy_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Tyy function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread started processing compute_Tyy_chunk from n={lower_bound} to n={upper_bound}")

    for n in range(lower_bound, upper_bound):
        for m in range(0, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients a, b, c and convert to Decimal
                a = mpf(float(a_nm(n, m)))
                b = mpf(float(b_nm(n, m)))
                c = mpf(float(c_nm(n, m)))

                legendre_m_2 = retrieve_legendre_data(n, m - 2)
                legendre_m = retrieve_legendre_data(n, m)
                legendre_m_2_plus = retrieve_legendre_data(n, m + 2)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * cos_term)) + mpf((S * sin_term))) * (
                        mpf(a * legendre_m_2) + mpf(b * legendre_m) + mpf(
                    c * legendre_m_2_plus))

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")

    print(f"Thread finished processing compute_Tyy_chunk from n={lower_bound} to n={upper_bound}")


def compute_Tyz_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Tyz function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread started processing compute_Tyz_chunk from n={lower_bound} to n={upper_bound}")

    for n in range(lower_bound, upper_bound):
        for m in range(1, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients d, g, h and convert to Decimal
                miu = mpf(float(miu_nm(n, m)))
                v = mpf(float(v_nm(n, m)))

                legendre_m_1 = retrieve_legendre_data(n - 1, m - 1)
                legendre_m_1_plus = retrieve_legendre_data(n - 1, m + 1)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * sin_term)) - mpf((S * cos_term))) * (
                        mpf(miu * legendre_m_1) + (
                        v * legendre_m_1_plus))

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")

    print(f"Thread finished processing compute_Tyz_chunk from n={lower_bound} to n={upper_bound}")


def compute_Tzz_chunk(r, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Tzz function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    global legendre_data
    global threading_lock
    ratio = mpf(A) / mpf(r)

    print(f"Thread started processing compute_Tzz_chunk from n={lower_bound} to n={upper_bound}")

    for n in range(lower_bound, upper_bound):
        for m in range(0, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                legendre_m = retrieve_legendre_data(n, m)

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = ((mpf(n + 1)) * (mpf(n + 2)) * power_term * (mpf((C * cos_term)) + mpf((S * sin_term))) * (
                    mpf(legendre_m)))

                with threading_lock:
                    part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")

    print(f"Thread finished processing compute_Tzz_chunk from n={lower_bound} to n={upper_bound}")


def Txx_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Txx function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Txx function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        t = np.sin(phi)

        print(f"{phi=}, {landa=}")

        for n in range(2, 361):
            for m in range(0, n + 1):
                # Calculate the Legendre functions element (p_nm(n,m-2))
                existence_status = legendre_data_existence(n, m - 2)
                if not existence_status:
                    legendre_m_2 = normal_pnm(n, m - 2, t)
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m - 2] = legendre_m_2

                # Calculate the Legendre functions element (p_nm(n,m))
                existence_status = legendre_data_existence(n, m)
                if not existence_status:
                    legendre_m = normal_pnm(n, m, t)
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m] = legendre_m

                # Calculate the Legendre functions element (p_nm(n,m+2))
                existence_status = legendre_data_existence(n, m + 2)
                if not existence_status:
                    legendre_m_2_plus = normal_pnm(n, m + 2, t)
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m + 2] = legendre_m_2_plus

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Txx_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Txx function value: {e}")


def Txy_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Txy function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Txy function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        t = np.sin(phi)

        print(f"{phi=}, {landa=}")

        for n in range(2, 361):
            for m in range(1, n + 1):
                # Calculate the Legendre functions element (p_nm(n-1,m-2))
                existence_status = legendre_data_existence(n - 1, m - 2)
                if not existence_status:
                    legendre_m_2 = normal_pnm(n - 1, m - 2, t)
                    if (n - 1) not in legendre_data:
                        legendre_data[n - 1] = {}
                    legendre_data[n - 1][m - 2] = legendre_m_2

                # Calculate the Legendre functions element (p_nm(n-1,m))
                existence_status = legendre_data_existence(n - 1, m)
                if not existence_status:
                    legendre_m = normal_pnm(n - 1, m, t)
                    if (n - 1) not in legendre_data:
                        legendre_data[n - 1] = {}
                    legendre_data[n - 1][m] = legendre_m

                # Calculate the Legendre functions element (p_nm(n-1,m+2))
                existence_status = legendre_data_existence(n - 1, m + 2)
                if not existence_status:
                    legendre_m_2_plus = normal_pnm(n - 1, m + 2, t)
                    if (n - 1) not in legendre_data:
                        legendre_data[n - 1] = {}
                    legendre_data[n - 1][m + 2] = legendre_m_2_plus

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Txy_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Txy function value: {e}")


def Txz_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Txz function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Txz function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        t = np.sin(phi)

        print(f"{phi=}, {landa=}")

        for n in range(2, 361):
            for m in range(0, n + 1):
                # Calculate the Legendre functions element (p_nm(n,m-1))
                existence_status = legendre_data_existence(n, m - 1)
                if not existence_status:
                    legendre_m_1 = normal_pnm(n, m - 1, t)
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m - 1] = legendre_m_1

                # Calculate the Legendre functions element (p_nm(n,m+1))
                existence_status = legendre_data_existence(n, m + 1)
                if not existence_status:
                    legendre_m_1_plus = normal_pnm(n, m + 1, t)
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m + 1] = legendre_m_1_plus

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Txz_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Txz function value: {e}")


def Tyy_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Tyy function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Tyy function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(-1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        t = np.sin(phi)

        print(f"{phi=}, {landa=}")

        for n in range(2, 361):
            for m in range(0, n + 1):
                # Calculate the Legendre functions element (p_nm(n,m-2))
                existence_status = legendre_data_existence(n, m - 2)
                if not existence_status:
                    legendre_m_2 = normal_pnm(n, m - 2, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m - 2] = legendre_m_2

                # Calculate the Legendre functions element (p_nm(n,m))
                existence_status = legendre_data_existence(n, m)
                if not existence_status:
                    legendre_m = normal_pnm(n, m, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m] = legendre_m

                # Calculate the Legendre functions element (p_nm(n,m+2))
                existence_status = legendre_data_existence(n, m + 2)
                if not existence_status:
                    legendre_m_2_plus = normal_pnm(n, m + 2, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m + 2] = legendre_m_2_plus

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Tyy_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Tyy function value: {e}")


def Tyz_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Tyz function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Tyz function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        for n in range(2, 361):
            for m in range(1, n + 1):
                # Calculate the Legendre functions element (p_nm(n-1,m-1))
                existence_status = legendre_data_existence(n - 1, m - 1)
                if not existence_status:
                    legendre_m_1 = normal_pnm(n - 1, m - 1, np.sin(phi))
                    if (n - 1) not in legendre_data:
                        legendre_data[n - 1] = {}
                    legendre_data[n - 1][m - 1] = legendre_m_1

                # Calculate the Legendre functions element (p_nm(n-1,m+1))
                existence_status = legendre_data_existence(n - 1, m + 1)
                if not existence_status:
                    legendre_m_1_plus = normal_pnm(n - 1, m + 1, np.sin(phi))
                    if (n - 1) not in legendre_data:
                        legendre_data[n - 1] = {}
                    legendre_data[n - 1][m + 1] = legendre_m_1_plus

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Tyz_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Tyz function value: {e}")


def Tzz_function(r, phi, landa, maximum_of_counter):
    """
    Calculate the Tzz function value for the given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.

    Returns:
    - mpf: The result of the Tzz function.
    """
    try:
        global legendre_data
        global part_two
        global counter

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        threads_pool = []

        for n in range(2, 361):
            for m in range(0, n + 1):
                # Calculate the Legendre functions element (p_nm(n,m))
                existence_status = legendre_data_existence(n, m)
                if existence_status:
                    legendre_m = retrieve_legendre_data(n, m)
                else:
                    legendre_m = normal_pnm(n, m, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m] = legendre_m

        # Create and start threads manually
        for i in range(number_of_threads):
            lower_bound = i * chunk_size + 2
            upper_bound = min((i + 1) * chunk_size + 2, Nmax + 1)
            thread = threading.Thread(target=compute_Tzz_chunk, args=(r, phi, landa, lower_bound, upper_bound))
            threads_pool.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads_pool:
            thread.join()

        # Final result
        result = part_one * part_two

        if counter == maximum_of_counter:
            legendre_data.clear()
            print("legendre_data cleared !!")
            counter = 1
        else:
            counter += 1

        part_two = mpf(0)

        print('--------------------------------')
        print(f'{result=}')
        print('--------------------------------')
        return result

    except Exception as e:
        print(f"Error during calculating Tzz function value: {e}")
