from mpmath import mp, mpf, sqrt, factorial
from GeoGraphicaPr.Txx_Plotter import EGM96_data, Constant
import numpy as np

# Load EGM96 data and constants
EGM96_data_dictionary = EGM96_data.data
constants = Constant.Constants()

# Constants for calculations
EOTVOS = mpf(constants.EOTVOS)
Gm = mpf(constants.Gm())
A = mpf(constants.A())
Nmax = constants.Nmax()
PRECISION = constants.PRECISION()

# Set the precision for mpmath operations
mp.dps = PRECISION

# Initialize a dictionary to store computed Legendre polynomial data
legendre_data = {}


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
    - mpf: The computed a_nm coefficient.
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
    - mpf: The computed b_nm coefficient.
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
    - mpf: The computed c_nm coefficient.
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


part_two = mpf(0)  # Part two of Txx function calculation


def Txx_function_calculation(r, phi, landa, lower_bound, upper_bound):
    """
    Perform the calculation for the Txx function using given parameters.

    Parameters:
    - r (mpf): Radial distance.
    - phi (float): Latitude in radians.
    - landa (float): Longitude in radians.
    - lower_bound (int): Lower bound for iteration.
    - upper_bound (int): Upper bound for iteration.
    """
    global part_two
    ratio = mpf(A) / mpf(r)

    for n in range(2, Nmax + 1):
        for m in range(0, n + 1):
            try:
                # Fetch coefficients
                C = mpf(C_nm(n, m))
                S = mpf(S_nm(n, m))

                # Calculate the coefficients a, b, c and convert to Decimal
                a = mpf(float(a_nm(n, m)))
                b = mpf(float(b_nm(n, m)))
                c = mpf(float(c_nm(n, m)))

                # Calculate the Legendre functions element (p_nm(n,m-2))
                existence_status = legendre_data_existence(n, m - 2)
                if existence_status:
                    legendre_m_2 = retrieve_legendre_data(n, m - 2)
                else:
                    legendre_m_2 = normal_pnm(n, m - 2, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m - 2] = legendre_m_2

                # Calculate the Legendre functions element (p_nm(n,m))
                existence_status = legendre_data_existence(n, m)
                if existence_status:
                    legendre_m = retrieve_legendre_data(n, m)
                else:
                    legendre_m = normal_pnm(n, m, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m] = legendre_m

                # Calculate the Legendre functions element (p_nm(n,m+2))
                existence_status = legendre_data_existence(n, m + 2)
                if existence_status:
                    legendre_m_2_plus = retrieve_legendre_data(n, m + 2)
                else:
                    legendre_m_2_plus = normal_pnm(n, m + 2, np.sin(phi))
                    if n not in legendre_data:
                        legendre_data[n] = {}
                    legendre_data[n][m + 2] = legendre_m_2_plus

                # Compute the power term
                power_term = mpf(pow(ratio, n + 3))

                # Calculate the term involving trigonometric functions
                cos_term = mpf(np.cos(float(m * landa)))
                sin_term = mpf(np.sin(float(m * landa)))

                # Accumulate part_two
                term = power_term * (mpf((C * cos_term)) + mpf((S * sin_term))) * (
                        mpf(a * legendre_m_2) + (mpf((b - (n + 1) * (n + 2))) * legendre_m) + (
                        c * legendre_m_2_plus))

                part_two += term

            except KeyError as ke:
                print(f"KeyError: n={n}, m={m} - {ke}")
            except ValueError as ve:
                print(f"ValueError: n={n}, m={m} - {ve}")
            except OverflowError as oe:
                print(f"OverflowError: n={n}, m={m} - {oe}")
            except Exception as e:
                print(f"Error in iteration n={n}, m={m}: {e}")


def Txx_function(r, phi, landa):
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

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        # Final result
        result = part_one * part_two
        return result

    except Exception as e:
        print(f"Error during calculating Txx function value: {e}")
