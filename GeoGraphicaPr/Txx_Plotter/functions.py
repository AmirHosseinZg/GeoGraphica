import math
import numpy as np
from mpmath import mp, factorial, power
from GeoGraphicaPr.Txx_Plotter import EGM96_data, Constant

EGM96_data_dictionary = EGM96_data.data
constants = Constant.Constants()
mp.dps = 50  # Decimal places of precision


def C_nm(n, m):
    return EGM96_data_dictionary[n][m][0]


def S_nm(n, m):
    return EGM96_data_dictionary[n][m][1]


def a_nm(n, m):
    if abs(m) == 0 or abs(m) == 1:
        result = 70
    elif 2 <= abs(m) <= n:
        if abs(m) == 2:
            result = ((math.sqrt(2) / 4) * (math.sqrt(pow(n, 2) - pow((abs(m) - 1), 2))) * (math.sqrt(n + abs(m))) * (
                math.sqrt(n - abs(m) + 2)))
        else:
            result = ((1 / 4) * (math.sqrt(pow(n, 2) - pow((abs(m) - 1), 2))) * (math.sqrt(n + abs(m))) * (
                math.sqrt(n - abs(m) + 2)))

    else:
        raise Exception("invalid argument ; a_nm()")

    return result


def b_nm(n, m):
    if abs(m) == 0 or abs(m) == 1:
        result = (((n + abs(m) + 1) * (n + abs(m) + 2)) / 2 * (abs(m) + 1))
    elif 2 <= abs(m) <= n:
        result = (pow(n, 2) + pow(m, 2) + 3 * n + 2) / 2

    else:
        raise Exception("invalid argument ; b_nm()")

    return result


def c_nm(n, m):
    if abs(m) == 0 or abs(m) == 1:
        if abs(m) == 0:
            result = ((math.sqrt(2) / 4) * (math.sqrt(pow(n, 2) - pow((abs(m) + 1), 2))) * (math.sqrt(n - abs(m))) * (
                math.sqrt(n + abs(m) + 2)))
        else:
            result = ((1 / 4) * (math.sqrt(pow(n, 2) - pow((abs(m) + 1), 2))) * (math.sqrt(n - abs(m))) * (
                math.sqrt(n + abs(m) + 2)))

    elif 2 <= abs(m) <= n:
        result = ((1 / 4) * (math.sqrt(pow(n, 2) - pow((abs(m) + 1), 2))) * (math.sqrt(n - abs(m))) * (
            math.sqrt(n + abs(m) + 2)))
    else:
        raise Exception("invalid argument ; c_nm()")

    return result


def P_nm(n, m, t):
    # Determine the largest integer r such that r <= (n - m) / 2
    r_max = (n - m) // 2

    # Initialize the sum
    sum_result = mp.mpf(0)

    # Compute the sum
    for k in range(r_max + 1):
        # Check if the factorial arguments are non-negative
        if (n - m - 2 * k) < 0:
            continue  # Skip this term since factorial of a negative number is not defined

        # Compute the numerator and denominator using mpmath functions
        numerator = power(-1, k) * factorial(2 * n - 2 * k)
        denominator = factorial(k) * factorial(n - k) * factorial(n - m - 2 * k)

        # Add the term to the sum
        sum_result += (numerator / denominator) * power(t, n - m - 2 * k)

    # Compute the prefactor using mpmath
    prefactor = power(2, -n) * power(1 - t ** 2, m / 2)

    # Final result
    return prefactor * sum_result


def Txx_function(r, phi, landa):
    try:
        part_one = (1 / constants.EOTVOS) * (constants.Gm() / pow(constants.A(), -3))

        part_two = 0
        for n in range(2, constants.Nmax() + 1):
            for m in range(0, n + 1):
                try:
                    # Fetch coefficients
                    C = C_nm(n, m)
                    S = S_nm(n, m)

                    # Check if values are within a valid range before calculating
                    if n < m or m < 0:
                        raise ValueError(f"Invalid n and m values: n={n}, m={m}")

                    # Ensure the inputs to sqrt and other functions are valid
                    if n ** 2 - (m - 1) ** 2 < 0 or n - m + 2 < 0:
                        raise ValueError(f"Math domain error at n={n}, m={m}")

                    a = a_nm(n, m)
                    b = b_nm(n, m)
                    c = c_nm(n, m)

                    # Check for large results before performing operations
                    if abs(a) > 1e308 or abs(b) > 1e308 or abs(c) > 1e308:
                        raise OverflowError(f"Value too large to handle at n={n}, m={m}")

                    # Calculate Legendre functions
                    legendre_m_2 = P_nm(n, m - 2, np.sin(phi))
                    legendre_m = P_nm(n, m, np.sin(phi))
                    legendre_m_2_plus = P_nm(n, m + 2, np.sin(phi))

                    # Accumulate part_two
                    part_two += (pow((constants.A() / r), n + 3) *
                                 ((C * np.cos(m * landa)) + (S * np.sin(m * landa))) *
                                 ((a * legendre_m_2) +
                                  ((b - (n + 1) * (n + 2)) * legendre_m) +
                                  (c * legendre_m_2_plus)))
                except KeyError as ke:
                    print(f"KeyError: n={n}, m={m} - {ke}")
                except ValueError as ve:
                    print(f"ValueError: n={n}, m={m} - {ve}")
                except OverflowError as oe:
                    print(f"OverflowError: n={n}, m={m} - {oe}")
                except Exception as e:
                    print(f"Error in iteration n={n}, m={m}: {e}")

        result = part_one * part_two
        return result
    except Exception as e:
        print(f"Error during calculating Txx function value: {e}")
