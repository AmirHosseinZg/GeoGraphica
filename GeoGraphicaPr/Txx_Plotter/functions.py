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
        return 70
    elif 2 <= abs(m) <= n:
        sqrt_term_1 = pow(n, 2) - pow((abs(m) - 1), 2)
        sqrt_term_2 = n - abs(m) + 2
        if sqrt_term_1 < 0 or sqrt_term_2 < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in a_nm function")

        if abs(m) == 2:
            result = ((math.sqrt(2) / 4) * (math.sqrt(sqrt_term_1)) *
                      (math.sqrt(n + abs(m))) * (math.sqrt(sqrt_term_2)))
        else:
            result = ((1 / 4) * (math.sqrt(sqrt_term_1)) *
                      (math.sqrt(n + abs(m))) * (math.sqrt(sqrt_term_2)))
    else:
        raise ValueError(f"Invalid argument for a_nm() with n={n}, m={m}")

    return result


def b_nm(n, m):
    if abs(m) == 0 or abs(m) == 1:
        result = (((n + abs(m) + 1) * (n + abs(m) + 2)) / 2 * (abs(m) + 1))
    elif 2 <= abs(m) <= n:
        sqrt_term = pow(n, 2) + pow(m, 2) + 3 * n + 2
        if sqrt_term < 0:
            raise ValueError(f"Invalid sqrt input at n={n}, m={m} in b_nm function")
        result = sqrt_term / 2
    else:
        raise ValueError(f"Invalid argument for b_nm() with n={n}, m={m}")

    return result


def c_nm(n, m):
    if m == n:
        return 0
    sqrt_term_1 = pow(n, 2) - pow((abs(m) + 1), 2)
    sqrt_term_2 = n + abs(m) + 2
    if sqrt_term_1 < 0 or sqrt_term_2 < 0:
        raise ValueError(f"Invalid sqrt input at n={n}, m={m} in c_nm function")

    if abs(m) == 0 or abs(m) == 1:
        if abs(m) == 0:
            result = ((math.sqrt(2) / 4) * (math.sqrt(sqrt_term_1)) *
                      (math.sqrt(n - abs(m))) * (math.sqrt(sqrt_term_2)))
        else:
            result = ((1 / 4) * (math.sqrt(sqrt_term_1)) *
                      (math.sqrt(n - abs(m))) * (math.sqrt(sqrt_term_2)))
    elif 2 <= abs(m) <= n:
        result = ((1 / 4) * (math.sqrt(sqrt_term_1)) *
                  (math.sqrt(n - abs(m))) * (math.sqrt(sqrt_term_2)))
    else:
        raise ValueError(f"Invalid argument for c_nm() with n={n}, m={m}")

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

        # Initialize part_two as a standard float array
        part_two = np.zeros_like(phi, dtype=float)

        # Iterate over n and m
        for n in range(2, constants.Nmax() + 1):
            for m in range(0, n + 1):
                try:
                    # Fetch coefficients
                    C = C_nm(n, m)
                    S = S_nm(n, m)

                    # Calculate the coefficients a, b, c using mpmath
                    a = float(a_nm(n, m))
                    b = float(b_nm(n, m))
                    c = float(c_nm(n, m))

                    # Ensure the inputs to sqrt and other functions are valid
                    if n ** 2 - (m - 1) ** 2 < 0 or n - m + 2 < 0:
                        raise ValueError(f"Math domain error at n={n}, m={m}")

                    # Convert phi and landa to Python float if they are scalar, or process element-wise if they are arrays
                    phi_flat = np.ravel(phi)
                    landa_flat = np.ravel(landa)

                    # Calculate the Legendre functions element-wise
                    legendre_m_2 = np.array([float(P_nm(n, m - 2, np.sin(p))) for p in phi_flat])
                    legendre_m = np.array([float(P_nm(n, m, np.sin(p))) for p in phi_flat])
                    legendre_m_2_plus = np.array([float(P_nm(n, m + 2, np.sin(p))) for p in phi_flat])

                    # Reshape back to the original shape
                    legendre_m_2 = legendre_m_2.reshape(phi.shape)
                    legendre_m = legendre_m.reshape(phi.shape)
                    legendre_m_2_plus = legendre_m_2_plus.reshape(phi.shape)

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

        # Final result
        result = part_one * part_two
        return result

    except Exception as e:
        print(f"Error during calculating Txx function value: {e}")
