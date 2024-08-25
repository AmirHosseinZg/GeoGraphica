import math
import numpy as np
from GeoGraphicaPr.Txx_Plotter import EGM96_data, Constant
from decimal import Decimal, getcontext, InvalidOperation

EGM96_data_dictionary = EGM96_data.data

constants = Constant.Constants()
EOTVOS = Decimal(constants.EOTVOS)
Gm = Decimal(constants.Gm())
A = Decimal(constants.A())
Nmax = constants.Nmax()

legendre_data = {}


def retrieve_legendre_data(n, m):
    return legendre_data[n][m]


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
    sum_result = 0

    # Compute the sum
    for k in range(r_max + 1):
        # Check if the factorial arguments are non-negative
        if (n - m - 2 * k) < 0:
            continue  # Skip this term since factorial of a negative number is not defined

        # Compute the numerator and denominator using mpmath functions
        numerator = pow(-1, k) * math.factorial(2 * n - 2 * k)
        denominator = math.factorial(k) * math.factorial(n - k) * math.factorial(n - m - 2 * k)

        # Add the term to the sum
        sum_result += (numerator / denominator) * pow(t, n - m - 2 * k)

    # Compute the prefactor using mpmath
    prefactor = pow(2, -n) * pow(1 - t ** 2, m / 2)

    # Final result
    return prefactor * sum_result


def Txx_function(r, phi, landa):
    try:
        # Set the precision for Decimal operations
        getcontext().prec = 1000

        part_one = Decimal(1 / EOTVOS * (Gm / A ** -3))

        part_two = Decimal(0)

        ratio = Decimal(A) / Decimal(r)

        # Iterate over n and m
        for n in range(2, Nmax + 1):
            for m in range(0, n + 1):
                try:
                    # Fetch coefficients
                    C = Decimal(C_nm(n, m))
                    S = Decimal(S_nm(n, m))

                    # Calculate the coefficients a, b, c using mpmath and convert to Decimal
                    a = Decimal(float(a_nm(n, m)))
                    b = Decimal(float(b_nm(n, m)))
                    c = Decimal(float(c_nm(n, m)))

                    # Calculate the Legendre functions element
                    if n in legendre_data and m - 2 in legendre_data[n]:
                        legendre_m_2 = retrieve_legendre_data(n, m - 2)
                    else:
                        legendre_m_2 = Decimal(float(P_nm(n, m - 2, np.sin(phi))))
                        legendre_data[n] = {}
                        legendre_data[n][m - 2] = legendre_m_2

                    if n in legendre_data and m in legendre_data[n]:
                        legendre_m = retrieve_legendre_data(n, m)
                    else:
                        legendre_m = Decimal(float(P_nm(n, m, np.sin(phi))))
                        legendre_data[n] = {}
                        legendre_data[n][m] = legendre_m

                    if n in legendre_data and m + 2 in legendre_data[n]:
                        legendre_m_2_plus = retrieve_legendre_data(n, m + 2)
                    else:
                        legendre_m_2_plus = Decimal(float(P_nm(n, m + 2, np.sin(phi))))
                        legendre_data[n] = {}
                        legendre_data[n][m + 2] = legendre_m_2_plus

                    # Compute the power term
                    power_term = Decimal(pow(ratio, n + 3))

                    # Calculate the term involving trigonometric functions
                    cos_term = Decimal(np.cos(float(m * landa)))
                    sin_term = Decimal(np.sin(float(m * landa)))

                    # Accumulate part_two
                    term = Decimal(power_term * ((C * cos_term) + (S * sin_term)) * (
                                (a * legendre_m_2) + ((b - (n + 1) * (n + 2)) * legendre_m) + (c * legendre_m_2_plus)))

                    # Safeguard against invalid Decimal operations
                    part_two += term

                except InvalidOperation as ioe:
                    print(f"InvalidOperation: n={n}, m={m} - {ioe}")
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
