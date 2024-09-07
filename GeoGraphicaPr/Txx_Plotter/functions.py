from mpmath import mp, mpf, sqrt, factorial
from GeoGraphicaPr.Txx_Plotter import EGM96_data, Constant
import numpy as np

EGM96_data_dictionary = EGM96_data.data

constants = Constant.Constants()
EOTVOS = mpf(constants.EOTVOS)
Gm = mpf(constants.Gm())
A = mpf(constants.A())
Nmax = constants.Nmax()
PRECISION = constants.PRECISION()

# Set the precision for mpmath operations
mp.dps = PRECISION

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
            result = ((sqrt(2) / 4) * (sqrt(sqrt_term_1)) *
                      (sqrt(n + abs(m))) * (sqrt(sqrt_term_2)))
        else:
            result = ((1 / 4) * (sqrt(sqrt_term_1)) *
                      (sqrt(n + abs(m))) * (sqrt(sqrt_term_2)))
    else:
        raise ValueError(f"Invalid argument for a_nm() with n={n}, m={m}")

    return result


def b_nm(n, m):
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
    # Initial values for P_m^m(x) when m = 0
    pmm = mp.mpf(1.0)
    if m > 0:
        somx2 = sqrt((1 - x) * (1 + x))
        fact = mp.mpf(1.0)
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2

    if n == m:
        return pmm

    # Initial value for P_{m+1}^m(x)
    pmmp1 = x * (2 * m + 1) * pmm
    if n == m + 1:
        return pmmp1

    # Use the recurrence relation to compute higher order values
    for i in range(m + 2, n + 1):
        pmm, pmmp1 = pmmp1, ((2 * i - 1) * x * pmmp1 - (i + m - 1) * pmm) / (i - m)

    return pmmp1


def normal_pnm(n, m, t):
    # Convert n and m to mpf for high precision
    n, m = mpf(n), mpf(m)

    if abs(m) > n:
        return 0  # if m > n then (n-abs(m))! is not defined ( negative factorial not defined)

    # Calculate the Kronecker delta δ_{m,0}
    delta_m0 = mpf(1 if m == 0 else 0)

    # Compute the normalization factor using mpmath
    normalization_factor = sqrt((2 * n + 1) * (2 - delta_m0) * (factorial(n - abs(m)) / factorial(n + abs(m))))

    # Calculate the associated Legendre polynomial P_n^m(x)
    legendre_value = mpf(legendre_recurrence(int(n), int(abs(m)), t))

    # Apply the final formula including the (-1)^m term
    result = normalization_factor * ((-1) ** int(m)) * legendre_value

    return result


def Txx_function(r, phi, landa):
    try:

        part_one = (mpf(1) / mpf(EOTVOS)) * ((mpf(Gm) / (mpf(A) ** mpf(3))))

        part_two = mpf(0)

        ratio = mpf(A) / mpf(r)

        # Iterate over n and m
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

                    # Calculate the Legendre functions element
                    if n in legendre_data and m - 2 in legendre_data[n]:
                        legendre_m_2 = retrieve_legendre_data(n, m - 2)
                    else:
                        legendre_m_2 = normal_pnm(n, m - 2, np.sin(phi))
                        if n not in legendre_data:
                            legendre_data[n] = {}
                        legendre_data[n][m - 2] = legendre_m_2

                    if n in legendre_data and m in legendre_data[n]:
                        legendre_m = retrieve_legendre_data(n, m)
                    else:
                        legendre_m = normal_pnm(n, m, np.sin(phi))
                        if n not in legendre_data:
                            legendre_data[n] = {}
                        legendre_data[n][m] = legendre_m

                    if n in legendre_data and m + 2 in legendre_data[n]:
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

        # Final result
        result = part_one * part_two
        return result

    except Exception as e:
        print(f"Error during calculating Txx function value: {e}")
