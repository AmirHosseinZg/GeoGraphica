import math

from GeoGraphicaPr.Txx_Plotter import EGM96_data, Constant

EGM96_data_dictionary = EGM96_data.data
constants = Constant.Constants()


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
        raise Exception("invalid entry")

    return result


def b_nm(n, m):
    if abs(m) == 0 or abs(m) == 1:
        result = (((n + abs(m) + 1) * (n + abs(m) + 2)) / 2 * (abs(m) + 1))
    elif 2 <= abs(m) <= n:
        result = (pow(n, 2) + pow(m, 2) + 3 * n + 2) / 2

    else:
        raise Exception("invalid entry")

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
        raise Exception("invalid entry")

    return result


def Txx_function(r, phi, landa):
    part_one = (1 / constants.EOTVOS) * (constants.Gm() / pow(constants.A(), 3))

    part_two = 0
    for n in range(2, constants.Nmax() + 1):
        for m in range(0, n + 1):
            part_two += pow((constants.A() / r), n + 3) * (
                    (C_nm(n, m) * np.cos(m * landa)) + (S_nm(n, m) * np.sin(m * landa)))

    part_three = None  # TODO
