class Constants:
    def __init__(self):
        self.__EOTVOS = pow(10, -9)
        self.__GM = 3.986004418 * pow(10, 14)
        self.__a = 6378137.0
        self.__Nmax = 360
        self.__PRECISION = 20
        self.__f = 1 / 298.257223563
        self.__e2 = (2 * self.__f) - pow(self.__f, 2)
        self.__h = 100
        self.__G = 6.6742 * pow(10, -11)
        self.__p = 2670

    @property
    def EOTVOS(self):
        return self.__EOTVOS

    def Gm(self):
        return self.__GM

    def get_a(self):
        return self.__a

    def Nmax(self):
        return self.__Nmax

    def PRECISION(self):
        return self.__PRECISION

    def get_f(self):
        return self.__f

    def get_e2(self):
        return self.__e2

    def get_h(self):
        return self.__h

    def get_G(self):
        return self.__G

    def get_p(self):
        return self.__p
