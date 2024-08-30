class Constants:
    def __init__(self):
        self.__EOTVOS = pow(10, -9)
        self.__GM = 3.986004418 * pow(10, 14)
        self.__A = 6378137.0
        self.__Nmax = 360
        self.__PRECISION = 20

    @property
    def EOTVOS(self):
        return self.__EOTVOS

    def Gm(self):
        return self.__GM

    def A(self):
        return self.__A

    def Nmax(self):
        return self.__Nmax

    def PRECISION(self):
        return self.__PRECISION
