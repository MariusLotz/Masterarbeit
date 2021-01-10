import scipy.interpolate as pol
from Interpolation.get_data import get_data as get_data


def interpolater(type = 0, x=get_data()[0], fx=get_data()[1], fill_value = float):
    if type == 0:
        EurCall = []
        for el in fx:
            EurCall.append(el[0])
        # Lineare Interpolationsfunktion:
        EurCall_polator = pol.LinearNDInterpolator(x, EurCall)
        return EurCall_polator

    if type == 1:
        AmCall = []
        for el in fx:
            AmCall.append(el[1])
        # Lineare Interpolationsfunktion:
        EurCall_polator = pol.LinearNDInterpolator(x, AmCall)
        return EurCall_polator

    if type == 2:
        AmEurCall = []
        for el in fx:
            AmEurCall.append(el[2])
        # Lineare Interpolationsfunktion:
        AmEurCall_polator = pol.LinearNDInterpolator(x, AmEurCall)
        return AmEurCall_polator

    if type == 3:
        Prem = []
        for el in fx:
            Prem.append(el[3])
        # Lineare Interpolationsfunktion:
        Prem_polator = pol.LinearNDInterpolator(x, Prem)
        return Prem_polator

if __name__=="__main__":
    interpolater()


