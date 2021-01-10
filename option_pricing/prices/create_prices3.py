from option_pricing.option_price import price_call
import numpy as np
from numpy.random import seed
from numpy.random import uniform

def create_prices():
    """randomGen Settings"""
    # seed random number generator
    seed(1)
    # generate random numbers between 0-1
    random_numbers = uniform(size=3000)
    v1 = random_numbers[:1000]
    v2 = random_numbers[1000:2000]
    v3 = random_numbers[2000:3000]

    """Simple for-loop for option price creation"""
    # Settings:
    K=100
    r=0
    #divR=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    divR=[0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115]
    #S=[70, 80, 90, 100, 110, 120, 130, 140]
    S=[72, 77, 82, 87, 92, 97, 102, 107, 112, 117, 122, 127, 132, 137]
    #SIGMA=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    SIGMA=[0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57]
    m=12
    file = open('price_E0E6p_9Jan_UniformTest2100txt', 'w')
    ############################################
    for s in S:
        for v in SIGMA:
            for q in divR:
                price_am = price_call(s, K, v, r, q, 1, 1, 2020, 1, m, 2021, engine=0)
                price_eu = price_call(s, K, v, r, q, 1, 1, 2020, 1, m, 2021, engine=6)
                alpha = price_am / price_eu
                prem = price_am - price_eu
                x = [s, np.log(s/K), K, v, q, r, price_eu, price_am, alpha, prem]
                #if alpha<10 and alpha>=1:
                if alpha>3:
                    print(str(x))
                file.write(str(x))
                file.write("\n")
    file.close()

if __name__ == '__main__':
    create_prices()