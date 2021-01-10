from option_pricing.option_price import price_call
import numpy as np

def create_prices():
    """Simple for-loop for option price creation."""
    # Settings:
    K=100
    r=0
    divR=[0.03, 0.05, 0.07, 0.09, 0.11]
    S=[75, 85, 95, 105, 115, 125, 135]
    SIGMA=[0.15, 0.25, 0.35, 0.45, 0.55]
    m=12
    file = open('price_E0E6p_26Novb.txt', 'w')
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