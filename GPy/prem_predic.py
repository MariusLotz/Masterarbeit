from random import random
import numpy as np
from option_pricing.option_price import price_call
from tensorflow import tfp.edwa


def get_data(dest_file='/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt'):
    """Getting data out of a text file and convert to right data type"""

    file = open(dest_file, 'r')
    x_list = []
    fx_list = []

    while True:
        # Get next line from file
        line = file.readline()
        if not line:
            break

        # Training data x = (s,sigma,q) and fx = premium
        x = [np.log(float(line.strip('][').split(', ')[0])/100), # log(S/K=100)
             float(line.strip('][').split(', ')[2]), # sigma
             float(line.strip('][').split(', ')[3])] # q
        x_list.append(x)
        fx = float(line.strip('][').split(', ')[-2][:-1]) # prem
        fx_list.append(fx)
    file.close()

    return [x_list, fx_list]

# Trainingsdata
data = get_data()
def branin(X):
    y = (X[:,1]-5.1/(4*np.pi**2)*X[:,0]**2+5*X[:,0]/np.pi-6)**2
    y += 10*(1-1/(8*np.pi))*np.cos(X[:,0])+10
    return(y)

# Training set defined as a 5*5 grid:
xg1 = np.linspace(-5,10,5)
xg2 = np.linspace(0,15,5)
X = np.zeros((xg1.size * xg2.size,2))
for i,x1 in enumerate(xg1):
    for j,x2 in enumerate(xg2):
        X[i+xg1.size*j,:] = [x1,x2]

Y = branin(X)[:,None]


# GPR modell
k = GPy.kern.RBF(input_dim=2,  variance=7., lengthscale=0.2)
model = GPy.models.GPRegression(X, Y, k)




def test():
    data = get_data()
    for i in range(10):
        r = random()
        S = 70 + r*50
        vol = r / 2
        q = r / 10
        x = np.array([np.log(S/100), vol, q])
        prem_approx = model.predict(x)
        price_am = price_call(S, 100, vol, 0, q,
          spot_day=1, spot_month=11, spot_year=2020 , mat_day=1, mat_month=11, mat_year=2021,
               engine=0)
        price_eu = price_call(S, 100, vol, 0, q,
          spot_day=1, spot_month=11, spot_year=2020 , mat_day=1, mat_month=11, mat_year=2021,
               engine=6)
        prem = price_am - price_eu
        #alpha = price_am / price_eu
        print(S, vol, q, prem, prem_approx)

test()






