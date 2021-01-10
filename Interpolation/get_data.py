import numpy as np

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
        x = [#float(line.strip('][').split(', ')[0]), #S
            float(line.strip('][').split(', ')[1]), # log(S/K=100)
             float(line.strip('][').split(', ')[3]), # sigma
             float(line.strip('][').split(', ')[4])] # q
        x_list.append(x)
        fx = [float(line.strip('][').split(', ')[-4][:-1]), # EurCall
            float(line.strip('][').split(', ')[-3][:-1]), # AmCall
            float(line.strip('][').split(', ')[-2][:-1]), # alpha
            float(line.strip('][').split(', ')[-1][:-2])] # premium
        fx_list.append(fx)
    file.close()

    return x_list, fx_list