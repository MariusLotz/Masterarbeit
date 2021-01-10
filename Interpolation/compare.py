from get_data import get_data
from linear_interpolation import interpolater as interpolater


def main(type=3,
         data_for_polater=get_data(
             "/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Nov.txt"),
         real_data=get_data("/home/user/Documents/Masterarbeit/option_pricing/prices/price_E0E6p_26Novb.txt")):
    x_list = real_data[0]
    size = len(x_list)
    y_list = [real_data[1][i][type] for i in range(size)]
    polater = interpolater(type, data_for_polater[0], data_for_polater[1])
    y1_list = []
    # Creating Testprices:
    for i in range(size):
        y1_list.append(polater.__call__(x_list[i]))
    # Printing EW, VAR, Min, Max:
    complete_compare(x_list, y_list, y1_list)


def complete_compare(x_list, y_list, y1_list):
    size = len(y_list)
    sum = 0
    qu_sum = 0
    max_diff_plus = 0
    max_diff_x_plus = 0
    max_diff_minus = 0
    max_diff_x_minus = 0
    min_diff = 10 ** (27)
    min_diff_x = 0
    for i in range(size):
        res = y1_list[i] - y_list[i]
        sum =+ res
        qu_sum =+ res ** 2
        if res > max_diff_plus:
            max_diff_plus = res
            max_diff_x_plus = x_list[i]
        if res < max_diff_minus:
            max_diff_minus = res
            max_diff_x_minus = x_list[i]
        if abs(res) < abs(min_diff):
            min_diff = res
            min_diff_x = x_list[i]
    EW = sum / size
    qu_EW = qu_sum / size
    VAR = qu_EW - EW ** 2
    #print(EW, VAR)
    #print(min_diff_x, min_diff)
    print(max_diff_x_minus, max_diff_minus)
    print(max_diff_x_plus, max_diff_plus)


def compare_element(y, y1):
    res = y1 - y
    return [x, res]


def EW_res(y_list, y1_list):
    size = len(y_list)
    sum = 0
    for i in range(size):
        res = y1_list[i] - y_list[i]
        sum += res
    return (sum / size)


def Var(y_list, y1_list, mu):
    size = len(y_list)
    sum = 0
    for i in range(size):
        res = y1_list[i] - y_list[i]
        sum += (res - mu) ** 2
    return (sum / size)


if __name__ == "__main__":
    main()
