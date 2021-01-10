import numpy as np

def gradient_descent_method(x0, f, gradient_f, errorbar=10**(-5), alphabar=0.1, betabar=0.7, max_iter = 999):
    """ Full method with solution x,f(x) hopefully close to x_min,f(x_min) regarding |df| < errorbar.
    For functionality please select 0 < alphabar < 1 and alphabar < betabar <1.
    Please choose errorbar not to small, at least errorbar >= 10^-7.
    Algorithm doesnt take more then max_iter steps."""

    print("starting in", x0)
    x = np.array(x0, dtype=np.float64)
    count = 0

    # Main gradient descent method:
    while np.linalg.norm(gradient_f(x)) > errorbar:
        count += 1
        current_gradient = gradient_f(x)
        direction = - current_gradient  # any downward direction is allowed

        # Finding the step size:
        step = lazy_wolfe_step(f, x, gradient_f, current_gradient, direction, alphabar, betabar, max_iter)

        # Going one step:
        x = gradient_descent_step(x, step, direction)

        # Max_iter restriction:
        if count > max_iter:
            break

    return x, f(x)


def gradient_descent_step(x, step, direction):
    """single gradient step"""

    x = x + step * direction
    print(x) # See way algorithm takes.

    return x


def lazy_wolfe_step(f, x, gradient_f, current_gradient, direction, alphabar, betabar, max_iter):
    """ Modified Powell-Wolfe step algorithm. """

    count = 0
    step_plus = 1

    # Choosing step_plus such that powell_wolfe_2condition holds,
    # df(x + step_plus * direction) * direction >= betabar * df(x) * direction:
    while powell_wolfe_2condition(step_plus, gradient_f, x, current_gradient, direction, betabar) == False:
        count += 1
        step_plus = 3 * step_plus
        #print(step_plus)
        if count > max_iter:
            break

    step_minus = step_plus

    # Choosing step_minus such that armijo_step_condition holds,
    # df(x + step_minus * direction) <= f(x) + alphabar * step_minus * df(x) * direction:
    while armijo_step_condition(step_minus, f, x, current_gradient, direction, alphabar) == False:
        count +=1
        step_minus = (1/3) * step_minus
        print(step_minus)
        if count > max_iter:
            break

    # Checking if by chance step_minus also fulfils powell_wolfe_2condition:
    if powell_wolfe_2condition(step_minus, gradient_f, x, current_gradient, direction, betabar) == True:

        return step_minus

    else: # Finding a big step that fulfils armijo_step_condition
        step = (step_minus + step_plus) / 2
        while armijo_step_condition(step, f, x, current_gradient, direction, alphabar) == False:
            count += 1
            step = (step_minus + step) / 2
            if count > max_iter:
                break

        return step


def armijo_step_condition(step, f, x, current_gradient, direction, alphabar):

    # Checking for right datatype:
    if current_gradient.dtype != "float64":
        current_gradient = np.array(current_gradient , dtype=np.float64)
    if direction.dtype != "float64":
        direction = np.array(direction , dtype=np.float64)

    # Armijo condition:
    print(len(f(x + step * direction)))
    if f(x + step * direction) <= f(x) + alphabar * step * np.dot(current_gradient, direction.T):
        return True

    else:
        return False

def powell_wolfe_2condition(step, gradient_f, x, current_gradient, direction, betabar):
    gradient_plus = gradient_f(x + step * direction)

    # Checking for right datatype:
    if current_gradient.dtype != "float64":
        current_gradient = np.array(current_gradient, dtype=np.float64)
    if direction.dtype != "float64":
        direction = np.array(direction, dtype=np.float64)
    if current_gradient.dtype != "float64":
        gradient_plus = np.array(gradient_plus, dtype=np.float64)

    # Powell_wolfe_2condition:
    if (np.dot(gradient_plus, direction) >= betabar * np.dot(current_gradient, direction)):
        return True

    else:
        return False



