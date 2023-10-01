def function(x: list) -> float:
    """
    The original function of minimization problem.
    :param x:
    :return: value of the function at x
    """
    func = (x[0] - 3)**2 + 2 * (x[1])**2
    return func

def inequality_constraints(x: list) -> list:
    """
    Inequality constraints in a following form: g_i(x) <= 0
    :param x:
    :return: list of values g_i(x) at x
    """
    ineq_num = 1
    ineq = [0] * ineq_num
    ineq[0] = (x[0] - x[1])**2 - 9
    return ineq

def equality_constraints(x: list) -> list:
    """
    Equality constraints in a following form: h_i(x) = 0
    :param x:
    :return: list of values h_i(x) at x
    """
    eq_num = 1
    eq = [0] * eq_num
    eq[0] = x[0] + x[1] - 4
    return eq