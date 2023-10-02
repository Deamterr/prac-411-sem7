from functions import function
from functions import inequality_constraints
from functions import equality_constraints


def gradient(func, x: list, eps: float = 1e-12) -> list:
    """
    Calculates gradient of func at point x with accuracy eps.
    :param func: function with one parameter
    :param x: point
    :param eps: accuracy
    :return: gradient of a func at point x
    """
    dim_num = len(x)
    grad = [0] * dim_num
    for i in range(dim_num):
        new_x = list(x)
        new_x[i] += eps
        grad[i] = (func(new_x) - func(x)) / eps
    return grad


def golden_section_search(func, x_min: float, x_max: float, eps: float = 1e-12) -> float:
    """
    Golden-section search for finding argmin of func inside interval [x_min, x_max]
    with accuracy eps.
    :param func: function with one parameter
    :param x_min: left boundary of the interval
    :param x_max: right boundary of the interval
    :param eps: accuracy
    :return: argmin of the func
    """
    a, b = x_min, x_max
    phi = (1 + 5 ** 0.5) / 2
    x_1 = b - (b - a) / phi
    x_2 = a + (b - a) / phi
    while (b - a) / 2 >= eps:
        a_prev, b_prev = a, b
        if func(x_1) > func(x_2):
            a = x_1
            x_1 = x_2
            x_2 = b - (x_1 - a)
        else:
            b = x_2
            x_2 = x_1
            x_1 = a + (b - x_2)

        if not x_min <= (a + b) / 2 <= x_max:
            a, b = a_prev, b_prev
            break
    assert x_min <= (a + b) / 2 <= x_max, 'golden_section out of bounds!'
    return (a + b) / 2


class SteepestDescent:
    def __init__(self, func, starting_point: list, eps: float = 1e-12, max_iter: int = 1000):
        self.func = func
        self.starting_point = starting_point
        self.eps = eps
        self.max_iter = max_iter
        self.dim_num = len(starting_point)
        self.x = [[0] * len(starting_point) for _ in range(max_iter)]
        self.x[0] = starting_point
        self.step = 0

    def opt_step_size_function(self, step_size: float) -> float:
        """
        Function for which the optimal step needs to be found.
        :param step_size:
        :return:
        """
        new_x = [0] * self.dim_num
        for i in range(self.dim_num):
            new_x[i] = self.x[self.step][i] - step_size * gradient(self.func, self.x[self.step])[i]
        return self.func(new_x)

    def launch(self) -> list:
        """
        Launching the steepest descent method.
        :return: argmin of a function that needs to be optimized
        """
        while True:
            step_size = golden_section_search(self.opt_step_size_function, x_min=0, x_max=100)
            for i in range(self.dim_num):
                self.x[self.step + 1][i] = \
                    self.x[self.step][i] - step_size * gradient(self.func, self.x[self.step])[i]
            self.step += 1
            if self.func(self.x[self.step]) - self.func(self.x[self.step]) <= self.eps:
                break
            if self.step + 1 < self.max_iter:
                assert self.step + 1 < self.max_iter, 'Penalty method does not converge!'
        return self.x[self.step]


class PenaltyMethod:
    def __init__(self, starting_point: list, termination_scalar: float,
                 penalty_param: float = 1, scalar: float = 1.25, max_iter: int = 100):
        self.starting_point = starting_point
        self.termination_scalar = termination_scalar
        self.function = function
        self.inequality_constraints = inequality_constraints
        self.equality_constraints = equality_constraints
        self.penalty_param = penalty_param
        self.scalar = scalar
        self.dim_num = len(starting_point)
        self.x = [[0] * len(starting_point) for _ in range(max_iter)]
        self.x[0] = starting_point
        self.step = 0
        self.max_iter = max_iter

    def penalty_function(self, x: list) -> float:
        """
        Penalty function of penalty method.
        :param x: point
        :return: value of penalty function at x
        """
        ineq_num = len(self.inequality_constraints(x))
        eq_num = len(self.equality_constraints(x))
        penalty_func = 0
        for i in range(ineq_num):
            penalty_func += (max(self.inequality_constraints(x)[i], 0))**2
        for i in range(eq_num):
            penalty_func += (self.equality_constraints(x)[i])**2
        return penalty_func

    def minimize_function(self, x: list) -> float:
        """
        Final function that needs to be minimized using the steepest descent method.
        :param x: point
        :return: value of the function at x
        """
        return self.function(x) + self.penalty_param * self.penalty_function(x)

    def launch(self) -> (list, float, int):
        """
        Launching penalty method.
        :return: point of local optimum, local optimum value, number of iterations
        """
        while True:
            descent = SteepestDescent(func=self.minimize_function,
                                      starting_point=self.x[self.step])
            self.x[self.step + 1] = descent.launch()
            self.step += 1
            if self.penalty_function(self.x[self.step]) < self.termination_scalar:
                break
            else:
                self.penalty_param *= self.scalar
            assert self.step + 1 < self.max_iter, 'Penalty method does not converge!'
        return self.x[self.step], self.function(self.x[self.step]), self.step
