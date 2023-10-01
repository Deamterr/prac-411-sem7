from functions import function
from functions import inequality_constraints
from functions import equality_constraints


def gradient(func, x: list, eps=1e-9) -> list:
    dim_num = len(x)
    grad = [0] * dim_num
    for i in range(dim_num):
        new_x = list(x)
        new_x[i] += eps
        grad[i] = (func(new_x) - func(x)) / eps
    return grad


def golden_section_search(func, x_min: float, x_max: float, eps: float = 1e-9) -> float:
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

        if (a + b) / 2 < 0:
            a, b = a_prev, b_prev
            break
    assert (a + b) / 2 >= 0, 'negative step_size!'
    return (a + b) / 2


class SteepestDescent:
    def __init__(self, func, starting_point: list, eps: float = 1e-9, max_iter: int = 1000):
        self.func = func
        self.starting_point = starting_point
        self.eps = eps
        self.max_iter = max_iter
        self.dim_num = len(starting_point)
        self.x = [[0] * len(starting_point) for i in range(max_iter)]
        self.x[0] = starting_point
        self.step = 0

    def opt_step_size_function(self, step_size: float) -> float:
        new_x = [0] * self.dim_num
        for i in range(self.dim_num):
            new_x[i] = self.x[self.step][i] - step_size * gradient(self.func, self.x[self.step])[i]
        return self.func(new_x)

    def launch(self):
        while True:
            step_size = golden_section_search(self.opt_step_size_function, x_min=0, x_max=1000)
            for i in range(self.dim_num):
                self.x[self.step + 1][i] = \
                    self.x[self.step][i] - step_size * gradient(self.func, self.x[self.step])[i]
            self.step += 1
            if self.func(self.x[self.step]) - self.func(self.x[self.step]) <= self.eps:
                break
            assert self.step + 1 < self.max_iter, 'Steepest descent does not converge!'
        return self.x[self.step]


class PenaltyMethod:
    def __init__(self, starting_point, termination_scalar, penalty_param=1, scalar=1.25, max_iter=100):
        self.starting_point = starting_point
        self.termination_scalar = termination_scalar
        self.function = function
        self.inequality_constraints = inequality_constraints
        self.equality_constraints = equality_constraints
        self.penalty_param = penalty_param
        self.scalar = scalar
        self.dim_num = len(starting_point)
        self.x = [[0] * len(starting_point) for i in range(max_iter)]
        self.x[0] = starting_point
        self.step = 0
        self.max_iter = max_iter

    def penalty_function(self, x: list) -> float:
        ineq_num = len(self.inequality_constraints(x))
        eq_num = len(self.equality_constraints(x))
        penalty_func = 0
        for i in range(ineq_num):
            penalty_func += (max(self.inequality_constraints(x)[i], 0))**2
        for i in range(eq_num):
            penalty_func += (self.equality_constraints(x)[i])**2
        return penalty_func

    def minimize_function(self, x) -> float:
        return self.function(x) + self.penalty_param * self.penalty_function(x)

    def launch(self):
        while True:
            descent = SteepestDescent(func=self.minimize_function,
                                      starting_point=self.x[self.step])
            self.x[self.step + 1] = descent.launch()
            self.step += 1
            #print('step=', self.x[self.step], self.penalty_param, (self.x[self.step][1] - self.x[self.step][0])**2)
            if self.penalty_function(self.x[self.step]) < self.termination_scalar:
                break
            else:
                self.penalty_param *= self.scalar
            assert self.step + 1 < self.max_iter, 'Penalty method does not converge!'
        return self.x[self.step], self.function(self.x[self.step]), self.step
