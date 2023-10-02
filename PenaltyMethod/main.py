from method import PenaltyMethod

def launch_method() -> None:
    starting_point = [float(i) for i in input('Input starting point (for example: "0, 1"):\n').split(',')]
    termination_scalar = float(input('Input termination_scalar (for example: "1e-9"):\n'))
    method = PenaltyMethod(starting_point=starting_point, termination_scalar=termination_scalar)
    optimum, optimum_value, iterations = method.launch()
    print('\nPoint of local optimum: {}\nLocal optimum value: '
          '{}\nNumber of iterations: {}\n'.format(optimum, optimum_value, iterations))

if __name__ == '__main__':
    launch_method()