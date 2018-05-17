######
######  Main script that compares different optimization methods 
######

import numpy as np
import hw1_functions as hw1_func
import algorithms as alg


# the hessian matrix of the quadratic function
H = np.matrix('1 0; 0 30')

# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')

# Choose the quadratic objective. This notation defines a "closure", returning
# an oracle function which takes x (and order) as its only parameter, and calls
# obj.quadratic with parameters H and b defined above, and the parameters of 
# the closure (x and order)
func = lambda x, order: hw1_func.quadratic( H, b, x, order )


# Find the (1e-4)-suboptimal solution
eps = 1e-4
maximum_iterations = 65536


# Run the algorithms on the weird function
x = alg.bisection( hw1_func.weird_func, -100, 100, eps, maximum_iterations)
print('Optimum of the weird function found by bisection', x)

x, values, runtimes, gd_xs = alg.gradient_descent( hw1_func.weird_func, 0.0, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the weird function found by GD with exact line search', x)

x, values, runtimes, gd_xs = alg.newton( hw1_func.weird_func, 0.0, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the weird function found by Newton with exact line search', x)

x, values, runtimes, gd_xs = alg.gradient_descent( hw1_func.weird_func, 0.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by GD with backtracking line search', x)

x, values, runtimes, gd_xs = alg.newton( hw1_func.weird_func, 0.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by Newton with backtracking line search', x)


# Run algorithms on the quadratic function
# Start at (4,0.3)
initial_x = np.matrix('4.0; 0.3')

x, values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the quadratic function found by GD with exact line search', x.T)

x, values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the quadratic function found by Newton with exact line search', x.T)

x, values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the quadratic function found by GD with backtracking line search', x.T)

x, values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the quadratic function found by Newton with backtracking line search', x.T)

# Draw contour plots
hw1_func.draw_contour( func, gd_xs, newton_xs, 0, levels=np.arange(5, 400, 20), x=np.arange(-5, 5.1, 0.1), y=np.arange(-5, 5.1, 0.1))


# Run Netwon on the quadratic function
# Start at (-1.0,-1.0)
eps = 1e-5
initial_x = np.matrix('-1.0; 1.0')
maximum_iterations = 100

x, values, runtimes, gd_xs = alg.gradient_descent( hw1_func.boyd_example_func, initial_x, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the Boyd\'s function found by GD with exact line search', x.T)

x, values, runtimes, newton_xs = alg.newton( hw1_func.boyd_example_func, initial_x, eps, maximum_iterations, alg.exact_line_search )
print('Optimum of the Boyd\'s function found by Newton with exact line search', x.T)

x, values, runtimes, gd_xs = alg.gradient_descent( hw1_func.boyd_example_func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the Boyd\'s function found by GD with backtracking line search', x.T)

x, values, runtimes, newton_xs = alg.newton( hw1_func.boyd_example_func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the Boyd\'s function found by Newton with backtracking line search', x.T)

# Draw contour plots
hw1_func.draw_contour( hw1_func.boyd_example_func, gd_xs, newton_xs, 1, levels=np.arange(0, 15, 1), x=np.arange(-2, 2, 0.1), y=np.arange(-2, 2, 0.1))
