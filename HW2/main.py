######
######  Main script that compares different optimization methods 
######

import numpy as np
import hw2_functions as hw_func
import algorithms as alg
import matplotlib.pyplot as plt


# the hessian matrix of the quadratic function
H = np.matrix('1 0; 0 30')

# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')

# Choose the quadratic objective. This notation defines a "closure", returning
# an oracle function which takes x (and order) as its only parameter, and calls
# obj.quadratic with parameters H and b defined above, and the parameters of 
# the closure (x and order)
func = lambda x, order: hw_func.quadratic( H, b, x, order )


# Run algorithms on the weird function
eps = 1e-4
maximum_iterations = 65536

x = alg.bisection( hw_func.weird_func, -100, 100, eps, maximum_iterations)
print('Optimum of the weird function found by bisection', x)

x, values, runtimes, gd_xs = alg.gradient_descent( hw_func.weird_func, 0.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by GD with backtracking line search', x)

x, values, runtimes, gd_xs = alg.newton( hw_func.weird_func, 0.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by Newton with backtracking line search', x)

x, values, runtimes, gd_xs = alg.bfgs( hw_func.weird_func, 0.0, 1.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by BFGS with backtracking line search', x)


# Run algorithms on the quadratic function
eps = 1e-10
initial_x = np.matrix('4.0; 0.3')
maximum_iterations=20

x, gd_back_values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the quadratic function found by GD with backtracking line search', x.T)

x, newton_back_values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the quadratic function found by Newton with backtracking line search', x.T)

x, bfgs_inexact_values, runtimes, bfgs_xs = alg.bfgs( func, initial_x, 1.0, eps, maximum_iterations, alg.backtracking_line_search )
print('Optimum of the weird function found by BFGS with backtracking line search', x.T)


# Draw contour plots
#hw_func.draw_contour( func, gd_xs, newton_xs, 0, levels=np.arange(5, 400, 20), x=np.arange(-5, 5.1, 0.1), y=np.arange(-5, 5.1, 0.1))


eps = 1e-30
initial_x = np.matrix('1.2; 1.2')
maximum_iterations = 20

x, gd_back_values, runtimes, gd_xs = alg.gradient_descent(hw_func.rosenbrock_func, initial_x, eps, maximum_iterations, alg.backtracking_line_search)
print('Optimum of the Rosenbrock function found by GD with backtracking line search', x.T)

x, newton_back_values, runtimes, newton_xs = alg.newton(hw_func.rosenbrock_func, initial_x, eps, maximum_iterations, alg.backtracking_line_search)
print('Optimum of the Rosenbrock function found by Newton with backtracking line search', x.T)

x, bfgs_back_values, runtimes, bfgs_xs = alg.bfgs( hw_func.rosenbrock_func, initial_x, 1.0, eps, maximum_iterations, alg.backtracking_line_search)
print('Optimum of the Rosenbrock function found by BFGS with backtracking line search', x.T)

plt.figure(1)
line_gd, = plt.semilogy([x - hw_func.rosenbrock_func([1.0, 1.0]) for x in gd_back_values], linewidth=2, color='r', marker='o', label='GD')
line_newton, = plt.semilogy([x - hw_func.rosenbrock_func([1.0, 1.0]) for x in newton_back_values], linewidth=2, color='m', marker='o',label='Newton')
line_newton, = plt.semilogy([x - hw_func.rosenbrock_func([1.0, 1.0]) for x in bfgs_back_values], linewidth=2, color='g', marker='o',label='bfgs')
plt.show()

# Draw contour plots
hw_func.draw_contour( hw_func.rosenbrock_func, gd_xs, newton_xs, 1, levels=np.arange(0, 500, 10), x=np.arange(-2, 2, 0.1), y=np.arange(-2, 2, 0.1))
