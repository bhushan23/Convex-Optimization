import numpy as np
import hw4_functions as hw_func
import algorithms as alg
import matplotlib.pyplot as plt

data = np.loadtxt('HIGGS_subset.csv', delimiter=',')

labels = np.asmatrix(2*data[:,0]-1).T
features = np.asmatrix(data[:,1:])

d = features.shape[1]

w = np.zeros((d,1))

minibatch_size=10
max_iterations_sd=20
max_iterations_sgd=int(max_iterations_sd*features.shape[0]/minibatch_size)
points_to_plot=20

func_stochastic = lambda x, order: hw_func.svm_objective_function_stochastic( x, features, labels, order, minibatch_size )
func = lambda x, order: hw_func.svm_objective_function( x, features, labels, order )

initial_x=np.zeros((d,1))

sgd_x, sgd_values, sgd_runtimes, sgd_xs = alg.subgradient_descent( func_stochastic, initial_x, max_iterations_sgd, 1)
print('Solution found by stochastic subgradient descent', sgd_x)
print('Objective function', func(sgd_x,0))
sgd_values=[func(sgd_xs[i],0) for i in range(0,max_iterations_sgd,int(max_iterations_sgd/points_to_plot))]

ada_x, ada_values, ada_runtimes, ada_xs = alg.adagrad( func_stochastic, initial_x, max_iterations_sgd, 1)
print('Solution found by stochastic adagrad', ada_x)
print('Objective function', func(ada_x,0))
ada_values=[func(ada_xs[i],0) for i in range(0,max_iterations_sgd,int(max_iterations_sgd/points_to_plot))]

sd_x, sd_values, sd_runtimes, sd_xs = alg.subgradient_descent( func, initial_x, max_iterations_sd, 1)
print('Solution found by subgradient descent', sd_x)
print('Objective function', func(sd_x,0))
sd_values=[func(sd_xs[i],0) for i in range(0,max_iterations_sd,int(max_iterations_sd/points_to_plot))]

#Obj func vs time
plt.figure(1)
line_sgd, = plt.semilogx(sgd_runtimes[0::int(max_iterations_sgd/points_to_plot)], sgd_values, linewidth=2, color='r', marker='o', label='SGD')
line_sd, = plt.semilogx(sd_runtimes[0::int(max_iterations_sd/points_to_plot)], sd_values, linewidth=2, color='g', marker='o', label='SD')
line_ada, = plt.semilogx(ada_runtimes[0::int(max_iterations_sgd/points_to_plot)], ada_values, linewidth=2, color='b', marker='o', label='AdaGrad')
plt.xlabel('seconds')
plt.ylabel('Obj Func')

# Obj func vs number of iterations
plt.figure(2)
line_sgd, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),sgd_values, linewidth=2, color='r', marker='o', label='SGD')
line_sd, = plt.semilogx(range(1,max_iterations_sd+1,int(max_iterations_sd/points_to_plot)),sd_values, linewidth=2, color='g', marker='o', label='SD')
line_ada, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_values, linewidth=2, color='b', marker='o', label='AdaGrad')
plt.xlabel('iterations')
plt.ylabel('Obj Func')
plt.show()
