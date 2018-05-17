######
######  This file includes different optimization methods
######

import time
import numpy as np
import math


def bisection( one_d_fun, MIN, MAX, eps=1e-8, maximum_iterations=65536 ):

  # counting the number of iterations
  iterations = 0

  if eps <= 0:
      raise ValueError("Epsilon must be positive")

  while True:

    MID = ( MAX + MIN ) / 2

    # Oracle access to the function value and derivative
    value, derivative = one_d_fun( MID, 1 )

    # f(MID)-f(x^*) <= |f'(MID)*(MID-x^*)| <= |f'(MID)|*(MAX-MID)
    suboptimality = abs(derivative) * (MAX - MID)

    if suboptimality <= eps:
        break

    if derivative > 0:
        MAX=MID
    else:
        MIN=MID

    iterations += 1
    if iterations>=maximum_iterations:
        break

  return MID


def function_on_line(func, x, direction, order):
    if order==0:
        value = func(x, order)
        return value
    elif order==1:
        value, gradient = func(x, order)
        return (value, gradient.T*direction)
    elif order==2:
        value, gradient, hessian = func(x, order)
        return (value, gradient.T*direction, direction.T*hessian*direction)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


###############################################################################
def exact_line_search( func, x, direction, eps=1e-9, maximum_iterations=65536 ):
    """ 
    'Exact' linesearch (using bisection method)
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """
    
    x = np.matrix( x )
    direction = np.matrix( direction )
    
    value_0 = func( x , 0 )
    
    # setting an upper bound on the optimum.
    MIN_eta = 0
    MAX_eta = 1
    iterations = 0
    
    value = func( x + MAX_eta * direction, 0 )
    value = np.double( value )
    
    while value<value_0 :
        
        MAX_eta *= 2
        
        value = func( x + MAX_eta * direction, 0 )
        
        iterations += 1
        
        if iterations >= maximum_iterations/2:
            break
    
    #construct new function equal to f on the line
    func_on_line = lambda eta, order: function_on_line( func, x + eta * direction, direction, order )

    # bisection search in the interval (MIN_t, MAX_t)    
    return bisection(func_on_line, MIN_eta, MAX_eta, eps, maximum_iterations/2)
    

###############################################################################
def backtracking_line_search( func, x, direction, alpha=0.4, beta=0.9, maximum_iterations=65536 ):
    """ 
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
    maximum_iterations: the maximum allowed number of iterations
    """

    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if alpha >= 0.5:
        raise ValueError("Alpha must be less than 0.5")
    if beta <= 0:
        raise ValueError("Beta must be positive")
    if beta >= 1:
        raise ValueError("Beta must be less than 1")
        
    x = np.matrix( x )
    value_0, gradient_0 = func(x, 1)
    value_0 = np.double( value_0 )
    gradient_0 = np.matrix( gradient_0 )

    t = 1
    iterations = 0
    while True:        
    
        if func(x+t*direction, 0)<value_0 + alpha * t * gradient_0.T*direction:
            break

        t=t*beta
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
    
    return t


###############################################################################
def gradient_descent( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=exact_line_search, *linesearch_args ):
    """ 
    Gradient Descent
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point, should be a float
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """
    
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    
    # gradient updates
    while True:
        
        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )
    
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )
        
        direction = -gradient

        if np.vdot(direction,direction)<=eps:
            break            
        
        t = linesearch( func, x, direction, *linesearch_args )
        
        x = x + t * direction
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
                
    return (x, values, runtimes, xs)
    
    


###############################################################################
def newton( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=exact_line_search, *linesearch_args  ):
    """ 
    Newton's Method
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """
    
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix( initial_x )
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    
    # newton's method updates
    while True:
        
        value, gradient, hessian = func( x , 2 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        hessian = np.matrix( hessian )
        
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        direction = -np.linalg.inv(hessian) * gradient
                    
        if -direction.T*gradient <= eps:
            break
        
        t = linesearch( func, x, direction )

        x = x + t * direction
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
    
    return (x, values, runtimes, xs)

###############################################################################
def bfgs( func, initial_x, initial_inv_h, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """ 
    BFGS Algorithm
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    initial_inv_h:      the initialization for the inverse hessian
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix( initial_x )

    if np.isscalar( initial_inv_h ):
        inv_h = initial_inv_h
    else:
        inv_h = np.asmatrix( initial_inv_h.copy() )
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    m = len( x )
    iterations = 0
    old_x = np.zeros( x.shape )
    old_gradient = np.zeros( x.shape )
    direction = np.zeros( x.shape )
    
    # BFGS gradient updates
    while True:
        
        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # termination criterion
        if np.vdot( gradient, gradient ) <= eps:
            break

        # BFGS: estimating the hessian
        if iterations > 0:
            s = x - old_x
            y = gradient - old_gradient
            if y.T*s>1e-9:
                tau = np.double(1.0 / ( y.T*s ))
                inv_h = ( np.eye(m) - tau *(s*y.T) ) * inv_h * ( np.eye(m) - tau * (y* s.T) ) + np.asmatrix( tau * np.outer( s, s) )
        old_x = x.copy()
        old_gradient = gradient.copy()
        
        # direction of update
        gradient = np.matrix( gradient )
        direction = - inv_h * gradient
                    
        t = linesearch( func, x, direction )

        x = x + t * direction
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
    
    return (x, values, runtimes, xs)
    
###############################################################################
def objective_log_barrier( f, phi, x, t, order=0 ):
    """ 
    Log-barrier Objective
    f:          the objective function
    phi:        the log-barrier constraint function
    x:          the current iterate
    t:          the scale of the log barrier
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the Hessian
    """

    if order == 0:
        f_value = f(x, 0)
        phi_value = phi(x, 0)
        value = t * f_value + phi_value
        return value
    
    elif order == 1:
        f_value, f_gradient = f(x, 1)
        phi_value, phi_gradient = phi(x, 1)
        value = t * f_value + phi_value
        gradient = t * f_gradient + phi_gradient
        return (value, gradient)
        
    elif order == 2:
        f_value, f_gradient, f_hessian = f(x, 2)
        phi_value, phi_gradient, phi_hessian = phi(x, 2)
        value = t * f_value + phi_value
        gradient = t * f_gradient + phi_gradient
        hessian = t * f_hessian + phi_hessian
        return (value, gradient, hessian)
        
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
        
###############################################################################
def objective_scalar_constraints( constraints, x, order=0 ):
    """ 
    Log barrier constraint function
    constraints:    a list of handles to the constraint functions
    x:              the current iterate
    order:          the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """

    if len(constraints) < 1:
        raise ValueError("Constraint cannot be empty")
        
    values = []
    gradients = []
    hessians = []    
    
    # Fill in values, gradients, and hessians with the values, gradients and hessians of each constraint
    for constraint in constraints:
    
        if order == 0:
            constraint_value = constraint(x, 0)
        
        elif order == 1:    
            constraint_value, constraint_gradient = constraint(x, 1)
            
        elif order == 2:
 
            constraint_value, constraint_gradient, constraint_hessian = constraint(x, 2)
            
        else:
            raise ValueError("The argument \"order\" should be 0, 1 or 2")

        value = - np.log( np.maximum( constraint_value, 0 ) )
        
        values.append(value)
        
        if order >= 1:
            gradient = -constraint_gradient/constraint_value
            gradients.append(gradient)

            if order == 2:
                hessian = -constraint_hessian/constraint_value+constraint_gradient*constraint_gradient.T/(constraint_value*constraint_value)
                hessians.append(hessian)
            
    # sum the values, gradients and hessians for all constraints
    value = np.sum( values )
    if order == 0:
        return value
    elif order == 1:
        gradient = 0
        for g in gradients:
            gradient += g
        return (value, gradient)
    elif order == 2:
        gradient = 0
        for g in gradients:
            gradient += g
        hessian = 0
        for h in hessians:
            hessian += h
        return (value, gradient, hessian)
    
###############################################################################
def log_barrier( func, constraints, initial_x, initial_t, mu, m, newton_eps=1e-5, log_barrier_eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """ 
    Log-barrier Method
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    initial_t:          the starting log-barrier scale
    mu:                 the update parameter for t
    m:                  the dimension of the constraints
    newton_eps:         the Newton stopping threshold
    log_barrier_eps:    the log-barrier stopping threshold
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """
    
    if newton_eps <= 0:
        raise ValueError("Newton epsilon must be positive")
    if log_barrier_eps <= 0:
        raise ValueError("Log barier epsilon must be positive")
    if mu <= 1:
        raise ValueError("Mu must be greater than one")
        
    phi = lambda x, order: objective_scalar_constraints( constraints, x, order )
    
    newton_iterations = []
    
    x = np.asarray( initial_x.copy() )
    t = initial_t
    iterations = 0
    while True:
        newton_f = lambda x, order: objective_log_barrier( func, phi, x, t, order)
        x, newton_values, runtimes, xs = newton( newton_f, x, newton_eps, maximum_iterations, linesearch )
        newton_iterations.append( len( newton_values ) )
       
        t = t*mu
        
        if m/t<=log_barrier_eps: break; end

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")
    
    return (x, newton_iterations)


def subgradient_descent( func, initial_x, maximum_iterations=65536, initial_stepsize=1):
    """ 
    Subgradient Descent
    func:               the function to optimize. It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point, should be a float
    maximum_iterations: the maximum allowed number of iterations
    initial_stepsize:   the initial stepsize, should be a float
    """
    
    x = np.matrix(initial_x)
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    
    # subgradient updates
    while True:
        
        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )
    
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )
        
        # x = ( TODO: update of subgradient descent )
           
        iterations += 1
        x = x - (initial_stepsize / np.sqrt(iterations)) * gradient  
        if iterations >= maximum_iterations:
            break
                
    return (x, values, runtimes, xs)


def adagrad( func, initial_x, maximum_iterations=65536, initial_stepsize=1, initial_sum_of_squares=1e-3):
    """ 
    adagrad
    func:                   the function to optimize. It is called as "value, gradient = func( x, 1 )
    initial_x:              the starting point, should be a float
    maximum_iterations:     the maximum allowed number of iterations
    initial_stepsize:       the initial stepsize, should be a float
    initial_sum_of_squares: initial sum of squares
    """
    
    x = np.matrix(initial_x)
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    gradient_sum = 0

    # subgradient updates
    while True:
        
        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        # updating the logs

        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )
        
        # x = ( TODO: update of adagrad )
        if iterations == 0:
            gradient_sum = np.square(gradient)
        else:
            gradient_sum += np.square(gradient)

        eta = initial_stepsize / np.sqrt(initial_sum_of_squares + gradient_sum)
        x = x - np.multiply(eta, gradient)
        iterations += 1

        if iterations >= maximum_iterations:
            break
                
    return (x, values, runtimes, xs)
