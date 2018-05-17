######
######  This file includes different optimization methods
######

import time
import numpy as np


def bisection( one_d_fun, MIN, MAX, eps=1e-5, maximum_iterations=65536 ):

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
    eps:                the maximum allowed error in the resulting stepsize t
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

        if np.linalg.norm(direction)<eps:
            break            
        
        t = linesearch( func, x, direction, *linesearch_args )
        
        x = x + t * direction
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
                
    return (x, values, runtimes, xs)
    
    


###############################################################################
def newton( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
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
                    
        if -direction.T*gradient < eps:
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
        inv_h = initial_inv_h *  np.identity(x.size)
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
    
    # BFGS gradient updatesa
    inv_h = initial_inv_h
    while True:
        
        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # TODO: code the entire bfgs algorithm
        if np.linalg.norm(gradient)**2 <= eps:
            break

        p_k = -inv_h*gradient
        alpha_k = linesearch(func, x, p_k)
        old_x = x
        x = old_x + alpha_k * p_k

        _, gradient_k1 = func( x , 1 )
        gradient_k1 = np.matrix( gradient_k1 )
        
        s_k = x - old_x
        y_k = gradient_k1 - gradient
        
        condition = np.asscalar( y_k.T * s_k )

        if condition >= 1e-9: 
            rho_k = 1 / condition
            inv_h = (np.identity(s_k.shape[0]) - rho_k * s_k * y_k.T) * inv_h * (np.identity(y_k.shape[0]) - rho_k * y_k * s_k.T) + rho_k * s_k * s_k.T
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
    
    return (x, values, runtimes, xs)
