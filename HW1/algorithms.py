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

    # if (TODO: TERMINATION CRITERION): break
    if derivative == 0:
        break;
    # if derivative... (TODO: LINE SEARCH)
    if derivative > 0:
        MAX = MID
    else:
        MIN = MID
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
    
    # look for a stepsize that gives a function value greater than in the current point
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
    
        # if (TODO: TERMINATION CRITERION): break
        lvalue = func( x + t * direction, 0 )
        rvalue = func( x , 0) + alpha * t * gradient_0.T * direction
        if lvalue <= rvalue:
            break;
        # t = TODO: BACKTRACKING LINE SEARCH
        else:
            t = beta * t
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
        
        # direction= (TODO)
        direction = -gradient

        # if (TODO: TERMINATION CRITERION): break
        if np.linalg.norm(gradient, ord = 2) ** 2 <= eps:
            break

        t = linesearch( func, x, direction, *linesearch_args )
        
        # x= (TODO: UPDATE x)
        #print(value)
        #print(gradient)
        x = x + t * -gradient
        #print('X', x)
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

        # direction = (TODO)
        direction = - (hessian.I * gradient)

        # if (TODO: TERMINATION CRITERION): break
        deltaX = gradient.T * hessian.I * gradient
        if deltaX**2 <= eps:
            break
        t = linesearch( func, x, direction )

        # x = (TODO: UPDATE x)
        x = x + t * ( -1 * hessian.I * gradient)
        iterations += 1
        if iterations >= maximum_iterations:
            break
    
    return (x, values, runtimes, xs)
