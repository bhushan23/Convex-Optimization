######
######  This file includes different functions used in HW3
######

import time
import numpy as np
import algorithms as alg
import matplotlib.pyplot as plt


###############################################################################
def main_quadratic():
    np.seterr(divide='ignore')
    # the hessian matrix of the quadratic function
    Q = np.matrix('5 2; 2 2')
    
    # the vector of linear coefficient of the quadratic function
    v = np.matrix('21; 13')
    
    A = np.matrix('2 1; 2 -5;-1 1')
    
    b = np.matrix('8; 10; 3')
    
    t = len( b )
    
    func = lambda x, order: quadratic_log_barrier( Q, v, A, b, x, t, order )
    
    
    # Start at (0,0)
    initial_x = np.matrix('0.0; 0.0')
    # Find the (1e-8)-suboptimal solution
    eps = 1e-8
    maximum_iterations = 65536
    
    # Run the algorithms
    x, values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.backtracking )
    
    x, values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.backtracking )
    
    # Draw contour plots
    draw_contour( func, gd_xs, newton_xs, levels=np.arange(-300, 300, 10), x=np.arange(-4, 2.1, 0.1), y=np.arange(-4, 2.1, 0.1) )
    input("Press Enter to continue...")




###############################################################################
def main_quadratic2():
    np.seterr(divide='ignore')

    # the hessian matrix of the quadratic function
    Q = np.matrix('5 2; 2 2')
    
    # the vector of linear coefficient of the quadratic function
    v = np.matrix('21; 13')
    
    A = np.matrix('2 1; 2 -5;-1 1')
    
    b = np.matrix('8; 10; 3')
    
    mu = np.sqrt( 2 )
    
    lower_t = 0.1249
    
    upper_t = 60
    
    
    # Start at (0,0)
    initial_x = np.matrix('0.0; 0.0')
    # Find the (1e-8)-suboptimal solution
    eps = 1e-8
    maximum_iterations = 65536
    
    ts = []
    gd_iterations = []
    newton_iterations = []
    
    t = lower_t
    while t <= upper_t:
        func = lambda x, order: quadratic_log_barrier( Q, v, A, b, x, t, order )
        ts.append( t )
        
        # Run the algorithms
        x, values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.backtracking )
        gd_iterations.append(len(values)-1)
        x, values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.backtracking )
        newton_iterations.append(len(values)-1) 
        
        t *= mu
    
    plt.plot( ts, gd_iterations, linewidth=2, color='r', label='GD', marker='o' )
    plt.plot( ts, newton_iterations, linewidth=2, color='m', label='Newton', marker='o')
    plt.xlabel('t')
    plt.ylabel('iterations')
    plt.legend()
    plt.show()



    
###############################################################################
def main_linear():
    np.seterr(divide='ignore')

    Q = np.zeros((5,5))
    v = np.matrix('170; 160; 175; 180; 195')
    A = np.matrix('1, 0, 0, 0, 0 ; 1, 1, 0, 0, 0 ; 1, 1, 0, 0, 0 ; 1, 1, 1, 0, 0 ; 0, 1, 1, 0, 0 ; 0, 0, 1, 1, 0 ; 0, 0, 1, 1, 0 ; 0, 0, 0, 1, 0 ; 0, 0, 0, 1, 1, ; 0, 0, 0, 0, 1 ; 1, 0, 0, 0, 0 ; 0, 1, 0, 0, 0 ; 0, 0, 1, 0, 0 ; 0, 0, 0, 1, 0 ; 0, 0, 0, 0, 1')
    b = np.matrix('-48; -79; -65; -87; -64; -73; -82; -43; -52; -15; -0; -0; -0; -0; -0')
    #A = np.matrix('1, 0, 0, 0, 0 ; 1, 1, 0, 0, 0 ; 1, 1, 1, 0, 0 ; 0, 1, 1, 0, 0 ; 0, 0, 1, 1, 0 ; 0, 0, 0, 1, 0 ; 0, 0, 0, 1, 1, ; 0, 0, 0, 0, 1 ; 0, 1, 0, 0, 0 ; 0, 0, 1, 0, 0')
    #b = np.matrix('-48; -79; -87; -64; -82; -43; -52; -15; -0; -0')
    
    constraints = []
    constraints.append( lambda x, order: quadratic(Q, A[0,:].T, b[0], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[1,:].T, b[1], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[2,:].T, b[2], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[3,:].T, b[3], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[4,:].T, b[4], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[5,:].T, b[5], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[6,:].T, b[6], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[7,:].T, b[7], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[8,:].T, b[8], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[9,:].T, b[9], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[10,:].T, b[10], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[11,:].T, b[11], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[12,:].T, b[12], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[13,:].T, b[13], x, order ) )
    constraints.append( lambda x, order: quadratic(Q, A[14,:].T, b[14], x, order ) )
        
    initial_x = np.matrix('100.0; 100.0; 100.0; 100.0; 100.0')
    newton_eps = 1e-8
    eps = 1e-4
    maximum_iterations = 65536
    
    m = len( b );
    lower_mu = 1 + 1 / np.sqrt( m );
    upper_mu = m / eps;
    delta_mu = 2;
    initial_t = 1;
    
    f = lambda x, order: quadratic( Q, v, 0, x, order );
    
    mus = []
    iterations =[]
    mu = lower_mu
    
    while  mu <= upper_mu:
        mus.append(mu)
        x, newton_iterations = alg.log_barrier( f, constraints, initial_x, initial_t, mu, m, newton_eps, eps, maximum_iterations, alg.backtracking )
        iterations.append( np.sum( newton_iterations ) )
        mu *= delta_mu
     
    print(x)
    print(v.T * x)
    
    plt.semilogx( mus, iterations, linewidth=2, color='r', marker='o')
    plt.xlabel(r'$\mu$')
    plt.ylabel('Total Newton Iterations')
    plt.show()
        



###############################################################################
def quadratic_log_barrier( Q, v, A, b, x, t, order=0 ):

    """ 
    Quadratic objective 0.5 * x' * Q * x + v' * x with linear log-barrier constraints A * x + b >= 0
    f:          the objective function
    Q:          the symmetric matrix of the quadratic form
    v:          the vector of linear coefficients
    A:          the constraint matrix
    b:          the constraint vector    
    x:          the current iterate
    t:          the scale of the log barrier
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """

    # we need to make sure that we get infinity (not a complex number) if x violates the constraints
    safe_log = np.log( np.maximum( A * x + b, 0) )
    value = t * ( 0.5 * x.T * Q * x + v.T * x ) - safe_log.sum()

    if order == 0:
        return value
    elif order == 1:
        # gradient = ( TODO: gradient )
        gradient = t * (Q * x + v) - A.T * ( 1 / (A*x + b))

        return (value, gradient)
        
    elif order == 2:
        # gradient = ( TODO: gradient )
        gradient =  t * (Q * x + v) - A.T * ( 1 / (A*x + b))
        # hessian = ( TODO: hessian )
        A1 = A / np.square(A * x + b)
        hessian = t * Q + A1.T * A   
        return (value, gradient, hessian)
        
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
            
            
            
        
###############################################################################
def quadratic( Q, v, c, x, order=0 ):
    """ 
    Quadratic Objective
    Q:          the symmetric matrix of the quadratic form
    v:          the vector of linear coefficients
    c:          the constant
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    Q = np.asmatrix(Q)
    v = np.asmatrix(v)
    x = np.asmatrix(x)
    
    value = 0.5 * x.T * Q * x + v.T * x + c

    if order == 0:
        return value
    
    elif order == 1:
        
        gradient = Q * x + v
        
        return (value, gradient)
        
    elif order == 2:
        
        gradient = Q * x + v

        hessian = Q

        return (value, gradient, hessian)
        
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")




###############################################################################
def draw_contour( func, gd_xs, newton_xs, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05) ):
    """ 
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    newton_xs:  Newton's method iterates
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )
    
    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()
    
    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )
    
    L = plt.legend(handles=[line_gd, line_newton])
    plt.draw()
    time.sleep(1)
    
    for i in range( 1, max(len(gd_xs), len(newton_xs) ) ):
        
        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )

        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )
         
        
        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )    
        
        plt.draw()
        if i < 10:
            input("Press Enter to continue...")


#main_quadratic()
#main_quadratic2()
#main_linear()
