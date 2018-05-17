######
######  This file includes different functions used in HW1
######

import numpy as np
import matplotlib.pyplot as plt
import time
from math import exp

def weird_func( x, order=0 ):

  # f(x) = x^4 + 6x^2 + 12(x-4)e^(x-1)
  value = np.double(pow(x, 4) + 6 * pow(x, 2) + 12 * (x - 4) * exp(x - 1))
  
  if order==0:
      return value
  elif order==1:
      # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
      gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

      return (value, gradient)
  elif order==2:
      # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
      gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

      # f''(x)= 12 (1 + e^(-1 + x) (-2 + x) + x^2)
      hessian = 12 * (1 + (x-2) * exp(x-1) + pow(x,2))

      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


###############################################################################
def boyd_example_func(x, order=0):
  a=np.matrix('1  3')
  b=np.matrix('1  -3')
  c=np.matrix('-1  0')
  x=np.asmatrix(x).reshape(2,1)
  
  value = np.double(exp(a*x-0.1)+exp(b*x-0.1)+exp(c*x-0.1))
  if order==0:
      return value
  elif order==1:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      return (value, gradient)
  elif order==2:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      hessian = a.T*a*exp(a*x-0.1)+b.T*b*exp(b*x-0.1)+c.T*c*exp(c*x-0.1)
      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")

###############################################################################
def rosenbrock_func(x, order=0):
  x=np.asmatrix(x).reshape(2,1)
  
  value = np.double(100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1-x[0])*(1-x[0]))
  if order==0:
      return value
  elif order==1:
      gradient = np.empty([2, 1])
      gradient[0]=400*x[0]*x[0]*x[0]-400*x[0]*x[1]+2*x[0]-2
      gradient[1]=200*(x[1]-x[0]*x[0])
      return (value, gradient)
  elif order==2:
      gradient = np.empty([2, 1])
      gradient[0]=400*x[0]*x[0]*x[0]-400*x[0]*x[1]+2*x[0]-2
      gradient[1]=200*(x[1]-x[0]*x[0])
      hessian = np.empty([2, 2])
      hessian[0,0]=1200*x[0]*x[0]-400*x[1]+2
      hessian[1,0]=-400*x[0]
      hessian[0,1]=-400*x[0]
      hessian[1,1]=200
      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")
###############################################################################
def quadratic( H, b, x, order=0 ):
    """ 
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x).reshape(2,1)
    
    value = np.double(0.5 * x.T * H * x + b.T * x)

    if order == 0:
        return value
    elif order == 1:
        gradient = H * x + b
        return (value, gradient)
    elif order == 2:
        gradient = H * x + b
        hessian = H
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


###############################################################################
def draw_contour( func, gd_xs, newton_xs, fig, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05)):
    """ 
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    newton_xs:  Newton iterates
    fig:        figure index
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )
    
    plt.figure(fig)
    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()
    
    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )
    
    L = plt.legend(handles=[line_gd,line_newton])
    plt.draw()
    time.sleep(1)
    
    for i in range( 1, max(len(gd_xs), len(newton_xs)) ):
        
        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )
        
        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )

        
        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )    
        
        plt.draw()
        input("Press Enter to continue...")
