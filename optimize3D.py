import numpy as np
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
from Jensen import *
import time
from scipy.interpolate import interp1d
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO

def weibull_prob(x):
	a = 1.8
	avg = 8.
	lamda = avg/(((a-1)/a)**(1/a))
	return a/lamda*(x/lamda)**(a-1)*exp(-(x/lamda)**a)


def speed_frequ(speeds):
	x = speeds
	size = 30./(speeds)
	dx = 0.01
	x1 = 0.
	x2 = x1+dx
	location = size
	frequency = np.zeros(speeds)
	for i in range(3, speeds):
		while x1 <= location:
			dfrequency = dx*(weibull_prob(x1)+weibull_prob(x2))/2
			frequency[i] += dfrequency
			x1 = x2
			x2 += dx
		location += size
	return frequency


def wind_frequency_funcion():
	input_file = open("amalia_windrose_8.txt")
	wind_data = np.loadtxt(input_file)

	length_data = np.linspace(0,72.01,len(wind_data))
	f = interp1d(length_data, wind_data)
	return f


def frequ(bins):
	f = wind_frequency_funcion()
	bin_size = 72./bins
	dx = 0.01
	x1 = 0.
	x2 = x1+dx
	bin_location = bin_size
	frequency = np.zeros(bins)
	for i in range(0, bins):
		while x1 <= bin_location:
			dfrequency = dx*(f(x1)+f(x2))/2
			frequency[i] += dfrequency
			x1 = x2
			x2 += dx
		bin_location += bin_size
	return frequency


def calc_AEP(x):
    
    nTurbs = (len(x)-2)/2
    xHAWT = x[0:nTurbs]
    yHAWT = x[nTurbs:nTurbs*2]
    zTall = x[len(x)-2]
    zShort = x[len(x)-1]

    nTall = nTurbs/2
    zHAWT = np.zeros(nTurbs)
    zHAWT[0:nTall] = zTall
    zHAWT[nTall:nTurbs] = zShort
    
    theta = 0.1
    alpha = sp.tan(theta)
    rho = 1.1716
    a = 1. / 3.
    Cp = 4.*a*(1-a)**2.
    
    r_0 = np.ones(nTurbs)*63.2
    U_velocity = 8

    numDir = 18
    numSpeed = 18
    freqDir = frequ(numDir)
    freqSpeed = speed_frequ(numSpeed)
    AEP = 0
    for i in range(numDir):
        binSizeDir = 2.*pi/numDir
        direction = i*binSizeDir+binSizeDir/2.
        #print "Direction: ", i
        for j in range(numSpeed):
            #print "Speed: ", j
            binSizeSpeed = 27./numSpeed
            speed = 3+j*binSizeSpeed+binSizeSpeed/2.
            AEP += freqDir[i]*freqSpeed[j]*1.e6*jensen_power(xHAWT, yHAWT, zHAWT, r_0, alpha, a, speed, rho, Cp, direction)*24.*365.

    constraints = np.array([])

    for i in range(len(xHAWT)):
        for j in range(len(xHAWT)):
            if i==j:
                constraints = np.append(constraints,0)
            else:
                dx = xHAWT[i]-xHAWT[j]
                dy = yHAWT[i]-yHAWT[j]
                constraints = np.append(constraints, dx**2+dy**2-36.*r_0**2)

    constraints = constraints


    return -AEP/1e9, constraints/1e6

if __name__=="__main__":
    x = np.array([0, 0, 50, 50,300,300]) #x coordinates of the turbines
    y = np.array([0, 400, 0, 600,50,365]) #y coordinates of the turbines
    # z = np.array([150, 150, 150, 250, 250, 250, 350, 500, 350]) #hub height of each turbine
    # r_0 = np.array([40, 40, 40, 50, 50, 50, 60, 75, 60])
    # r_0 = np.ones(len(x))*20
    zTall = 150
    zShort = 50
    xin = np.hstack([x,y,zTall,zShort])
    
    lower = np.zeros(len(x)*2)
    upper = np.ones(len(x)*2)*400
    for i in range(2):
        lower = np.append(lower, 75)
        upper = np.append(upper, 125)
    print "Running..."

    startTime = time.time()
    optimizer = SNOPT()
    #optimizer.setOption('maxGen',100)
    xopt, fopt, info = optimize(calc_AEP, xin, lower, upper, optimizer)
    print "Time to run: ", time.time()-startTime

    print 'SNOPT:'
    print 'xopt: ', xopt
    print 'fopt: ', fopt
    print 'info: ', info

    print "Starting Power: ", calc_AEP(xin)[0]
    nTurbs = len(x)
    nTall = nTurbs/2
    zstart = np.zeros(len(x))
    zstart[0:nTall] = zTall
    zstart[nTall:nTurbs] = zShort

    xHAWT = xopt[0:nTurbs]
    yHAWT = xopt[nTurbs:nTurbs*2]
    zTall = xopt[len(xopt)-2]
    zShort = xopt[len(xopt)-1]
    zHAWT = np.zeros(nTurbs)
    zHAWT[0:nTall] = zTall
    zHAWT[nTall:nTurbs] = zShort
    r_0 = 30
    
    plt.figure(1)
    ax = Axes3D(plt.gcf())
    for i in range(len(x)):
        ax.scatter(x[i], y[i], zstart[i], c = 'r', s=pi*r_0**2, marker='.')
        fillstyles = ('none')
        #plt.axis(U_direction_radians)
        xtemp = (x[i], x[i])
        ytemp = (y[i], y[i])
        ztemp = (0, zstart[i]-r_0)
        ax.plot(xtemp, ytemp, ztemp, zdir='z', c='b', linewidth = 5.0)
     
    ax.set_xlim([0, np.max(x)])   
    ax.set_ylim([0, np.max(y)])  
    ax.set_zlim([0, np.max(zstart)])
    plt.title('Start')

    plt.figure(2)
    ax = Axes3D(plt.gcf())
    for i in range(len(x)):
        ax.scatter(xHAWT[i], yHAWT[i], zHAWT[i], c = 'r', s=pi*r_0**2, marker='.')
        fillstyles = ('none')
        #plt.axis(U_direction_radians)
        xtemp = (xHAWT[i], xHAWT[i])
        ytemp = (yHAWT[i], yHAWT[i])
        ztemp = (0, zHAWT[i]-r_0)
        ax.plot(xtemp, ytemp, ztemp, zdir='z', c='b', linewidth = 5.0)
     
    ax.set_xlim([0, np.max(xHAWT)])   
    ax.set_ylim([0, np.max(yHAWT)])  
    ax.set_zlim([0, np.max(zHAWT)])
    plt.title('Optimized')

    plt.show()
