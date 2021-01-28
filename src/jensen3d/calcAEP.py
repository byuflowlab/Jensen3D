import numpy as np
from math import sqrt, pi, exp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Jensen import *
from datetime import datetime
from scipy.interpolate import interp1d


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


def calc_AEP(xin, params, numDir, numSpeed):
    nVAWT = params[0]
    rh = params[1]
    rv = params[2]
    rt = params[3]
    U_dir = params[4]
    U_vel = params[5]
    freqDir = frequ(numDir)
    freqSpeed = speed_frequ(numSpeed)
    # print "Direction Frequency Vector: ", freqDir
    # print "Speed Fequency Vector: ", freqSpeed
    AEP = 0
    for i in range(numDir):
        binSizeDir = 2.*pi/numDir
        direction = i*binSizeDir+binSizeDir/2.
        # print "Direction: ", i
        for j in range(numSpeed):
            # print "Speed: ", j
            binSizeSpeed = 27./numSpeed
            speed = 3+j*binSizeSpeed+binSizeSpeed/2.
            params = tuple([nVAWT, rh, rv,rt, direction, speed])
            AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, params)*24.*365.

    return AEP/1e6


if __name__=="__main__":

    xHAWT = np.array([0,0,0,500,500,500,1000,1000,1000])
    yHAWT = np.array([0,500,1000,0,500,1000,0,500,1000])
    rh = 40.
    nRows = 10   # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    """points = np.linspace(start=0, stop=(nRows-1)*spacing*rh, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    xHAWT = np.ndarray.flatten(xpoints)
    yHAWT = np.ndarray.flatten(ypoints)"""

    xVAWT = np.array([250])
    yVAWT = np.array([250])


    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    
    rv = 3.
    rt = 5.
    direction = 5.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.

    params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
    print("Running...")


    print(calc_AEP(xin, params, 20,20))
    """for i in range(1,100):
        print i
        print calc_AEP(xin, params, i,10), "MWhrs"""
    
    """AEP = np.array([])
    for i in range(1, 72):
        AEP = np.append(AEP, calc_AEP(xin, params, i, 30))
        print calc_AEP(xin, params, i, 30)
        print i
        np.savetxt("numDirConvergence.txt", np.c_[AEP])"""
    
    
