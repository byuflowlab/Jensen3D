import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

def Jensen_Wake_Model(x):
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
    # nTurbines = len(xin)/2.
    r_0 = np.ones(nTurbs)*63.2
    U_velocity = 8.
    U_direction = pi/3.5
    "Make the graphic for the turbines and wakes"
    # jensen_plot(x, y, r_0, alpha, U_direction_radians)

    "Calculate power from each turbine"
    return jensen_power(xHAWT, yHAWT, zHAWT, r_0, alpha, a, U_velocity, rho, Cp, U_direction)

    # plt.show()


#Determine how much of the turbine is in the wake of the other turbines
def overlap(x, xdown, y, ydown, z, zdown, r, rdown, alpha):
    overlap_fraction = np.zeros(np.size(x))
    #print "x: ", x
    #print "y: ", y
    #print "z: ", z
    for i in range(0, np.size(x)):
        #define dx as the upstream x coordinate - the downstream x coordinate then rotate according to wind direction
        dx = xdown - x[i]
        #define dy as the upstream y coordinate - the downstream y coordinate then rotate according to wind direction
        dy = abs(ydown - y[i])
        dz = abs(zdown - z[i])
        d = sp.sqrt(dy**2.+dz**2.)
        R = r[i]+dx*alpha #The radius of the wake depending how far it is from the turbine
        A = rdown**2*pi #The area of the turbine
        if dx > 0:
            #if d <= R-rdown:
            #    overlap_fraction[i] = 1 #if the turbine is completely in the wake, overlap is 1, or 100%
            if d == 0:
                #print "Area of turbine: ", A
                #print "Area of wake: ", pi*R**2
                if A <= pi*R**2:
                    overlap_fraction[i] = 1.
                else: 
                    overlap_fraction[i] = pi*R**2/A
            elif d >= R+rdown:
                overlap_fraction[i] = 0 #if none of it touches the wake, the overlap is 0
            else:
                #if part is in and part is out of the wake, the overlap fraction is defied by the overlap area/rotor area
                overlap_area = rdown**2.*sp.arccos((d**2.+rdown**2.-R**2.)/(2.0*d*rdown))+R**2.*sp.arccos((d**2.+R**2.-rdown**2.)/(2.0*d*R))-0.5*sp.sqrt((-d+rdown+R)*(d+rdown-R)*(d-rdown+R)*(d+rdown+R))
                overlap_fraction[i] = overlap_area/A
        else:
            overlap_fraction[i] = 0 #turbines cannot be affected by any wakes that start downstream from them

    # print overlap_fraction
    return overlap_fraction #retrun the n x n matrix of how each turbine is affected by all of the others
                            #for example [0, 0.5]
                                        #[0, 0] means that the first turbine (row one) has half of its area in the
                                        #wake of the second turbine (row two). The overlap_fraction on the second
                                        #turbine is zero, so we can conclude that it is upstream of the first


#Jensen wake decay to determine the total velocity deficit at each turbine
def loss(r_0, a, alpha, x_focus, x, y_focus, y, overlap):
    loss = np.zeros(np.size(x))
    loss_squared = np.zeros(np.size(x))
    dx = np.zeros(len(x))
    dy = np.zeros(len(y))
    for i in range(0, np.size(x)):
        dx = x_focus-x[i]
        dy = abs(y_focus-y[i])
        R = r_0[i]+dx*alpha
        if dx > 0:          
            loss[i] = overlap[i]*2.*a*(r_0[i]/(r_0[i]+alpha*(dx)))**2*0.5*(np.cos(-dy*pi/(R+4*r_0[i])))
            loss_squared[i] = loss[i]**2
        else:
            loss[i] = 0
            loss_squared[i] = 0
    total_loss = sp.sqrt(np.sum(loss_squared))
    return total_loss


def jensen_power(x, y, z, r_0, alpha, a, U_velocity, rho, Cp, U_direction):
    "Effective velocity at each turbine"
    x_r, y_r = rotate(x, y, U_direction)
    """print "x: ", x
    print "y: ", y
    print "x_r: ", x_r
    print "y_r: ", y_r"""
    V = np.zeros([np.size(x)])
    total_loss = np.zeros([np.size(x)])
    P = np.zeros([np.size(x)])
    for i in range(0, np.size(x)):
        A = r_0[i]**2*pi
        overlap_fraction = overlap(x_r, x_r[i], y_r, y_r[i], z, z[i], r_0, r_0[i], alpha)
        total_loss[i] = loss(r_0, a, alpha, x_r[i], x_r, y_r[i], y_r, overlap_fraction)
        V = (1-total_loss[i])*U_velocity
        P[i] = 0.5*rho*A*Cp*V**3
    "Calculate Power from each turbine and the total"
    

    P_total = np.sum(P)
    return P_total


def jensen_plot(x, y, r_0, alpha, U_direction_radians):
    #plt.plot([x], [y], 'ro', markersize=10)

    wakes = np.linspace(0, 1000, num=101)

    for i in range(0, np.size(y)):
        turbine_y_top = y[i]+r_0
        turbine_y_bottom = y[i]-r_0
        turbine_x = [x[i], x[i], x[i]]
        turbine_y = [turbine_y_bottom, y[i], turbine_y_top]
        plt.plot(turbine_x, turbine_y, linewidth=2, c='r')
        for j in range(1, np.size(wakes)):
            wake_x = x[i]+wakes[j]
            wake_top_y = y[i]+r_0+wakes[j]*alpha
            wake_bottom_y = y[i]-r_0-wakes[j]*alpha
            plt.plot(wake_x, wake_top_y, 'b.', markersize=2)
            plt.plot(wake_x, wake_bottom_y, 'b.', markersize=2)


def rotate(x, y, U_direction_radians):
    x_r = x*np.cos(U_direction_radians)-y*np.sin(U_direction_radians)
    y_r = x*np.sin(U_direction_radians)+y*np.cos(U_direction_radians)
    return x_r, y_r


if __name__ == '__main__':
 
    "Define Variables"
    x = np.array([0, 0, 0, 500, 500, 500, 1000, 1000, 1000]) #x coordinates of the turbines
    y = np.array([0, 500, 1000, 0, 500, 1000, 0, 500, 1000]) #y coordinates of the turbines
    # z = np.array([150, 150, 150, 250, 250, 250, 350, 500, 350]) #hub height of each turbine
    # r_0 = np.array([40, 40, 40, 50, 50, 50, 60, 75, 60])
    # r_0 = np.ones(len(x))*20
    zTall = 150
    zShort = 50
    xin = np.hstack([x,y,zTall,zShort])
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = -90.
    

    U_direction_radians = (U_direction+90) * pi / 180.
    #print U_direction_radians

    # x_r, y_r = rotate(x, y, U_direction_radians)

    # Jensen_Wake_Model(overlap, loss, jensen_power, jensen_plot, x, y, r_0, alpha, U_direction_radians)
    # xin = np.hstack([x, y])
    print Jensen_Wake_Model(xin)
    ax = Axes3D(plt.gcf())
    """for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c = 'r', s=pi*r_0[i]**2, marker='.')
        fillstyles = ('none')
        #plt.axis(U_direction_radians)
        xtemp = (x[i], x[i])
        ytemp = (y[i], y[i])
        ztemp = (0, z[i]-r_0[i])
        ax.plot(xtemp, ytemp, ztemp, zdir='z', c='b', linewidth = 5.0)
     
    ax.set_xlim([0, np.max(x)])   
    ax.set_ylim([0, np.max(y)])  
    ax.set_zlim([0, np.max(z)])
    plt.show()"""

    # wakes = np.linspace(0, 1000, num=101)
    #
    # plt.figure(2)
    # for i in range(0, np.size(y)):
    #     turbine_x_top = ((wakes[0])*sp.cos(-U_direction_radians)-(r_0+alpha*wakes[0])*sp.sin(-U_direction_radians))+x[i]
    #     turbine_y_top = ((wakes[0])*sp.sin(-U_direction_radians)+(r_0+alpha*wakes[0])*sp.cos(-U_direction_radians))+y[i]
    #     turbine_x_bottom = ((wakes[0])*sp.cos(-U_direction_radians)-(-r_0-alpha*wakes[0])*sp.sin(-U_direction_radians))+x[i]
    #     turbine_y_bottom = ((wakes[0])*sp.sin(-U_direction_radians)+(-r_0-alpha*wakes[0])*sp.cos(-U_direction_radians))+y[i]
    #     turbine_x = [turbine_x_bottom, x[i], turbine_x_top]
    #     turbine_y = [turbine_y_bottom, y[i], turbine_y_top]
    #     plt.plot(turbine_x, turbine_y, linewidth=2, c='r')
    #     for j in range(1, np.size(wakes)):
    #         top_x = ((wakes[j])*sp.cos(-U_direction_radians)-(r_0+alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
    #         top_y = ((wakes[j])*sp.sin(-U_direction_radians)+(r_0+alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
    #         bottom_x = ((wakes[j])*sp.cos(-U_direction_radians)-(-r_0-alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
    #         bottom_y = ((wakes[j])*sp.sin(-U_direction_radians)+(-r_0-alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
    #         plt.plot([top_x], [top_y], 'b.', markersize=2)
    #         plt.plot([bottom_x], [bottom_y], 'b.', markersize=2)
    # plt.show()
