####################################################################################################
# Modelling the cooling of a cup of tea using Runge-Kutta numerical method.
#
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from math import pi


# function for the differential equation
# this, and the parameters in it can be modified
def f(t, T):
    # assume parameters
    sigma = const.sigma
    T_room = 292.15  # room temperature (K)
    k_mug = 1.5  # thermal conductivity of the mug material (W/m/K)
    r = 0.045 # radius of the mug (m)
    height = 0.09
    area = pi*r**2
    area_surf = (2*pi*r*height) + (2*area)
    rho = 1000
    m = rho * area * height
    cp = 4183  # specific heat capacity of the liquid (J/kg/K)
    h = 10  # convective heat transfer coefficient (W/m^2K)
    d = 0.002  # thickness of the mug wall (m)
    em = 0.9  # emissivity of the mug surface

    h_eff = h*area + (k_mug*area_surf)/d
    Q_dot_rad = em*sigma*area_surf*((T**4) - (T_room**4))
    Q_dot_cond_conv = h_eff*area_surf*(T - T_room)

    dT_dt = -(Q_dot_rad + Q_dot_cond_conv) / (m*cp)

    return dT_dt


# runge-kutta (RK4) method
def runge_kutta(t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        y_values[i] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t_values, y_values


def euler(t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        y_values[i] = y + h*f(t, y)
    return t_values, y_values


def experimental_data():
    return


if __name__ == "__main__":
    # set parameters
    t0 = 0 # minutes
    # starting temperature (K)
    y0 = 364.15
    #y0 = 356.15
    # how long to run the simulation (s)
    t_end = 7200
    # step size
    h = 1

    # Runge-Kutta method
    t_values, y_values = runge_kutta(t0, y0, t_end, h)
    y_values -= 273.15  # convert to Celsius for plotting
    # Euler method
    #t_values_euler, y_values_euler = euler(t0, y0, t_end, h)
    #y_values_euler -= 273.15

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values,"-", label='Runge-Kutta (just water)', color='blue')
    #plt.plot(t_values_euler/60, y_values_euler,".", label='Euler', color='red')

    #plt.plot(t_end, 55, "x", label='55', color='green')

    plt.title("Modelled temperature of tea against time")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (deg C)")
    plt.legend()
    plt.grid()
    plt.show()
