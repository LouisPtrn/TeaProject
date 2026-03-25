####################################################################################################
# Modelling the cooling of a cup of tea using Runge-Kutta numerical method.
#
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from math import pi


# functions for the differential equations:
# tea equation
def f(t, T, T_mug):
    # equation 1
    Q_dot = -(h_t*area*(T - T_room) + em*area*sigma*(T**4 - T_room**4) + (k_mug*area_s/d)*(T - T_mug))

    dT_dt = Q_dot/(m_tea*c_t)

    return dT_dt

# mug equation
def g(t, T, T_mug):
    # equation 2
    Q_dot = ((k_mug*area_s/d)*(T - T_mug) - h_m*area_s*(T_mug - T_room)
             - em_m*sigma*area_s*(T_mug**4 - T_room**4))

    dTmug_dt = Q_dot/(m_mug*c_m)

    return dTmug_dt

# runge-kutta (RK4) method
def runge_kutta(t0, x0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    x_values = np.zeros(len(t_values))
    y_values = np.zeros(len(t_values))
    x_values[0] = x0
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        x = x_values[i-1]
        y = y_values[i-1]

        k1x = f(t, x, y)
        k1y = g(t, x, y)

        k2x = f(t + h/2, x + h*k1x/2, y + h*k1y/2)
        k2y = g(t + h/2, x + h*k1x/2, y + h*k1y/2)

        k3x = f(t + h/2, x + h*k2x/2, y + h * k2y/2)
        k3y = g(t + h/2, x + h*k2x/2, y + h * k2y / 2)

        k4x = f(t + h, x + h*k3x, y + h*k3y)
        k4y = g(t + h, x + h*k3x, y + h*k3y)

        x_values[i] = x + (h/6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_values[i] = y + (h/6) * (k1y + 2 * k2y + 2 * k3y + k4y)

    return t_values, x_values


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
    # DEFINE PARAMETERS
    #########################################################################################
    sigma = const.sigma
    T_room = 289.15  # room temperature (K)
    k_mug = 1.5  # thermal conductivity of the mug material (W/m/K)
    r = 0.05  # radius of the mug (m)
    height = 0.09
    area = pi*r**2  # exposed drinking area
    area_s = (2*pi*r*height) + area # surface area of mug
    rho_tea = 1000
    rho_mug = 2400
    m_tea = rho_tea*area*height
    c_t = 4183  # specific heat cap of the tea
    c_m = 1050  # specific heat cap of the mug
    h_t = 10  # convective heat transfer coefficient of tea
    h_m = 25  # convective heat transfer coefficient of mug
    d = 0.005  # thickness of the mug wall (m)
    em = 0.95  # emissivity of the tea
    em_m = 0.92  # emissivity of the mug
    m_mug = rho_mug*2*pi*r*height*d + rho_mug*pi*r**2*d  # mass of mug
    #########################################################################################

    # set inital conditions
    t0 = 0
    x0 = 364.15 # tea starting temperature (K)
    #x0 = 356.15
    y0 = 289.15 # mug starting temp
    # how long to run the simulation (s)
    t_end = 1200
    # step size
    h = 1

    # Runge-Kutta method
    t_values, T_values = runge_kutta(t0, x0, y0, t_end, h)

    T_values -= 273.15  # convert to Celsius for plotting
    # Euler method
    #t_values_euler, y_values_euler = euler(t0, y0, t_end, h)
    #y_values_euler -= 273.15

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, T_values,"-", label='Runge-Kutta (just water)', color='blue')
    #plt.plot(t_values_euler/60, y_values_euler,".", label='Euler', color='red')

    plt.plot(t_end, 55, "x", label='55', color='green')

    plt.title("Modelled temperature of tea against time")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (deg C)")
    plt.legend()
    plt.grid()
    plt.show()
