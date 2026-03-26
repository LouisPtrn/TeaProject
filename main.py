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
    global assigned_T_eq
    global T_eq

    if t < t_milk:
        Q_dot = -(h_t*area*(T - T_room) + em*area*sigma*(T**4 - T_room**4) + (k_mug*area_s/d)*(T - T_mug))
        dT_dt = Q_dot/(m_tea*c_t)
    elif t > t_milk + t_equilibrium:
        Q_dot = -(h_t*area*(T - T_room) + em*area*sigma*(T**4 - T_room**4) + (k_mug*area_s/d)*(T - T_mug))
        dT_dt = Q_dot/(m_mixed*c_mixed)
    else:
        if not assigned_T_eq:

            # T_eq = (m_tea * c_t * T + m_milk * c_milk * T_milk + m_mug) / (m_tea * c_t + m_milk * c_milk)
            T_eq = (m_tea * c_t * T + m_milk * c_milk * T_milk + m_mug * c_m * T_mug) / (m_tea * c_t + m_milk * c_milk + m_mug * c_m)
            # T_eq = 344.15
            assigned_T_eq = True

        Q_dot = -(h_t*area*(T - T_room) + em*area*sigma*(T**4 - T_room**4) + (k_mug*area_s/d)*(T - T_mug))
        # get temperature of tea and milk in equilibrium
        dT_dt = Q_dot/(m_tea*c_t)-(T-T_eq)/tau_stir

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

    return t_values, x_values, y_values


def euler(t0, y0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        y_values[i] = y + h*f(t, y)
    return t_values, y_values


if __name__ == "__main__":
    T_val_list = []
    T_mug_val_list = []
    milk_times = []
    it_num = 5

    for i in range(it_num):
        assigned_T_eq = False

        # DEFINE PARAMETERS
        #########################################################################################
        sigma = const.sigma
        T_room = 294.55  # room temperature (K)
        k_mug = 1.5  # thermal conductivity of the mug material (W/m/K)
        r = 0.04  # radius of the mug (m)
        height = 0.088
        area = pi*r**2  # exposed drinking area
        area_s = (2*pi*r*height) + area # surface area of mug
        rho_tea = 1000
        rho_mug = 2400
        vol_tea = 250*10**-6 # volume of tea (m^3)
        m_tea = rho_tea*vol_tea
        c_t = 4183  # specific heat cap of the tea
        c_m = 1050  # specific heat cap of the mug
        h_t = 10  # convective heat transfer coefficient of tea
        h_m = 10.2  # convective heat transfer coefficient of mug
        d = 0.002  # thickness of the mug wall (m)
        em = 0.95  # emissivity of the tea
        em_m = 0.92  # emissivity of the mug
        m_mug = rho_mug*2*pi*r*height*d + rho_mug*pi*r**2*d  # mass of mug
        # m_mug = 0.29

        c_milk = 3900 # specific heat cap of the milk

        vol_milk = 25*10**-6
        m_milk = vol_milk*1000
        T_milk = 282.25
        tau_stir = 6

        t_equilibrium = 4*tau_stir
        c_mixed = 4154.7  # mixed specific heat cap of tea and milk
        m_mixed = m_tea + m_milk

        # h_i = 750

        #########################################################################################

        # set inital conditions
        t0 = 0
        x0 = 363.75 # tea starting temperature (K)
        #x0 = 356.15
        y0 = 289.15 # mug starting temp
        # how long to run the simulation (s)
        t_end = 1400
        # step size
        h = 0.1

        ideal_temp = 57.8

        t_milk = 200*i+95 # time when milk is added

        # Runge-Kutta method
        t_values, T_values, T_mug_values = runge_kutta(t0, x0, y0, t_end, h)

        T_values -= 273.15  # convert to Celsius for plotting
        T_mug_values -= 273.15

        T_val_list.append(T_values)
        T_mug_val_list.append(T_mug_values)

        # Load experimental data
        # t_exp, T_exp_values = np.loadtxt("experiment_3.txt", unpack=True, delimiter=",")

        #get time when ideal temp is reached
        ideal_time_model = None
        for t, T in zip(t_values, T_values):
            if T <= ideal_temp:
                ideal_time_model = t
                break

        # ideal_time_exp = None
        # for t, T in zip(t_exp, T_exp_values):
        #     if T <= ideal_temp:
        #         ideal_time_exp = t
        #         break

        print(f"Ideal temperature reached at {ideal_time_model:.2f} seconds in the model, t_milk = {t_milk}.")
        milk_times.append(t_milk)
        # print(f"Ideal temperature reached at {ideal_time_exp:.2f} seconds in the experiment, t_milk = 240")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    n = 0
    for T in T_val_list:
        colors = ["red", "orange", "green", "blue", "purple"]
        if n == 4:
            plt.plot(t_values, T, "-", label=str(milk_times[n]) + " s (optimal time)",color=colors[n])
        else:
            plt.plot(t_values, T, "-", label=str(milk_times[n]) + " s", color=colors[n])
        n += 1
    # plt.plot(t_exp, T_exp_values, "-", label='Experiment', color='red')
    # plt.plot(t_values, T_mug_val_list[0], "-", label='Runge-Kutta mug', color='black')

    # plot y=57.8
    plt.axhline(y=ideal_temp, color='green', linestyle='--', label='Ideal temperature')

    # plot x=t_milk, x=t_milk+t_equilibrium
    # plt.plot([t_milk, t_milk], [min(T_values), max(T_values)], color='purple', linestyle='--', label='milk added')
    # plt.plot([t_milk + t_equilibrium, t_milk + t_equilibrium], [min(T_values), max(T_values)], color='purple', linestyle='--', label='equilibrium reached')

    plt.title("Temperature of tea against time for different milk addition times")
    plt.xlabel("Time (s)")
    plt.ylabel("Liquid Temperature (°C)")
    plt.legend()
    plt.grid()
    plt.show()
