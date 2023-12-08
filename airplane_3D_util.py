import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import fsolve
import control
from airplane_3D_EOM_Linear import continous_to_discrete

def calculate_Vin_and_motor_speed(V_airspeed, thrust):
    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)
    a = params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**4 * params['motor']['C_T0']/(2*np.pi)**2
    b = params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**3 * \
        params['motor']['C_T1'] * V_airspeed/(2*np.pi)
    c = params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**2 * \
        params['motor']['C_T2'] * V_airspeed**2 - thrust

    motor_speed = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

    # prop
    V_in = (params['environmental']['rho_air_density'] * params['motor']['D_prop']**5 * params['motor']['C_Q0']/(2*np.pi)**2 * motor_speed ** 2
            + (params['environmental']['rho_air_density'] * params['motor']['D_prop']**4 * params['motor']['C_Q1'] * V_airspeed/(2*np.pi)
                + params['motor']['K_Q'] * params['motor']['K_V'] / params['motor']['R_motor']) * motor_speed
            + params['environmental']['rho_air_density'] *
            params['motor']['D_prop']**3 *
            params['motor']['C_Q2'] * V_airspeed**2
            + params['motor']['K_Q'] * params['motor']['i_0']) * params['motor']['R_motor'] / params['motor']['K_Q']

    return (V_in, motor_speed)

def calculate_thrust_and_motor_speed(V_airspeed, V_in):
    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)
    a = params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**5 * params['motor']['C_Q0'] / (2*np.pi)**2
    b = params['environmental']['rho_air_density'] * params['motor']['D_prop']**4 * params['motor']['C_Q1'] * V_airspeed / (2*np.pi)  \
        + params['motor']['K_Q'] * params['motor']['K_V'] / \
        params['motor']['R_motor']
    c = params['environmental']['rho_air_density'] * params['motor']['D_prop']**3 * params['motor']['C_Q2'] * V_airspeed**2 \
        - params['motor']['K_Q'] * V_in / params['motor']['R_motor'] + \
        params['motor']['K_Q'] * params['motor']['i_0']

    motor_speed = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

    thrust = params['environmental']['rho_air_density'] * params['motor']['D_prop']**4 * params['motor']['C_T0']/(2*np.pi)**2 * motor_speed ** 2 \
        + params['environmental']['rho_air_density'] * params['motor']['D_prop']**3 * params['motor']['C_T1'] * V_airspeed/(2*np.pi) * motor_speed \
        + params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**2 * params['motor']['C_T2'] * V_airspeed**2
    return (thrust, motor_speed)


def calculate_thrust(V_airspeed, motor_speed):
    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)

    thrust = params['environmental']['rho_air_density'] * params['motor']['D_prop']**4 * params['motor']['C_T0']/(2*np.pi)**2 * motor_speed ** 2 \
        + params['environmental']['rho_air_density'] * params['motor']['D_prop']**3 * params['motor']['C_T1'] * V_airspeed/(2*np.pi) * motor_speed \
        + params['environmental']['rho_air_density'] * \
        params['motor']['D_prop']**2 * params['motor']['C_T2'] * V_airspeed**2
    return (thrust)

def find_steady_straight_flight(earth_to_wind_angle, V_magnitude):

    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)
    g = params['environmental']['gravity']
    mass = params['physical']['mass']

    def equations(vars):
        u, w, theta, alpha, elevator = vars

        C_L = params['long_coef']['C_L_0'] + params['long_coef']['C_L_alpha'] * alpha + \
            params['long_coef']['C_L_elevator'] * elevator

        C_D = params['long_coef']['C_D_0'] + params['long_coef']['C_D_alpha'] * alpha + \
            params['long_coef']['C_D_elevator'] * elevator

        L = 0.5 * params['environmental']['rho_air_density'] * \
            V_magnitude**2 * params['physical']['Surface_area_wings'] * C_L
        D = 0.5 * params['environmental']['rho_air_density'] * \
            V_magnitude**2 * params['physical']['Surface_area_wings'] * C_D

        C_M_moment = params['long_coef']['C_M_moment_0'] + params['long_coef']['C_M_moment_alpha'] * alpha + \
            params['long_coef']['C_M_moment_elevator'] * elevator
        
        w_dot = g*np.cos(theta) - D*np.sin(alpha)/mass - L*np.cos(alpha)/mass
        alpha_zero = alpha - np.arctan2(w, u)
        V_magnitude_zero = V_magnitude - np.sqrt(u**2 + w**2)
        earth_to_wind_angle_zero = earth_to_wind_angle - (theta-alpha)

        return [C_M_moment, w_dot, alpha_zero, V_magnitude_zero, earth_to_wind_angle_zero]

    u, w, theta, alpha, elevator = fsolve(
        equations, (1, 1, earth_to_wind_angle, 0, 0))

    # now solve for thrust
    V_magnitude = np.sqrt(u**2 + w**2)
    C_L = params['long_coef']['C_L_0'] + params['long_coef']['C_L_alpha'] * alpha + \
        params['long_coef']['C_L_elevator'] * elevator

    C_D = params['long_coef']['C_D_0'] + params['long_coef']['C_D_alpha'] * alpha + \
        params['long_coef']['C_D_elevator'] * elevator

    L = 0.5 * params['environmental']['rho_air_density'] * \
        V_magnitude**2 * params['physical']['Surface_area_wings'] * C_L
    D = 0.5 * params['environmental']['rho_air_density'] * \
        V_magnitude**2 * params['physical']['Surface_area_wings'] * C_D

    thrust = g*np.sin(theta)*mass + D*np.cos(alpha) - L*np.sin(alpha)

    return (u, w, theta, alpha, elevator, thrust)


def calculate_Q_R_P():
    u_max = 50
    v_max = 50
    w_max = 50
    phi_max = np.pi
    theta_max = np.pi/32
    psi_max = np.pi
    p_max = np.pi
    q_max = np.pi
    r_max = np.pi


    # u_weight = 0.0001
    # w_weight = 0.0001
    # q_weight = 0.01
    # theta_weight = 100.25

    # v_weight = 100
    # p_weight = 0.025
    # r_weight = 0.025
    # phi_weight = 1.05
    # psi_weight = 100

    # thrust_max = 48
    # elevator_max = 0.3927
    # aileron_max = 0.3927
    # rudder_max = 0.3927

    # thrust_weight = 0.75
    # elevator_weight = 0.25
    # aileron_weight = 0.50
    # rudder_weight = 0.50

    # R_long_weight = 1.0
    # R_lat_weight = 1.0

    # P_long_weight = 10
    # P_lat_weight = 10

    u_weight = 0.0001
    w_weight = 0.0001
    q_weight = 0.01
    theta_weight = 100.25

    v_weight = 0.01
    p_weight = 0.025
    r_weight = 0.025
    phi_weight = 1.05
    psi_weight = 100

    thrust_max = 48
    elevator_max = 0.3927
    aileron_max = 0.3927
    rudder_max = 0.3927

    thrust_weight = 0.75
    elevator_weight = 0.25
    aileron_weight = 0.50
    rudder_weight = 0.50

    R_long_weight = 1.0
    R_lat_weight = 1.0

    P_long_weight = 10
    P_lat_weight = 10
    

    Q_long = np.array([[u_weight/u_max**2, 0, 0, 0, 0, 0],
                       [0, w_weight/w_max**2, 0, 0, 0, 0],
                       [0, 0, q_weight/q_max**2, 0, 0, 0],
                       [0, 0, 0, theta_weight/theta_max**2, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])

    
    

    Q_lat = np.array([[v_weight/v_max**2, 0, 0, 0, 0, 0],
                      [0, p_weight/p_max**2, 0, 0, 0, 0],
                      [0, 0, r_weight/r_max**2, 0, 0, 0],
                      [0, 0, 0, phi_weight/phi_max**2, 0, 0],
                      [0, 0, 0, 0, psi_weight/psi_max**2, 0],
                      [0, 0, 0, 0, 0, 0]])

    
    R_long = np.array([[thrust_weight/thrust_max**2, 0],
                       [0, elevator_weight/elevator_max**2]])
    
    R_long = R_long * R_long_weight

    R_lat = np.array([[aileron_weight/aileron_max**2, 0],
                      [0, rudder_weight/rudder_max**2]])
    
    R_lat = R_lat * R_lat_weight


    P_lat = Q_lat * P_lat_weight
    P_long = Q_long * P_long_weight

    return Q_long, Q_lat, R_long, R_lat, P_long, P_lat

def calculate_K(state_eq, input_eq, dt):
    from airplane_3D_EOM_Linear import continuous_linearization, A_matrix_evalation

    A_long4, B_long4, A_long6, B_long6, A_lat5, B_lat5, A_lat6, B_lat6 = continuous_linearization(
        state_eq.squeeze(), input_eq.squeeze())

    u_max = 50
    v_max = 50
    w_max = 50
    phi_max = np.pi
    theta_max = np.pi/32
    psi_max = np.pi
    p_max = np.pi
    q_max = np.pi
    r_max = np.pi


    u_weight = 0.0001
    w_weight = 0.0001
    q_weight = 0.01
    theta_weight = 100.25

    Q_long = np.array([[u_weight/u_max**2, 0, 0, 0],
                       [0, w_weight/w_max**2, 0, 0],
                       [0, 0, q_weight/q_max**2, 0],
                       [0, 0, 0, theta_weight/theta_max**2]])

    
    v_weight = 100
    p_weight = 0.025
    r_weight = 0.025
    phi_weight = 1.05
    psi_weight = 100

    Q_lat = np.array([[v_weight/v_max**2, 0, 0, 0, 0],
                      [0, p_weight/p_max**2, 0, 0, 0],
                      [0, 0, r_weight/r_max**2, 0, 0],
                      [0, 0, 0, phi_weight/phi_max**2, 0],
                      [0, 0, 0, 0, psi_weight/psi_max**2]])

    thrust_max = 48
    elevator_max = 0.3927
    aileron_max = 0.3927
    rudder_max = 0.3927

    thrust_weight = 0.75
    elevator_weight = 0.25
    R_long = np.array([[thrust_weight/thrust_max**2, 0],
                       [0, elevator_weight/elevator_max**2]])
    R_long_weight = 1.0

    aileron_weight = 0.50
    rudder_weight = 0.50
    R_lat = np.array([[aileron_weight/aileron_max**2, 0],
                      [0, rudder_weight/rudder_max**2]])
    R_lat_weight = 1.0

    K_long, S, e = control.lqr(A_long4, B_long4, Q_long, R_long_weight*R_long)

    K_lat, S, e = control.lqr(A_lat5, B_lat5, Q_lat, R_lat_weight*R_lat)

    A_long_closed = A_long4 - B_long4 @ K_long
    A_lat_closed = A_lat5 - B_lat5 @ K_lat

    long_open_eig, _ = A_matrix_evalation(A_long4, dt)
    long_closed_eig, _ = A_matrix_evalation(A_long_closed, dt)
    lat_open_eig, _ = A_matrix_evalation(A_lat5, dt)
    lat_closed_eig, _ = A_matrix_evalation(A_lat_closed, dt)

    x = [ele.real for ele in long_open_eig]
    y = [ele.imag for ele in long_open_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of long open eig: {len(real_positive)}")

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.title("Longitudinal Open Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')

    ###

    x = [ele.real for ele in long_closed_eig]
    y = [ele.imag for ele in long_closed_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of long closed eig: {len(real_positive)}")

    plt.subplot(2, 2, 2)
    plt.title("Longitudinal Closed Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')

    ####
    x = [ele.real for ele in lat_open_eig]
    y = [ele.imag for ele in lat_open_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of lat open eig: {len(real_positive)}")

    plt.subplot(2, 2, 3)
    plt.title("Lat Open Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')

    ###

    x = [ele.real for ele in lat_closed_eig]
    y = [ele.imag for ele in lat_closed_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of lat closed eig: {len(real_positive)}")

    plt.subplot(2, 2, 4)
    plt.title("Lat Closed Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')

    plt.tight_layout()
    plt.show()

    return K_long, K_lat


def calculate_K2(state_eq, input_eq, dt):
    from airplane_3D_EOM_Linear import continuous_linearization, A_matrix_evalation

    A_long4, B_long4, A_long6, B_long6, A_lat5, B_lat5, A_lat6, B_lat6 = continuous_linearization(
        state_eq.squeeze(), input_eq.squeeze())

    u_max = 50
    v_max = 50
    w_max = 50
    phi_max = np.pi
    theta_max = np.pi/32
    psi_max = np.pi
    p_max = np.pi
    q_max = np.pi
    r_max = np.pi

    u_weight = 0.1
    w_weight = 0.1
    q_weight = 0.000001
    theta_weight = 10000.799999

    # u_weight = 100.25
    # w_weight = 0.25
    # q_weight = 0.25
    # theta_weight = 0.25

    Q_long = np.array([[u_weight/u_max**2, 0, 0, 0],
                       [0, w_weight/w_max**2, 0, 0],
                       [0, 0, q_weight/q_max**2, 0],
                       [0, 0, 0, theta_weight/theta_max**2]])

    v_weight = 100
    p_weight = 0.025
    r_weight = 0.025
    phi_weight = 0.05
    psi_weight = 100
    # v_weight = 0.2
    # p_weight = 0.2
    # r_weight = 0.2
    # phi_weight = 10.02
    # psi_weight = 10.02

    Q_lat = np.array([[v_weight/v_max**2, 0, 0, 0, 0],
                      [0, p_weight/p_max**2, 0, 0, 0],
                      [0, 0, r_weight/r_max**2, 0, 0],
                      [0, 0, 0, phi_weight/phi_max**2, 0],
                      [0, 0, 0, 0, psi_weight/psi_max**2]])

    thrust_max = 48
    elevator_max = 0.3927
    aileron_max = 0.3927
    rudder_max = 0.3927

    thrust_weight = 0.75
    elevator_weight = 0.25
    R_long = np.array([[thrust_weight/thrust_max**2, 0],
                       [0, elevator_weight/elevator_max**2]])
    R_long_weight = 1.0

    aileron_weight = 0.50
    rudder_weight = 0.50
    R_lat = np.array([[aileron_weight/aileron_max**2, 0],
                      [0, rudder_weight/rudder_max**2]])
    R_lat_weight = 1.0

    A_long4_dis, B_long4_dis = continous_to_discrete(A_long4, B_long4, dt)
    A_lat5_dis, B_lat5_dis = continous_to_discrete(A_lat5, B_lat5, dt)

    # K_long, S, e = control.lqr(A_long4, B_long4, Q_long, R_long_weight*R_long)
    # K_lat, S, e = control.lqr(A_lat5, B_lat5, Q_lat, R_lat_weight*R_lat)

    K_long, S, e = control.dlqr(A_long4_dis, B_long4_dis, Q_long, R_long_weight*R_long)
    K_lat, S, e = control.dlqr(A_lat5_dis, B_lat5_dis, Q_lat, R_lat_weight*R_lat)


    A_long_closed = A_long4_dis - B_long4_dis @ K_long
    A_lat_closed = A_lat5_dis - B_lat5_dis @ K_lat

    long_open_eig, _ = A_matrix_evalation(A_long4_dis, dt)
    long_closed_eig, _ = A_matrix_evalation(A_long_closed, dt)
    lat_open_eig, _ = A_matrix_evalation(A_lat5_dis, dt)
    lat_closed_eig, _ = A_matrix_evalation(A_lat_closed, dt)

    x = [ele.real for ele in long_open_eig]
    y = [ele.imag for ele in long_open_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of long open eig: {len(real_positive)}")

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.title("Longitudinal Open Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    # plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    # plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')
    x_cir = np.arange(-1, 1, 0.01)
    y_cir_positive = np.sqrt(1 - x_cir**2)
    plt.plot(x_cir, y_cir_positive, 'k', x_cir, -y_cir_positive, 'k')

    ###

    x = [ele.real for ele in long_closed_eig]
    y = [ele.imag for ele in long_closed_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of long closed eig: {len(real_positive)}")

    plt.subplot(2, 2, 2)
    plt.title("Longitudinal Closed Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    # plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    # plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')
    x_cir = np.arange(-1, 1, 0.01)
    y_cir_positive = np.sqrt(1 - x_cir**2)
    plt.plot(x_cir, y_cir_positive, 'k', x_cir, -y_cir_positive, 'k')

    ####
    x = [ele.real for ele in lat_open_eig]
    y = [ele.imag for ele in lat_open_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of lat open eig: {len(real_positive)}")

    plt.subplot(2, 2, 3)
    plt.title("Lat Open Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    # plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    # plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')
    x_cir = np.arange(-1, 1, 0.01)
    y_cir_positive = np.sqrt(1 - x_cir**2)
    plt.plot(x_cir, y_cir_positive, 'k', x_cir, -y_cir_positive, 'k')

    ###

    x = [ele.real for ele in lat_closed_eig]
    y = [ele.imag for ele in lat_closed_eig]
    real_positive = [ele for ele in x if ele <= 0]

    print(
        f"number of nonpositive real parts of lat closed eig: {len(real_positive)}")

    plt.subplot(2, 2, 4)
    plt.title("Lat Closed Eigen")
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    # plt.plot(np.arange(-2, 3, 1), np.zeros(5), 'k')
    # plt.plot(np.zeros(5), np.arange(-2, 3, 1), 'k')
    x_cir = np.arange(-1, 1, 0.01)
    y_cir_positive = np.sqrt(1 - x_cir**2)
    plt.plot(x_cir, y_cir_positive, 'k', x_cir, -y_cir_positive, 'k')

    plt.tight_layout()
    plt.show()

    return K_long, K_lat


def LQR_controller(state, input_init, dt, waypoint, is_new_waypoint,\
                    state_desired_list, input_list, K_long, K_lat):
    ### load in stored parameters and calculate other parameters###
    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)
    clip_inputs = True

    x = state[0]
    y = state[1]
    z = state[2]
    u = state[3]
    v = state[4]
    w = state[5]

    phi = state[6]
    theta = state[7]
    psi = state[8]
    p = state[9]
    q = state[10]
    r = state[11]

    long_variable = np.array([[u], [w], [q], [theta]])
    lat_variable = np.array([[v], [p], [r], [phi], [psi]])

    

    x_error = waypoint[0] - x
    y_error = waypoint[1] - y
    z_error = waypoint[2] - z

    psi_des = np.arctan2(y_error, x_error)
    earth_to_wind_angle_des = np.arctan2(-z_error,
                                         np.sqrt(x_error**2 + y_error**2))
    V_magnitude = 20

    u_des, w_des, theta_des, alpha_des, elevator_des, thrust_des = find_steady_straight_flight(
        earth_to_wind_angle_des, V_magnitude)

    long_state_des = np.array([[u_des], [w_des], [0], [theta_des]])
    lat_state_des = np.array([[0], [0], [0], [0], [psi_des]])

    state_desired_list[0].append(waypoint[0])
    state_desired_list[1].append(waypoint[1])
    state_desired_list[2].append(waypoint[2])
    state_desired_list[3].append(u_des)
    state_desired_list[4].append(0)
    state_desired_list[5].append(w_des)
    state_desired_list[6].append(0)
    state_desired_list[7].append(theta_des)
    state_desired_list[8].append(psi_des)
    state_desired_list[9].append(0)
    state_desired_list[10].append(0)
    state_desired_list[11].append(0)

    # if heading is far off, wait until heading is good, then recalculate K
    # heading_threshold = np.pi/32
    # if np.abs(psi_des - psi) > heading_threshold:
    #     heading_good[0] = False
    # else:
    #     if heading_good[0] == False:
    #         heading_good[0] = True
    #         K_lat_list.pop()
    #         K_long_list.pop()

    if is_new_waypoint:
        K_long, K_lat = calculate_K2(state, input_init, dt)

    
    long_input = -K_long @ (long_variable - long_state_des)
    lat_input = -K_lat @ (lat_variable - lat_state_des)
    thrust = input_init[0] + long_input[0, 0]
    elevator = input_init[1] + long_input[1, 0]
    aileron = input_init[2] + lat_input[0, 0]
    rudder = input_init[3] + lat_input[1, 0]

    # actually, just set thrust and elevator according to calculation
    thruts = thrust_des
    elevator = elevator_des

    # print(f"des input: thrust:{thrust}, elevator:{elevator}, aileron:{aileron}, rudder:{rudder}")
        

    # clip inputs
    # first, clip change in input
    #V_airspeed_mag = np.sqrt(u**2 + v**2 + w**2)
    if clip_inputs:
        if not input_list[0]:  # if list is empty, then this is first pass, compare with inital inputs
            thrust_previous = input_init[0]
            elevator_previous = input_init[1]
            aileron_previous = input_init[2]
            rudder_previous = input_init[3]
        else:
            thrust_previous = input_list[0][-1]
            elevator_previous = input_list[1][-1]
            aileron_previous = input_list[2][-1]
            rudder_previous = input_list[3][-1]

        if thrust - thrust_previous > params['control_limits']['thrust_max_change_per_second'] * dt:
            thrust = thrust_previous + \
                params['control_limits']['thrust_max_change_per_second'] * dt
        elif thrust_previous - thrust > params['control_limits']['thrust_max_change_per_second'] * dt:
            thrust = thrust_previous - \
                params['control_limits']['thrust_max_change_per_second'] * dt
        if elevator - elevator_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
            elevator = elevator_previous + \
                params['control_limits']['control_surface_max_change_per_second'] * dt
        elif elevator_previous - elevator > params['control_limits']['control_surface_max_change_per_second'] * dt:
            elevator = elevator_previous - \
                params['control_limits']['control_surface_max_change_per_second'] * dt
        if aileron - aileron_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
            aileron = aileron_previous + \
                params['control_limits']['control_surface_max_change_per_second'] * dt
        elif aileron_previous - aileron > params['control_limits']['control_surface_max_change_per_second'] * dt:
            aileron = aileron_previous - \
                params['control_limits']['control_surface_max_change_per_second'] * dt
        if rudder - rudder_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
            rudder = rudder_previous + \
                params['control_limits']['control_surface_max_change_per_second'] * dt
        elif rudder_previous - rudder > params['control_limits']['control_surface_max_change_per_second'] * dt:
            rudder = rudder_previous - \
                params['control_limits']['control_surface_max_change_per_second'] * dt

        # next, clip range of control surfaces
        # +- 15 degrees for conservative flight
        # +- 25 degrees for aggressive flight
        # past 15 degrees, the system loses linearity, so the linear model
        # may not be as good as an approximate
        # here we do +- 22.5 degrees
        if thrust > params['control_limits']['thrust_max']:
            thrust = params['control_limits']['thrust_max']
        elif thrust < params['control_limits']['thrust_min']:
            thrust = params['control_limits']['thrust_min']
        if elevator > params['control_limits']['control_surface_max']:
            elevator = params['control_limits']['control_surface_max']
        elif elevator < params['control_limits']['control_surface_min']:
            elevator = params['control_limits']['control_surface_min']
        if aileron > params['control_limits']['control_surface_max']:
            aileron = params['control_limits']['control_surface_max']
        elif aileron < params['control_limits']['control_surface_min']:
            aileron = params['control_limits']['control_surface_min']
        if rudder > params['control_limits']['control_surface_max']:
            rudder = params['control_limits']['control_surface_max']
        elif rudder < params['control_limits']['control_surface_min']:
            rudder = params['control_limits']['control_surface_min']

    input_list[0].append(thrust)
    input_list[1].append(elevator)
    input_list[2].append(aileron)
    input_list[3].append(rudder)

    input_controller = np.array([[thrust], [elevator], [aileron], [rudder]])

    return input_controller, K_long, K_lat