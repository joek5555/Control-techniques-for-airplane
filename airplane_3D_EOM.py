import numpy as np
from scipy.integrate import solve_ivp
import yaml
from airplane_3D_util import find_steady_straight_flight, \
    calculate_thrust_and_motor_speed, \
    calculate_Vin_and_motor_speed,\
    calculate_thrust,\
    calculate_K
    
from airplane_3D_EOM_Linear import continuous_linearization, continous_to_discrete


# these rotation matrixs take the angle between the earth and the body frame. 
# V_b = Rot_x V_e
# the rotation matrix can give you a vector in base frame if given vector in earth frame

# to get vector in earth frame given vector in base frame
# V_e = (Rot_x)^T V_b


def Rot_x(theta):
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(theta), np.sin(theta)],
                      [0, -np.sin(theta), np.cos(theta)]])
    return rot_x

def Rot_y(theta):
    rot_y = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1, 0],
                      [np.sin(theta), 0, np.cos(theta)]])
    return rot_y

def Rot_z(theta):
    rot_z = np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    return rot_z

def calculate_aero_forces(u,v,w, p,q,r,elevator,aileron, rudder, params):
    

    V_magnitude = np.sqrt(u*u + v*v + w*w)

    if w == 0 and u == 0:
        alpha = 0
    else:
        alpha = np.arctan2(w,u)

    if v == 0:
        beta = 0
    else:
        beta = np.arcsin(v/V_magnitude)

    # long coef and forces
    if V_magnitude == 0:
        L = 0
        D = 0
        M_moment = 0
        C_M_moment = 0
    else:
        C_L = params['long_coef']['C_L_0'] + params['long_coef']['C_L_alpha'] * alpha + \
            params['long_coef']['C_L_q']*(params['physical']['c_wing_chord']/(2 *V_magnitude))*q +\
            params['long_coef']['C_L_elevator'] * elevator
        
        C_D = params['long_coef']['C_D_0'] + params['long_coef']['C_D_alpha'] * alpha + \
            params['long_coef']['C_D_q']*(params['physical']['c_wing_chord']/(2 *V_magnitude))*q +\
            params['long_coef']['C_D_elevator'] * elevator
        
        C_M_moment = params['long_coef']['C_M_moment_0'] + params['long_coef']['C_M_moment_alpha'] * alpha + \
            params['long_coef']['C_M_moment_q']*(params['physical']['c_wing_chord']/(2 *V_magnitude))*q +\
            params['long_coef']['C_M_moment_elevator'] * elevator
        
    
        L = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings'] * C_L
        D = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings'] * C_D
        M_moment = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings']\
                    * params['physical']['c_wing_chord'] * C_M_moment
        
        # lat coef
        if V_magnitude == 0:
            F_y = 0
            L_moment = 0
            N_moment = 0
        else:

            C_y = params['lat_coef']['C_y_0'] + params['lat_coef']['C_y_beta'] * beta + \
                params['lat_coef']['C_y_p']*(params['physical']['b_wing_span']/(2 *V_magnitude))*p +\
                params['lat_coef']['C_y_r']*(params['physical']['b_wing_span']/(2 *V_magnitude))*r +\
                params['lat_coef']['C_y_aileron'] * aileron + params['lat_coef']['C_y_rudder'] * rudder
            
            C_L_moment = params['lat_coef']['C_L_moment_0'] + params['lat_coef']['C_L_moment_beta'] * beta + \
                params['lat_coef']['C_L_moment_p']*(params['physical']['b_wing_span']/(2 *V_magnitude))*p +\
                params['lat_coef']['C_L_moment_r']*(params['physical']['b_wing_span']/(2 *V_magnitude))*r +\
                params['lat_coef']['C_L_moment_aileron'] * aileron + params['lat_coef']['C_L_moment_rudder'] * rudder
            
            C_N_moment = params['lat_coef']['C_N_moment_0'] + params['lat_coef']['C_N_moment_beta'] * beta + \
                params['lat_coef']['C_N_moment_p']*(params['physical']['b_wing_span']/(2 *V_magnitude))*p +\
                params['lat_coef']['C_N_moment_r']*(params['physical']['b_wing_span']/(2 *V_magnitude))*r +\
                params['lat_coef']['C_N_moment_aileron'] * aileron + params['lat_coef']['C_N_moment_rudder'] * rudder
        
            F_y = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings'] * C_y
            L_moment = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings'] \
                        * params['physical']['b_wing_span'] * C_L_moment
            N_moment = 0.5 * params['environmental']['rho_air_density'] * V_magnitude**2 * params['physical']['Surface_area_wings']\
                        * params['physical']['b_wing_span'] * C_N_moment

    return [V_magnitude, alpha, beta, L, D, M_moment, F_y, L_moment, N_moment]



def simulate_airplane(state_init, waypoints, input_init, duration, dt, controller="LQR"):

    x_des_list = []
    y_des_list = []
    z_des_list = []
    u_des_list = []
    v_des_list = []
    w_des_list = []
    phi_des_list = []
    theta_des_list = []
    psi_des_list = []
    p_des_list = []
    q_des_list = []
    r_des_list = []

    thrust_list = []
    elevator_list = []
    aileron_list = []
    rudder_list = []
    t_list = []
    K_lat_list = []
    K_long_list = []


    A_long4_dis_list =[]
    B_long4_dis_list =[]
    A_lat5_dis_list =[]
    B_lat5_dis_list =[]

    wind_t = [-1]
    x_wind_list = []
    y_wind_list = []
    z_wind_list = []





    def airplane_model(t,state, waypoints, input_init, dt, controller):

        ### load in stored parameters and calculate other parameters###
        with open("aerosonde_parameters.yaml", 'r') as file:
            params = yaml.safe_load(file)
        clip_inputs = True
        is_distrubances = True

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

        
        # check to see if you have hit waypoint
        if not waypoints: # if no more waypoints, do not change any of the states
            return(np.array([0, 0, 0,
                         0, 0, 0,
                         0, 0, 0,
                         0, 0, 0]))
        
        waypoint_precision = 5
        # if you are close to a waypoint
        if np.abs(x - waypoints[0][0]) < waypoint_precision and\
            np.abs(y - waypoints[0][1]) < waypoint_precision and\
            np.abs(z - waypoints[0][2]) < waypoint_precision:

            print("Waypoint reached!")
            waypoints.pop(0)
            if controller == "LQR":
                K_lat_list.pop()
                K_long_list.pop()
            elif controller == "MPC":
                A_long4_dis_list.pop()
                B_long4_dis_list.pop()
                A_lat5_dis_list.pop()
                B_lat5_dis_list.pop()
            if not waypoints:
                print("All Waypoints Reached")
                return(np.array([0, 0, 0,
                         0, 0, 0,
                         0, 0, 0,
                         0, 0, 0]))

        x_error = waypoints[0][0] - x
        y_error = waypoints[0][1] - y
        z_error = waypoints[0][2] - z

        psi_des = np.arctan2(y_error,x_error)
        earth_to_wind_angle_des = np.arctan2(-z_error, np.sqrt(x_error**2 + y_error**2))
        V_magnitude = 20

        

        u_des, w_des, theta_des, alpha_des, elevator_des, thrust_des = find_steady_straight_flight(earth_to_wind_angle_des, V_magnitude)


        long_state_des = np.array([[u_des],[w_des],[0],[theta_des]])
        lat_state_des = np.array([[0],[0],[0],[0], [psi_des]])

        long_state_shifted = (long_variable - long_state_des )
        lat_state_shifted = (lat_variable - lat_state_des)

        # make sure the angles are kept between -pi and pi
        # if you are at -160 and you want to go to 160, you should have a 
        # desired of 40, not -320 

        if lat_state_shifted[4,0] > np.pi:
            lat_state_shifted[4,0] = -2*np.pi + lat_state_shifted[4,0]
        elif lat_state_shifted[4,0] < -np.pi:
            lat_state_shifted[4,0] = 2*np.pi + lat_state_shifted[4,0]


        x_des_list.append(waypoints[0][0])
        y_des_list.append(waypoints[0][1])
        z_des_list.append(waypoints[0][2])
        u_des_list.append(u_des)
        v_des_list.append(0)
        w_des_list.append(w_des)
        phi_des_list.append(0)
        theta_des_list.append(theta_des)
        psi_des_list.append(psi_des)
        p_des_list.append(0)
        q_des_list.append(0)
        r_des_list.append(0)
        t_list.append(t)


        steady_state_input = np.array([thrust_des, elevator_des, 0,0])
        # steady_state = np.array([[-x_error], [-y_error], [-z_error], 
        #                [long_state_shifted[0,0]], [lat_state_shifted[0,0]], [long_state_shifted[1,0]],
        #                [lat_state_shifted[3,0]], [long_state_shifted[3,0]], [lat_state_shifted[4,0]], 
        #                [lat_state_shifted[1,0]], [long_state_shifted[2,0]], [lat_state_shifted[2,0]]])



        if controller == "LQR":
            if not K_lat_list:
                # K_long, K_lat = calculate_K(state, input_init, dt)
                K_long, K_lat = calculate_K(state, steady_state_input, dt)
                # K_long, K_lat = calculate_K(steady_state, steady_state_input, dt)
                K_long_list.append(K_long)
                K_lat_list.append(K_lat)
                long_input = -K_long_list[0] @ long_state_shifted
                lat_input = -K_lat_list[0] @ lat_state_shifted
                thrust = steady_state_input[0] + long_input[0,0]
                elevator = steady_state_input[1] + long_input[1,0]
                aileron = steady_state_input[2] + lat_input[0,0]
                rudder = steady_state_input[3] + lat_input[1,0]

            else:
                long_input = -K_long_list[0] @ long_state_shifted
                lat_input = -K_lat_list[0] @ lat_state_shifted
                thrust = steady_state_input[0] + long_input[0,0]
                elevator = steady_state_input[1] + long_input[1,0]
                aileron = steady_state_input[2] + lat_input[0,0]
                rudder = steady_state_input[3] + lat_input[1,0]


        # clip inputs
        # first, clip change in input
        #V_airspeed_mag = np.sqrt(u**2 + v**2 + w**2)
        if clip_inputs:
            if not thrust_list: # if list is empty, then this is first pass, compare with inital inputs
                thrust_previous = input_init[0]
                elevator_previous = input_init[1]
                aileron_previous = input_init[2]
                rudder_previous = input_init[3]
            else:
                thrust_previous = thrust_list[-1]
                elevator_previous = elevator_list[-1]
                aileron_previous = aileron_list[-1]
                rudder_previous = rudder_list[-1]

            
            if thrust - thrust_previous > params['control_limits']['thrust_max_change_per_second'] * dt:
                thrust = thrust_previous + params['control_limits']['thrust_max_change_per_second'] * dt
            elif thrust_previous - thrust > params['control_limits']['thrust_max_change_per_second'] * dt:
                thrust = thrust_previous - params['control_limits']['thrust_max_change_per_second'] * dt
            if elevator - elevator_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
                elevator = elevator_previous + params['control_limits']['control_surface_max_change_per_second'] * dt
            elif elevator_previous - elevator > params['control_limits']['control_surface_max_change_per_second'] * dt:
                elevator = elevator_previous - params['control_limits']['control_surface_max_change_per_second'] * dt
            if aileron - aileron_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
                aileron = aileron_previous + params['control_limits']['control_surface_max_change_per_second'] * dt
            elif aileron_previous - aileron > params['control_limits']['control_surface_max_change_per_second'] * dt:
                aileron = aileron_previous - params['control_limits']['control_surface_max_change_per_second'] * dt
            if rudder - rudder_previous > params['control_limits']['control_surface_max_change_per_second'] * dt:
                rudder = rudder_previous + params['control_limits']['control_surface_max_change_per_second'] * dt
            elif rudder_previous - rudder > params['control_limits']['control_surface_max_change_per_second'] * dt:
                rudder = rudder_previous - params['control_limits']['control_surface_max_change_per_second'] * dt

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


        thrust_list.append(thrust)
        elevator_list.append(elevator)
        aileron_list.append(aileron)
        rudder_list.append(rudder)


        if is_distrubances and t > wind_t[0]:
            
            time_sample = 0.5
            wind_strength = 6
            average_how_often_wind_gust_in_seconds = 20
            prob_wind_starts = time_sample/average_how_often_wind_gust_in_seconds
            prob_wind_stays = 0.5

            wind_t[0] += time_sample
            random_vector = np.random.rand(3)
            if not x_wind_list or x_wind_list[-1] == 0:
                if random_vector[0] < prob_wind_starts:
                    x_wind = np.random.rand(1)[0]*2*wind_strength - wind_strength
                else:
                    x_wind = 0
            else:
                if random_vector[0] < prob_wind_stays:
                    x_wind = x_wind_list[-1]
                else:
                    x_wind = 0
            
            
            if not y_wind_list or y_wind_list[-1] == 0:
                if random_vector[1] < prob_wind_starts:
                    y_wind = np.random.rand(1)[0]*2*wind_strength - wind_strength
                else:
                    y_wind = 0
            else:
                if random_vector[1] < prob_wind_stays:
                    y_wind = y_wind_list[-1]
                else:
                    y_wind = 0

            if not z_wind_list or z_wind_list[-1] == 0:
                if random_vector[2] < prob_wind_starts:
                    z_wind = np.random.rand(1)[0]*2*wind_strength - wind_strength
                else:
                    z_wind = 0
            else:
                if random_vector[2] < prob_wind_stays:
                    z_wind = z_wind_list[-1]
                else:
                    z_wind = 0

            x_wind_list.append(x_wind)
            y_wind_list.append(y_wind)
            z_wind_list.append(z_wind)

        
        
        V_magnitude, alpha, beta, L, D, M_moment, F_y, L_moment, N_moment = calculate_aero_forces(u,v,w, p,q,r,elevator,aileron, rudder, params)
        ##################################################################################
        

        # Translation Kinematics
        pos_earth_dot = (Rot_z(psi).transpose() @ Rot_y(theta).transpose() @ Rot_x(phi).transpose()) \
                    @ np.array([[u], [v], [w]])
        if is_distrubances:
            x_dot = pos_earth_dot[0,0] + x_wind_list[-1]
            y_dot = pos_earth_dot[1,0] + y_wind_list[-1]
            z_dot = pos_earth_dot[2,0] + z_wind_list[-1]

        else:
            x_dot = pos_earth_dot[0,0]
            y_dot = pos_earth_dot[1,0]
            z_dot = pos_earth_dot[2,0]


        




        # rotation Kinematics

        rotation_dot = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                                    [0, np.cos(phi), -np.sin(phi)],
                                    [0, np.sin(phi)*(1/np.cos(theta)), np.cos(phi)*(1/np.cos(theta))]])\
                                    @ np.array([[p], [q], [r]])
        phi_dot = rotation_dot[0,0]
        theta_dot = rotation_dot[1,0]
        psi_dot = rotation_dot[2,0]


        # Translation dynamics
        g = params['environmental']['gravity']
        mass = params['physical']['mass']

        u_dot = -g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta)/mass + L*np.sin(alpha)/mass + thrust/mass -q*w+r*v
        v_dot = g*np.cos(theta)*np.sin(phi) - D*np.sin(beta)/mass + F_y/mass -r*u+p*w
        w_dot = g*np.cos(theta)*np.cos(phi) - D*np.sin(alpha)*np.cos(beta)/mass - L*np.cos(alpha)/mass -p*v+q*u

        
        
        # rotation dynamics
        Ixx = params['physical']['Ixx']
        Iyy = params['physical']['Iyy']
        Izz = params['physical']['Izz']
        Ixz = params['physical']['Ixz']
        
        p_dot = ( -q*r*(Izz-Iyy) + N_moment*Ixz/Izz - p*q*Ixz*(Iyy-Ixx)/Izz - q*r*Ixz*Ixz/Izz +p*q*Ixz + L_moment)\
                                / (Ixx - Ixz*Ixz/Izz)
        
        q_dot = ( M_moment + p*r*(Izz-Ixx)-(p**2 - r**2)*Ixz ) / Iyy

        r_dot = ( -p*q*(Iyy-Ixx) - q*r*Ixz + L_moment*Ixz/Ixx - q*r*Ixz*(Izz-Iyy)/Ixx + p*q*Ixz*Ixz/Ixx + N_moment)\
                                / (Izz - Ixz*Ixz/Ixx)

        
        
        
        return(np.array([x_dot, y_dot, z_dot,
                         u_dot, v_dot, w_dot,
                         phi_dot, theta_dot, psi_dot,
                         p_dot, q_dot, r_dot]))

    t_span = [0, duration] 
    t_eval = np.arange(0, duration+dt/2, dt)

    def no_waypoint(t, y, waypoints, input_init, dt):
        if not waypoints:
            return False # if waypoints is empty, returns False
        else:
            return True
    
    no_waypoint_lambda = lambda t, y:no_waypoint(t, y, waypoints, input_init, dt)
    no_waypoint_lambda.terminal = True
    no_waypoint_lambda.direction = 0

    sol = solve_ivp(lambda t, y: airplane_model(t, y, waypoints, input_init, dt, controller),\
                     t_span, state_init, t_eval=t_eval, events = no_waypoint_lambda)
    # sol = solve_ivp(lambda t, y: airplane_model(t, y, waypoints, input_init, dt, controller),\
    #                  t_span, state_init, events = no_waypoint_lambda)

    state_des_list = [x_des_list, y_des_list, z_des_list, u_des_list, v_des_list, w_des_list, \
                 phi_des_list, theta_des_list, psi_des_list, p_des_list, q_des_list, r_des_list]
    inputs_list = [thrust_list, elevator_list, aileron_list, rudder_list]
    return sol, state_des_list, inputs_list, t_list, x_wind_list, y_wind_list, z_wind_list


def simulate_airplane_old(state_init, input_u, duration, dt):


    def airplane_model_old(t,state, input_u):

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

        T = input_u[0]
        elevator = input_u[1]
        aileron = input_u[2]
        rudder = input_u[3]
        #print(f"true input: thrust:{T}, elevator:{elevator}, aileron:{aileron}, rudder:{rudder}")

        ### load in stored parameters and calculate other parameters###
        with open("aerosonde_parameters.yaml", 'r') as file:
            params = yaml.safe_load(file)

        V_magnitude, alpha, beta, L, D, M_moment, F_y, L_moment, N_moment = calculate_aero_forces(u,v,w, p,q,r,elevator,aileron, rudder, params)
        ##################################################################################


        # print(f"L:{L}, D:{D}, F_y{F_y}")
        # print(f"L_moment:{L_moment}, M_moment:{M_moment}, N_moment{N_moment}")

        # Translation Kinematics
        pos_earth_dot = (Rot_z(psi).transpose() @ Rot_y(theta).transpose() @ Rot_x(phi).transpose()) \
                    @ np.array([[u], [v], [w]])
        x_dot = pos_earth_dot[0,0]
        y_dot = pos_earth_dot[1,0]
        z_dot = pos_earth_dot[2,0]


        # rotation Kinematics

        rotation_dot = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                                    [0, np.cos(phi), -np.sin(phi)],
                                    [0, np.sin(phi)*(1/np.cos(theta)), np.cos(phi)*(1/np.cos(theta))]])\
                                    @ np.array([[p], [q], [r]])
        phi_dot = rotation_dot[0,0]
        theta_dot = rotation_dot[1,0]
        psi_dot = rotation_dot[2,0]

        # Translation dynamics
        g = params['environmental']['gravity']
        mass = params['physical']['mass']

        u_dot = -g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta)/mass + L*np.sin(alpha)/mass + T/mass -q*w+r*v
        v_dot = g*np.cos(theta)*np.sin(phi) - D*np.sin(beta)/mass + F_y/mass -r*u+p*w
        w_dot = g*np.cos(theta)*np.cos(phi) - D*np.sin(alpha)*np.cos(beta)/mass - L*np.cos(alpha)/mass -p*v+q*u


        # rotation dynamics
        Ixx = params['physical']['Ixx']
        Iyy = params['physical']['Iyy']
        Izz = params['physical']['Izz']
        Ixz = params['physical']['Ixz']

        p_dot = ( -q*r*(Izz-Iyy) + N_moment*Ixz/Izz - p*q*Ixz*(Iyy-Ixx)/Izz - q*r*Ixz*Ixz/Izz +p*q*Ixz + L_moment)\
                                / (Ixx - Ixz*Ixz/Izz)

        q_dot = ( M_moment + p*r*(Izz-Ixx)-(p**2 - r**2)*Ixz ) / Iyy

        r_dot = ( -p*q*(Iyy-Ixx) - q*r*Ixz + L_moment*Ixz/Ixx - q*r*Ixz*(Izz-Iyy)/Ixx + p*q*Ixz*Ixz/Ixx + N_moment)\
                                / (Izz - Ixz*Ixz/Ixx)



        # print(f"time: {t}")
        # print(f"alpha:{alpha}, beta:{beta}, theta_dot:{theta_dot}, psi_dot:{psi_dot}")
        # print(f"u_dot:{u_dot}, v_dot:{v_dot}, w_dot:{w_dot}")
        # print(f"p_dot:{p_dot}, q_dot:{q_dot}, r_dot:{r_dot}")



        return(np.array([x_dot, y_dot, z_dot,
                         u_dot, v_dot, w_dot,
                         phi_dot, theta_dot, psi_dot,
                         p_dot, q_dot, r_dot]))

    t_span = [0, duration] 
    t_eval = np.arange(0, duration, dt)
    sol = solve_ivp(lambda t, y: airplane_model_old(t, y, input_u), t_span, state_init, t_eval=t_eval)
    return sol

# def simulate_airplane6(state_init, state_des, u_input, K_long, K_lat, duration, dt):


#     def airplane_model6(t,state, state_des, u_input, K_long, K_lat):


#         x = state[0]
#         y = state[1]
#         z = state[2]
#         u = state[3]
#         v = state[4]
#         w = state[5]

#         phi = state[6]
#         theta = state[7]
#         psi = state[8]
#         p = state[9]
#         q = state[10]
#         r = state[11]

#         long_variable = np.array([[u], [w], [q], [theta],[x], [z]])
#         lat_variable = np.array([[v], [p], [r], [phi], [psi], [y]])

#         long_state_des = np.array([[state_des[3]],[state_des[5]],[state_des[10]],[state_des[7]], [state_des[0]], [state_des[2]]])
#         lat_state_des = np.array([[state_des[4]],[state_des[9]],[state_des[11]],[state_des[6]], [state_des[8]], [state_des[1]]])


#         long_input = -K_long @ (long_variable - long_state_des)
#         lat_input = -K_lat @ (lat_variable - lat_state_des)

#         thrust = u_input[0] + long_input[0,0]
#         elevator = u_input[1] + long_input[1,0]
#         aileron = u_input[2] + lat_input[0,0]
#         rudder = u_input[3] + lat_input[1,0]

#         ### load in stored parameters and calculate other parameters###
#         with open("aerosonde_parameters.yaml", 'r') as file:
#             params = yaml.safe_load(file)
        
#         V_magnitude, alpha, beta, L, D, M_moment, F_y, L_moment, N_moment = calculate_aero_forces(u,v,w, p,q,r,elevator,aileron, rudder, params)
#         ##################################################################################
        

#         # Translation Kinematics
#         pos_earth_dot = (Rot_z(psi).transpose() @ Rot_y(theta).transpose() @ Rot_x(phi).transpose()) \
#                     @ np.array([[u], [v], [w]])
#         x_dot = pos_earth_dot[0,0]
#         y_dot = pos_earth_dot[1,0]
#         z_dot = pos_earth_dot[2,0]


#         # rotation Kinematics

#         rotation_dot = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
#                                     [0, np.cos(phi), -np.sin(phi)],
#                                     [0, np.sin(phi)*(1/np.cos(theta)), np.cos(phi)*(1/np.cos(theta))]])\
#                                     @ np.array([[p], [q], [r]])
#         phi_dot = rotation_dot[0,0]
#         theta_dot = rotation_dot[1,0]
#         psi_dot = rotation_dot[2,0]

#         # Translation dynamics
#         g = params['environmental']['gravity']
#         mass = params['physical']['mass']

#         u_dot = -g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta)/mass + L*np.sin(alpha)/mass + thrust/mass -q*w+r*v
#         v_dot = g*np.cos(theta)*np.sin(phi) - D*np.sin(beta)/mass + F_y/mass -r*u+p*w
#         w_dot = g*np.cos(theta)*np.cos(phi) - D*np.sin(alpha)*np.cos(beta)/mass - L*np.cos(alpha)/mass -p*v+q*u

        
#         # rotation dynamics
#         Ixx = params['physical']['Ixx']
#         Iyy = params['physical']['Iyy']
#         Izz = params['physical']['Izz']
#         Ixz = params['physical']['Ixz']
        
#         p_dot = ( -q*r*(Izz-Iyy) + N_moment*Ixz/Izz - p*q*Ixz*(Iyy-Ixx)/Izz - q*r*Ixz*Ixz/Izz +p*q*Ixz + L_moment)\
#                                 / (Ixx - Ixz*Ixz/Izz)
        
#         q_dot = ( M_moment + p*r*(Izz-Ixx)-(p**2 - r**2)*Ixz ) / Iyy

#         r_dot = ( -p*q*(Iyy-Ixx) - q*r*Ixz + L_moment*Ixz/Ixx - q*r*Ixz*(Izz-Iyy)/Ixx + p*q*Ixz*Ixz/Ixx + N_moment)\
#                                 / (Izz - Ixz*Ixz/Ixx)

        
        
        
#         return(np.array([x_dot, y_dot, z_dot,
#                          u_dot, v_dot, w_dot,
#                          phi_dot, theta_dot, psi_dot,
#                          p_dot, q_dot, r_dot]))

#     t_span = [0, duration] 
#     t_eval = np.arange(0, duration+dt/2, dt)
#     sol = solve_ivp(lambda t, y: airplane_model6(t, y, state_des, u_input, K_long, K_lat), t_span, state_init, t_eval=t_eval, method='Radau')
#     return sol
# cvxpy


# def simulate_airplane(state_init, state_des, u_input, K_long, K_lat, duration, dt):


#     def airplane_model(t,state, state_des, u_input, K_long, K_lat):


#         x = state[0]
#         y = state[1]
#         z = state[2]
#         u = state[3]
#         v = state[4]
#         w = state[5]

#         phi = state[6]
#         theta = state[7]
#         psi = state[8]
#         p = state[9]
#         q = state[10]
#         r = state[11]

#         long_variable = np.array([[u], [w], [q], [theta]])
#         lat_variable = np.array([[v], [p], [r], [phi], [psi]])

#         long_state_des = np.array([[state_des[3]],[state_des[5]],[state_des[10]],[state_des[7]]])
#         lat_state_des = np.array([[state_des[4]],[state_des[9]],[state_des[11]],[state_des[6]], [state_des[8]]])

#         long_input = -K_long @ (long_variable - long_state_des)
#         lat_input = -K_lat @ (lat_variable - lat_state_des)

#         thrust = u_input[0] + long_input[0,0]
#         elevator = u_input[1] + long_input[1,0]
#         aileron = u_input[2] + lat_input[0,0]
#         rudder = u_input[3] + lat_input[1,0]

#         ### load in stored parameters and calculate other parameters###
#         with open("aerosonde_parameters.yaml", 'r') as file:
#             params = yaml.safe_load(file)
        
#         V_magnitude, alpha, beta, L, D, M_moment, F_y, L_moment, N_moment = calculate_aero_forces(u,v,w, p,q,r,elevator,aileron, rudder, params)
#         ##################################################################################
        

#         # Translation Kinematics
#         pos_earth_dot = (Rot_z(psi).transpose() @ Rot_y(theta).transpose() @ Rot_x(phi).transpose()) \
#                     @ np.array([[u], [v], [w]])
#         x_dot = pos_earth_dot[0,0]
#         y_dot = pos_earth_dot[1,0]
#         z_dot = pos_earth_dot[2,0]


#         # rotation Kinematics

#         rotation_dot = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
#                                     [0, np.cos(phi), -np.sin(phi)],
#                                     [0, np.sin(phi)*(1/np.cos(theta)), np.cos(phi)*(1/np.cos(theta))]])\
#                                     @ np.array([[p], [q], [r]])
#         phi_dot = rotation_dot[0,0]
#         theta_dot = rotation_dot[1,0]
#         psi_dot = rotation_dot[2,0]

#         # Translation dynamics
#         g = params['environmental']['gravity']
#         mass = params['physical']['mass']

#         u_dot = -g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta)/mass + L*np.sin(alpha)/mass + thrust/mass -q*w+r*v
#         v_dot = g*np.cos(theta)*np.sin(phi) - D*np.sin(beta)/mass + F_y/mass -r*u+p*w
#         w_dot = g*np.cos(theta)*np.cos(phi) - D*np.sin(alpha)*np.cos(beta)/mass - L*np.cos(alpha)/mass -p*v+q*u

        
#         # rotation dynamics
#         Ixx = params['physical']['Ixx']
#         Iyy = params['physical']['Iyy']
#         Izz = params['physical']['Izz']
#         Ixz = params['physical']['Ixz']
        
#         p_dot = ( -q*r*(Izz-Iyy) + N_moment*Ixz/Izz - p*q*Ixz*(Iyy-Ixx)/Izz - q*r*Ixz*Ixz/Izz +p*q*Ixz + L_moment)\
#                                 / (Ixx - Ixz*Ixz/Izz)
        
#         q_dot = ( M_moment + p*r*(Izz-Ixx)-(p**2 - r**2)*Ixz ) / Iyy

#         r_dot = ( -p*q*(Iyy-Ixx) - q*r*Ixz + L_moment*Ixz/Ixx - q*r*Ixz*(Izz-Iyy)/Ixx + p*q*Ixz*Ixz/Ixx + N_moment)\
#                                 / (Izz - Ixz*Ixz/Ixx)

        
        
        
#         return(np.array([x_dot, y_dot, z_dot,
#                          u_dot, v_dot, w_dot,
#                          phi_dot, theta_dot, psi_dot,
#                          p_dot, q_dot, r_dot]))

#     t_span = [0, duration] 
#     t_eval = np.arange(0, duration+dt/2, dt)
#     sol = solve_ivp(lambda t, y: airplane_model(t, y, state_des, u_input, K_long, K_lat), t_span, state_init, t_eval=t_eval)
#     return sol


