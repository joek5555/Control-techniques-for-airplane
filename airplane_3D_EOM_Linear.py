import numpy as np
import yaml
from scipy.linalg import expm

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

def continuous_linearization(state_eq, input_eq):

    x_0 = state_eq[0]
    y_0 = state_eq[1]
    z_0 = state_eq[2]
    u_0 = state_eq[3]
    v_0 = state_eq[4]
    w_0 = state_eq[5]

    phi_0 = state_eq[6]
    theta_0 = state_eq[7]
    psi_0 = state_eq[8]
    p_0 = state_eq[9]
    q_0 = state_eq[10]
    r_0 = state_eq[11]

    thrust_0 = input_eq[0]
    elevator_0 = input_eq[1]
    aileron_0 = input_eq[2]
    rudder_0 = input_eq[3]

    ### load in stored parameters and calculate other parameters###
    with open("aerosonde_parameters.yaml", 'r') as file:
        params = yaml.safe_load(file)

    g = params['environmental']['gravity']
    rho = params['environmental']['rho_air_density']
    S = params['physical']['Surface_area_wings']
    m = params['physical']['mass']
    Ixx = params['physical']['Ixx']
    Iyy = params['physical']['Iyy']
    Izz = params['physical']['Izz']
    Ixz = params['physical']['Ixz']
    c_wing_chord = params['physical']['c_wing_chord']
    b_wing_span =  params['physical']['b_wing_span']

    C_L_0 = params['long_coef']['C_L_0']
    C_L_alpha = params['long_coef']['C_L_alpha']
    C_L_q = params['long_coef']['C_L_q']
    C_L_elevator = params['long_coef']['C_L_elevator']

    C_D_0 = params['long_coef']['C_D_0']
    C_D_alpha = params['long_coef']['C_D_alpha']
    C_D_q = params['long_coef']['C_D_q']
    C_D_elevator = params['long_coef']['C_D_elevator']

    C_M_moment_0 = params['long_coef']['C_M_moment_0']
    C_M_moment_alpha = params['long_coef']['C_M_moment_alpha']
    C_M_moment_q = params['long_coef']['C_M_moment_q']
    C_M_moment_elevator = params['long_coef']['C_M_moment_elevator']

    C_y_0 = params['lat_coef']['C_y_0']
    C_y_beta = params['lat_coef']['C_y_beta']
    C_y_p = params['lat_coef']['C_y_p']
    C_y_r = params['lat_coef']['C_y_r']
    C_y_aileron = params['lat_coef']['C_y_aileron']
    C_y_rudder = params['lat_coef']['C_y_rudder']

    C_L_moment_0 = params['lat_coef']['C_L_moment_0']
    C_L_moment_beta = params['lat_coef']['C_L_moment_beta']
    C_L_moment_p = params['lat_coef']['C_L_moment_p']
    C_L_moment_r = params['lat_coef']['C_L_moment_r']
    C_L_moment_aileron = params['lat_coef']['C_L_moment_aileron']
    C_L_moment_rudder = params['lat_coef']['C_L_moment_rudder']

    C_N_moment_0 = params['lat_coef']['C_N_moment_0']
    C_N_moment_beta = params['lat_coef']['C_N_moment_beta']
    C_N_moment_p = params['lat_coef']['C_N_moment_p']
    C_N_moment_r = params['lat_coef']['C_N_moment_r']
    C_N_moment_aileron = params['lat_coef']['C_N_moment_aileron']
    C_N_moment_rudder = params['lat_coef']['C_N_moment_rudder']



    alpha_0 = np.arctan2(w_0,u_0)
    V_0 = np.sqrt(u_0**2 +v_0**2 + w_0**2)

    # linearized longitudial components

    a_u_u = -u_0*rho*S/m*(C_D_0 + C_D_alpha*alpha_0 + C_D_elevator*elevator_0) \
        + rho*S*w_0*C_D_alpha/(2*m) - rho*S*c_wing_chord*C_D_q*q_0* u_0/(2*m*V_0)
    a_u_w = -w_0*rho*S/m*(C_D_0 + C_D_alpha*alpha_0 + C_D_elevator*elevator_0) \
        - rho*S*u_0*C_D_alpha/(2*m) - rho*S*c_wing_chord*C_D_q*q_0* w_0/(2*m*V_0) - q_0
    a_u_q = -w_0 - rho*V_0*S*C_D_q*c_wing_chord/(4*m)
    a_u_theta = -g*np.cos(theta_0)
    b_u_elevator = -rho*V_0**2*S*C_D_elevator/(2*m)
    b_u_T = 1/m

    a_w_u = -u_0*rho*S/m*(C_L_0 + C_L_alpha*alpha_0 + C_L_elevator*elevator_0) \
        + rho*S*w_0*C_L_alpha/(2*m) - rho*S*c_wing_chord*C_L_q*q_0* u_0/(2*m*V_0) + q_0
    a_w_w = -w_0*rho*S/m*(C_L_0 + C_L_alpha*alpha_0 + C_L_elevator*elevator_0) \
        - rho*S*u_0*C_L_alpha/(2*m) - rho*S*c_wing_chord*C_L_q*q_0* w_0/(2*m*V_0) 
    a_w_q = u_0 - rho*V_0*S*C_L_q*c_wing_chord/(4*m)
    a_w_theta = -g*np.sin(theta_0)
    b_w_elevator = -rho*V_0**2*S*C_L_elevator/(2*m)
    
    a_q_u = u_0*rho*S*c_wing_chord/Iyy*(C_M_moment_0 + C_M_moment_alpha*alpha_0 +C_M_moment_elevator*elevator_0)\
            - rho*S*c_wing_chord*C_M_moment_alpha*w_0/(2*Iyy) + rho*S*c_wing_chord**2*C_M_moment_q*q_0*u_0/(2*Iyy*V_0)
    a_q_w = w_0*rho*S*c_wing_chord/Iyy*(C_M_moment_0 + C_M_moment_alpha*alpha_0 +C_M_moment_elevator*elevator_0)\
            + rho*S*c_wing_chord*C_M_moment_alpha*u_0/(2*Iyy) + rho*S*c_wing_chord**2*C_M_moment_q*q_0*w_0/(2*Iyy*V_0)
    a_q_q = rho*V_0*S*c_wing_chord**2*C_M_moment_q/(4*Iyy)
    b_q_elevator = rho*V_0**2*S*c_wing_chord*C_M_moment_elevator/(2*Iyy)

    a_theta_q = 1

    A_long4 = np.array([[a_u_u, a_u_w, a_u_q, a_u_theta],
                        [a_w_u, a_w_w, a_w_q, a_w_theta],
                        [a_q_u, a_q_w, a_q_q,     0],
                        [ 0   ,   0  , a_theta_q, 0]])
    B_long4 = np.array([[b_u_T, b_u_elevator],
                        [0    , b_w_elevator],
                        [0    , b_q_elevator],
                        [0    , 0           ]])
    
    a_x_u = np.cos(theta_0)
    a_x_w = np.sin(theta_0)
    a_x_theta = np.cos(theta_0)*w_0 - np.sin(theta_0)*u_0

    a_z_u = -np.sin(theta_0)
    a_z_w = np.cos(theta_0)
    a_z_theta = -np.cos(theta_0)*u_0 - np.sin(theta_0)*w_0

    A_long6 = np.array([[a_u_u, a_u_w, a_u_q, a_u_theta, 0, 0],
                        [a_w_u, a_w_w, a_w_q, a_w_theta, 0, 0],
                        [a_q_u, a_q_w, a_q_q,     0,     0, 0],
                        [ 0   ,   0  , a_theta_q, 0,     0, 0],
                        [a_x_u, a_x_w,    0,  a_x_theta, 0, 0],
                        [a_z_u, a_z_w,    0,  a_z_theta, 0, 0]])
    B_long6 = np.array([[b_u_T, b_u_elevator],
                        [0    , b_w_elevator],
                        [0    , b_q_elevator],
                        [0    , 0           ],
                        [0    , 0           ],
                        [0    , 0           ]])
    #################################################
    
    beta_0 = np.arcsin(v_0/V_0)
    a_v_v = rho*S*b_wing_span*v_0/(2*m*V_0)*(C_y_p * p_0 + C_y_r * r_0) \
            + rho*S*v_0/m*(C_y_0 + C_y_beta*beta_0 + C_y_aileron*aileron_0 + C_y_rudder * rudder_0)\
            + rho*S*C_y_beta/(2*m) * np.sqrt(u_0**2 + w_0**2)
    a_v_p = w_0 + rho*V_0*S*b_wing_span*C_y_p/(4*m)
    a_v_r = -u_0 + rho*V_0*S*b_wing_span*C_y_r/(4*m)
    a_v_phi = g*np.cos(theta_0)*np.cos(phi_0)
    b_v_aileron = rho*V_0**2*S*C_y_aileron/(2*m)
    b_v_rudder = rho*V_0**2*S*C_y_rudder/(2*m)

    a_p_v = rho*S*b_wing_span**2*v_0/(2*V_0)*(C_L_moment_p * p_0 + C_L_moment_r * r_0) \
            + rho*S*b_wing_span*v_0*(C_L_moment_0 + C_L_moment_beta * beta_0 + C_L_moment_aileron * aileron_0 + C_L_moment_rudder * rudder_0) \
            + rho*S*b_wing_span*C_L_moment_beta/2*np.sqrt(u_0**2+w_0**2)
    a_p_p = Ixz*(Ixx-Iyy+Izz)/(Ixx*Izz-Ixz*Ixz)*q_0 + rho*V_0*S*b_wing_span**2*C_L_moment_p/4
    a_p_r = -(Izz*(Izz-Iyy)+Ixz*Ixz)/(Ixx*Izz-Ixz*Ixz)*q_0 + rho*V_0*S*b_wing_span**2*C_L_moment_r/4
    b_p_aileron = rho*V_0**2*S*b_wing_span*C_L_moment_aileron/2
    b_p_rudder = rho*V_0**2*S*b_wing_span*C_L_moment_rudder/2

    a_r_v = rho*S*b_wing_span**2*v_0/(2*V_0)*(C_N_moment_p * p_0 + C_N_moment_r * r_0) \
            + rho*S*b_wing_span*v_0*(C_N_moment_0 + C_N_moment_beta*beta_0 + C_N_moment_aileron*aileron_0 +C_N_moment_rudder*rudder_0) \
            + rho*S*b_wing_span*C_N_moment_beta/2*np.sqrt(u_0**2 + w_0**2)
    a_r_p = ((Ixx-Iyy)*Ixx+Ixz*Ixz)/(Ixx*Izz-Ixz*Ixz)*q_0 + rho*V_0*S*b_wing_span**2*C_N_moment_p/4
    a_r_r = -Ixz*(Ixx-Iyy+Izz)/(Ixx*Izz-Ixz*Ixz)*q_0 + rho*V_0*S*b_wing_span**2*C_N_moment_r/4
    b_r_aileron = rho*V_0**2*S*b_wing_span*C_N_moment_aileron/2
    b_r_rudder = rho*V_0**2*S*b_wing_span*C_N_moment_rudder/2

    a_phi_p = 1
    a_phi_r = np.cos(phi_0)*np.tan(theta_0)
    a_phi_phi = q_0*np.cos(phi_0)*np.tan(theta_0)-r_0*np.sin(phi_0)*np.tan(theta_0)

    # A_lat4 = np.array([[a_v_v, a_v_p, a_v_r, a_v_phi],
    #                     [a_p_v, a_p_p, a_p_r, 0],
    #                     [a_r_v, a_r_p, a_r_r, 0],
    #                     [ 0, a_phi_p , a_phi_r, a_phi_phi]])
    # B_lat4 = np.array([[b_v_aileron, b_v_rudder],
    #                     [b_p_aileron, b_p_rudder],
    #                     [b_r_aileron, b_r_rudder],
    #                     [0    , 0           ]])
    
    
    a_psi_r = 1/np.cos(theta_0)*np.cos(phi_0)
    a_psi_phi = p_0*np.cos(phi_0)*(1/np.cos(theta_0)) - r_0*np.sin(phi_0)*(1/np.cos(theta_0))

    A_lat5 = np.array([[a_v_v, a_v_p, a_v_r, a_v_phi,      0],
                        [a_p_v, a_p_p, a_p_r, 0,           0],
                        [a_r_v, a_r_p, a_r_r, 0,           0],
                        [ 0, a_phi_p , a_phi_r, a_phi_phi, 0],
                        [0,     0,     a_psi_r, a_psi_phi, 0]])
    B_lat5 = np.array([[b_v_aileron, b_v_rudder],
                        [b_p_aileron, b_p_rudder],
                        [b_r_aileron, b_r_rudder],
                        [0,            0        ],
                        [0,            0        ]])

    a_y_v = 1
    a_y_psi = u_0*np.cos(theta_0)


    A_lat6 = np.array([[a_v_v, a_v_p, a_v_r, a_v_phi,      0, 0],
                        [a_p_v, a_p_p, a_p_r, 0,           0, 0],
                        [a_r_v, a_r_p, a_r_r, 0,           0, 0],
                        [ 0, a_phi_p , a_phi_r, a_phi_phi, 0, 0],
                        [0,     0,     a_psi_r, a_psi_phi, 0, 0],
                        [a_y_v, 0,     0,         0, a_y_psi, 0]])
    B_lat6 = np.array([[b_v_aileron, b_v_rudder],
                        [b_p_aileron, b_p_rudder],
                        [b_r_aileron, b_r_rudder],
                        [0,            0        ],
                        [0,            0        ],
                        [0,            0        ]])

    


    
 
    return(A_long4, B_long4, A_long6, B_long6, A_lat5, B_lat5, A_lat6, B_lat6)

    #return(A_lat4, B_lat4, A_lat5, B_lat5, A_lat6, B_lat6, A_long4, B_long4, A_long5, B_long5, A_long6, B_long6)


def A_matrix_evalation(A, dt):
    A_continuous = A
    A_discrete = expm(A*dt)

    continuous_eigenvalues, _ = np.linalg.eig(A_continuous)

    # Compute discretized eigenvalues
    discretized_eigenvalues, _ = np.linalg.eig(A_discrete)

    return continuous_eigenvalues, discretized_eigenvalues

def continous_to_discrete(A,B, dt):
    A_dis = np.eye(A.shape[0]) + dt*A
    B_dis = dt*B
    return A_dis, B_dis



def discrete_simulation(state_eq, input_eq, state_init, input_init, dt, duration):
    # A_discrete = np.eye(A.shape[0]) + dt*A
    # B_discrete = dt*B

    num_steps = int(duration/dt)
    state_history = state_init
    delta_state_long = np.array([[state_init[3,0]], [state_init[5,0]], [state_init[10,0]], [state_init[7,0]], [state_init[0,0]], [state_init[2,0]]]) \
                    -np.array([[state_eq[3,0]], [state_eq[5,0]], [state_eq[10,0]], [state_eq[7,0]], [state_eq[0,0]], [state_eq[2,0]]])
    delta_state_lat = np.array([[state_init[4,0]], [state_init[9,0]], [state_init[11,0]], [state_init[6,0]], [state_init[8,0]], [state_init[1,0]]]) \
                    - np.array([[state_eq[4,0]], [state_eq[9,0]], [state_eq[11,0]], [state_eq[6,0]], [state_eq[8,0]], [state_eq[1,0]]])
    delta_input = input_init - input_eq

    body_velocity_init = np.array([[state_eq[3,0]], [state_eq[4,0]], [state_eq[5,0]]])
    earth_velocity_init = Rot_z(state_eq[8,0]).T @ Rot_y(state_eq[7,0]).T @ Rot_x(state_eq[6,0]).T @ body_velocity_init
    x_velocity = earth_velocity_init[0,0]
    y_velocity = earth_velocity_init[1,0]
    z_velocity = earth_velocity_init[2,0]


    A_long4, B_long4, A_long6, B_long6, A_lat4, B_lat4, A_lat6, B_lat6 = continuous_linearization(state_eq.squeeze(), input_eq.squeeze())
    


    for k in range (num_steps):
        current_state = state_history[:,[-1]]
        # state_eq = np.array([[0], [0], [0], [current_state[0,0]], [0], [current_state[1,0]],
        #             [0], [current_state[3,0]], [0], [0], [current_state[2,0]], [0]])
        #input_eq = np.array([[input_init[0,0]], [input_init[1,0]], [0], [0]])

        
        A_long_dis = np.eye(A_long6.shape[0]) + dt*A_long6
        B_long_dis = dt*B_long6
        A_lat_dis = np.eye(A_lat6.shape[0]) + dt*A_lat6
        B_lat_dis = dt*B_lat6
        delta_state_long = A_long_dis @ delta_state_long + B_long_dis @ np.array([[delta_input[0,0]], [delta_input[1,0]]])
        delta_state_lat = A_lat_dis @ delta_state_lat + B_lat_dis @ np.array([[delta_input[2,0]], [delta_input[3,0]]])


        current_state = current_state + np.array([[delta_state_long[4,0]], [delta_state_lat[5,0]], [delta_state_long[5,0]],
                                                  [delta_state_long[0,0]], [delta_state_lat[0,0]], [delta_state_long[1,0]],
                                                  [delta_state_lat[3,0]], [delta_state_long[3,0]], [delta_state_lat[4,0]],
                                                  [delta_state_lat[1,0]], [delta_state_long[2,0]], [delta_state_lat[2,0]]])
        current_state[0,0] +=x_velocity*dt
        current_state[1,0] +=y_velocity*dt
        current_state[2,0] +=z_velocity*dt
        state_history = np.concatenate([state_history, current_state], axis=1)
        delta_input = np.array([[0],[0], [0], [0]])



    return state_history




##################################3333

def discrete_simulation2(state_init, input_init, input_new, dt, duration):
    # A_discrete = np.eye(A.shape[0]) + dt*A
    # B_discrete = dt*B

    num_steps = int(duration/dt)
    state_history = state_init
    delta_state_long = np.array([[0],[0],[0],[0], [0], [0]])
    delta_state_lat = np.array([[0],[0],[0],[0], [0], [0]])
    delta_input = input_new - input_init


    


    for k in range (num_steps):
        current_state = state_history[:,[-1]]
        # state_eq = np.array([[0], [0], [0], [current_state[0,0]], [0], [current_state[1,0]],
        #             [0], [current_state[3,0]], [0], [0], [current_state[2,0]], [0]])
        #input_eq = np.array([[input_init[0,0]], [input_init[1,0]], [0], [0]])

        body_velocity_init = np.array([[current_state[3,0]], [current_state[4,0]], [current_state[5,0]]])
        earth_velocity_init = Rot_z(current_state[8,0]).T @ Rot_y(current_state[7,0]).T @ Rot_x(current_state[6,0]).T @ body_velocity_init
        x_velocity = earth_velocity_init[0,0]
        y_velocity = earth_velocity_init[1,0]
        z_velocity = earth_velocity_init[2,0]


        A_long4, B_long4, A_long6, B_long6, A_lat4, B_lat4, A_lat6, B_lat6 = continuous_linearization(current_state.squeeze(), input_init.squeeze())
        A_long_dis = np.eye(A_long6.shape[0]) + dt*A_long6
        B_long_dis = dt*B_long6
        A_lat_dis = np.eye(A_lat6.shape[0]) + dt*A_lat6
        B_lat_dis = dt*B_lat6
        delta_state_long = A_long_dis @ delta_state_long + B_long_dis @ np.array([[delta_input[0,0]], [delta_input[1,0]]])
        delta_state_lat = A_lat_dis @ delta_state_lat + B_lat_dis @ np.array([[delta_input[2,0]], [delta_input[3,0]]])


        current_state = current_state + np.array([[delta_state_long[4,0]], [delta_state_lat[5,0]], [delta_state_long[5,0]],
                                                  [delta_state_long[0,0]], [delta_state_lat[0,0]], [delta_state_long[1,0]],
                                                  [delta_state_lat[3,0]], [delta_state_long[3,0]], [delta_state_lat[4,0]],
                                                  [delta_state_lat[1,0]], [delta_state_long[2,0]], [delta_state_lat[2,0]]])
        current_state[0,0] +=x_velocity*dt
        current_state[1,0] +=y_velocity*dt
        current_state[2,0] +=z_velocity*dt
        state_history = np.concatenate([state_history, current_state], axis=1)
        delta_input = np.array([[0],[0], [0], [0]])



    return state_history
