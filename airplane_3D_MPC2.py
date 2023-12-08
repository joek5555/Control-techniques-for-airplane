import numpy as np
import numpy.matlib
import casadi as ca
import yaml
from airplane_3D_util import find_steady_straight_flight, \
    calculate_thrust_and_motor_speed, \
    calculate_Vin_and_motor_speed,\
    calculate_thrust,\
    calculate_K


def MPC_controller(x_init, u_equilibrium, num_steps, A_eom_discrete, B_eom_discrete, param, obstacles_shifted = None):
    dim_x = x_init.shape[0]
    dim_u = u_equilibrium.shape[0]


    # dynamics
    x_mx = ca.MX.sym('x_mx', (dim_x, num_steps+1))
    u_mx = ca.MX.sym('u_mx', (dim_u, num_steps))
    A_mx = ca.MX.sym('A_mx', (A_eom_discrete.shape[0], A_eom_discrete.shape[1]))
    B_mx = ca.MX.sym('B_mx', (B_eom_discrete.shape[0], B_eom_discrete.shape[1]))
    p = ca.MX.sym('p', (dim_x,1+dim_x+dim_u))

    x_dynamics = ca.MX.sym("x_dynamics", (dim_x, 1))
    u_dynamics = ca.MX.sym("u_dynamics", (dim_u , 1))
    A_matrix = ca.MX.sym("A_matrix", (dim_x,dim_x))
    B_matrix = ca.MX.sym("B_matrix", (dim_x, dim_u))


    x_dynamics_next = A_matrix @ x_dynamics + B_matrix @ u_dynamics
#     x_dynamics_next = ca.vertcat(
#     x_dynamics[0],
#     x_dynamics[1],
#     x_dynamics[2],
#     x_dynamics[3],
#     x_dynamics[4],
#     x_dynamics[5]
# )

    Fun_dynamics_dt = ca.Function('f_dt', [x_dynamics, u_dynamics, A_matrix, B_matrix], [x_dynamics_next])

    # dynamics constraints
    cons_dynamics = []
    for k in range(num_steps):
        Fx = Fun_dynamics_dt(x_mx[:,k], u_mx[:,k], p[:,1:1+dim_x], p[:, 1+dim_x:])
        for j in range(dim_x):
            cons_dynamics.append(x_mx[j, k+1] - Fx[j])
    ub_dynamics = np.zeros((num_steps * dim_x, 1))
    lb_dynamics = np.zeros((num_steps * dim_x, 1))

    cons_init = [x_mx[:,0] - p[:,0]]
    ub_init = np.zeros((dim_x, 1))
    lb_init = np.zeros((dim_x, 1))


    cons_state = []
    ub_state = None
    lb_state = None


    if obstacles_shifted is not None:
        if param["variables"] == "long":
            for obstacle in obstacles_shifted:
                for k in range(num_steps):
                    cons_x = ca.if_else(x_mx[4,k+1] > obstacle[0], x_mx[4,k+1] - obstacle[0] - obstacle[3], -x_mx[4,k+1] + obstacle[0] - obstacle[3])
                    cons_z = ca.if_else(x_mx[5,k+1] > obstacle[2], x_mx[5,k+1] - obstacle[2] - obstacle[3], -x_mx[5,k+1] + obstacle[2] - obstacle[3])
                    cons_state.append(cons_x)
                    cons_state.append(cons_z)
                    # if x_mx[4,k+1] > obstacle[0]:
                    #     cons_state.append(x_mx[4,k+1] - obstacle[0] - obstacle[3])
                    # else:
                    #     cons_state.append(-x_mx[4,k+1] + obstacle[0] - obstacle[3])
                    # if x_mx[5,k+1] > obstacle[0]:
                    #     cons_state.append(x_mx[5,k+1] - obstacle[2] - obstacle[3])
                    # else:
                    #     cons_state.append(-x_mx[5,k+1] + obstacle[2] - obstacle[3])

        elif param["variables"] == "lat":
            for obstacle in obstacles_shifted:
                for k in range(num_steps):
                    cons_y = ca.if_else(x_mx[5,k+1] > obstacle[1], x_mx[5,k+1] - obstacle[1] - obstacle[3], -x_mx[5,k+1] + obstacle[1] - obstacle[3])
                    cons_state.append(cons_y)
                    # if x_mx[5,k+1] > obstacle[1]:
                    #     cons_state.append(x_mx[5,k+1] - obstacle[1] - obstacle[3])
                    # else:
                    #     cons_state.append(-x_mx[5,k+1] + obstacle[1] - obstacle[3])

        ub_state = np.tile(np.array([[1e4]]), (len(cons_state), 1))
        lb_state = np.tile(np.array([[0]]), (len(cons_state), 1))              


    

    if ub_state is not None and lb_state is not None:
        cons_NLP = cons_dynamics + cons_init + cons_state
        cons_NLP = ca.vertcat(*cons_NLP)
        lb_cons = np.concatenate((lb_dynamics, lb_init, lb_state))
        ub_cons = np.concatenate((ub_dynamics, ub_init, ub_state))
    else:
        cons_NLP = cons_dynamics + cons_init
        cons_NLP = ca.vertcat(*cons_NLP)
        lb_cons = np.concatenate((lb_dynamics, lb_init))
        ub_cons = np.concatenate((ub_dynamics, ub_init))



    # define the parameters
    Q = param["Q"]  # running cost
    R = param["R"]  # control input cost
    P = param["P"] # final cost
    # cost_quadratic_matrix = calculate_cost_quadratic_matrix(Q, R, P, dim_x, dim_u, num_steps)
    # cost_quadratic_mx = ca.MX.sym("cost_quadratic_mx", (cost_quadratic_matrix.shape[0], cost_quadratic_matrix.shape[1]))
    J = 0
    for i in range(dim_x):
        J = J + x_mx[i,-1] ** 2 * P[i,i]
    for k in range(num_steps):
        for i in range(dim_x):
            J = J + x_mx[i,k] ** 2 * Q[i,i]
        for j in range(dim_u):
            J = J + u_mx[j,k]**2 * R[j,j]

    vars_NLP   = ca.vertcat(u_mx.reshape((dim_u * num_steps, 1)), x_mx.reshape((dim_x * (num_steps+1), 1)))

    prob = {"x": vars_NLP, "p":p, "f": J, "g":cons_NLP}
    opts = {'ipopt.print_level': 0, 'print_time': 0} # , 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', prob , opts)




    state_ub = np.tile(np.array([ 1e4]), dim_x)
    state_lb = np.tile(np.array([-1e4]), dim_x)
    ctrl_ub = param["upper_bound"].squeeze()
    ctrl_lb = param["lower_bound"].squeeze()

    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, num_steps + 1, 1)
    lb_x = np.matlib.repmat(state_lb, num_steps + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, num_steps, 1)
    lb_u = np.matlib.repmat(ctrl_lb, num_steps, 1)

    ub_var = np.concatenate((ub_u.reshape((dim_u * num_steps, 1)), ub_x.reshape((dim_x * (num_steps+1), 1))))
    lb_var = np.concatenate((lb_u.reshape((dim_u * num_steps, 1)), lb_x.reshape((dim_x * (num_steps+1), 1))))

    x0_nlp    = np.random.randn(vars_NLP.shape[0], 1) * 0
    lamx0_nlp = np.random.randn(vars_NLP.shape[0], 1) * 0
    lamg0_nlp = np.random.randn(cons_NLP.shape[0], 1) * 0

    par_nlp = np.hstack((x_init, A_eom_discrete, B_eom_discrete))

    sol = solver(x0=x0_nlp, lam_x0=lamx0_nlp, lam_g0=lamg0_nlp,
                     lbx=lb_var, ubx=ub_var, lbg=lb_cons, ubg=ub_cons, p = par_nlp)
        

    return sol["x"].full()[0: dim_u].reshape(-1,1) + u_equilibrium, sol["x"].full()[num_steps*dim_u:]


def calculate_dynamics_equality_matrix(A, B, num_steps):
    matrix_1_1 = np.kron(np.eye(num_steps, dtype=int), A)
    matrix_1_1 = np.concatenate((matrix_1_1, np.zeros((matrix_1_1.shape[0], A.shape[0]))), axis=1)
    matrix_1_2 = np.kron(np.diag(np.ones(num_steps), 1), -np.eye(A.shape[0]))
    matrix_1_2 = matrix_1_2[0:-A.shape[0], :]
    matrix_1 = matrix_1_1 + matrix_1_2
    matrix_2 = np.kron(np.eye(num_steps, dtype=int), B)
    matrix = np.block([[matrix_1, matrix_2]])

    return matrix


def calculate_input_constraint_matrix(dim_x, dim_u, num_steps):
    matrix_1 = np.zeros((num_steps*dim_u,dim_x*(num_steps+1)))
    matrix_2 = np.eye(dim_u*num_steps)
    matrix = np.block([[matrix_1, matrix_2]])

    return matrix

def calculate_cost_quadratic_matrix(Q, R, P, dim_x, dim_u, num_steps):
    matrix_1 = np.kron(np.eye((num_steps+1), dtype=int), Q) # diagonal block matrix with Q on the diagonal
    matrix_1[(num_steps)*Q.shape[0]: , (num_steps)*Q.shape[1]:] = P # replace the last instance of Q with P
    matrix_2 = np.kron(np.eye(num_steps, dtype=int), R) # diagonal block matrix with R on the diagonal

    matrix = np.block([[matrix_1, np.zeros(((num_steps+1)*dim_x,num_steps*dim_u))],
                    [np.zeros((num_steps*dim_u,(num_steps+1)*dim_x)), matrix_2] ])
    
    return matrix



def shift_state(state, waypoints, obstacles):
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

    long_variable = np.array([[u], [w], [q], [theta], [x], [z]])
    lat_variable = np.array([[v], [p], [r], [phi], [psi], [y]])

    reached_last_waypoint = False
    # check to see if you have hit waypoint
    if not waypoints: # if no more waypoints, do not change any of the states
        reached_last_waypoint = True
        return None, None, None, None, None, reached_last_waypoint
    
    waypoint_precision = 5
    # if you are close to a waypoint
    if np.abs(x - waypoints[0][0]) < waypoint_precision and\
        np.abs(y - waypoints[0][1]) < waypoint_precision and\
        np.abs(z - waypoints[0][2]) < waypoint_precision:

        print("Waypoint reached!")
        waypoints.pop(0)

        if not waypoints:
            print("All Waypoints Reached")
            reached_last_waypoint = True
            return None, None, None, None, None, None, reached_last_waypoint

    x_error = waypoints[0][0] - x
    y_error = waypoints[0][1] - y
    z_error = waypoints[0][2] - z

    psi_des = np.arctan2(y_error,x_error)
    earth_to_wind_angle_des = np.arctan2(-z_error, np.sqrt(x_error**2 + y_error**2))
    V_magnitude = 20

    

    u_des, w_des, theta_des, alpha_des, elevator_des, thrust_des = find_steady_straight_flight(earth_to_wind_angle_des, V_magnitude)


    long_state_des = np.array([[u_des],[w_des],[0],[theta_des], [waypoints[0][0]], [waypoints[0][2]]])
    lat_state_des = np.array([[0],[0],[0],[0], [psi_des], [waypoints[0][1]]])

    long_state_shifted = (long_variable - long_state_des )
    lat_state_shifted = (lat_variable - lat_state_des)

    # make sure the angles are kept between -pi and pi
    # if you are at -160 and you want to go to 160, you should have a 
    # desired of 40, not -320 

    if lat_state_shifted[4,0] > np.pi:
        lat_state_shifted[4,0] = -2*np.pi + lat_state_shifted[4,0]
    elif lat_state_shifted[4,0] < -np.pi:
        lat_state_shifted[4,0] = 2*np.pi + lat_state_shifted[4,0]


    long_u_steady = np.array([[thrust_des], [elevator_des]])
    lat_u_steady = np.array([[0],[0]])

    state_des = [waypoints[0][0], waypoints[0][1], waypoints[0][2],
                u_des, 0, w_des,
                0, theta_des, psi_des,
                0, 0, 0]
    
    obstacles_shifted = []
    for obstacle in obstacles:
        obstacle_shifted = np.array([obstacle[0] - waypoints[0][0],
                                     obstacle[1] - waypoints[0][1],
                                     obstacle[2] - waypoints[0][2],
                                     obstacle[3]])
        obstacles_shifted.append(obstacle_shifted)
    

    return long_state_shifted, lat_state_shifted, long_u_steady, lat_u_steady, state_des, obstacles_shifted, reached_last_waypoint

def clip_input(calculated_input, last_input, dt):
    thrust = calculated_input[0,0]
    elevator = calculated_input[1,0]
    aileron = calculated_input[2,0]
    rudder = calculated_input[3,0]

    thrust_previous = last_input[0,0]
    elevator_previous = last_input[1,0]
    aileron_previous = last_input[2,0]
    rudder_previous = last_input[3,0]

    with open("aerosonde_parameters.yaml", 'r') as file:
            params = yaml.safe_load(file)

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

    clipped_input = np.array([[thrust], [elevator], [aileron], [rudder]])
    return clipped_input


# x_des_list =[]
#     y_des_list = []
#     z_des_list = []
#     u_des_list = []
#     v_des_list = []
#     w_des_list = []
#     phi_des_list = []
#     theta_des_list =[]
#     psi_des_list = []
#     p_des_list = []
#     q_des_list = []
#     r_des_list = []
#     x_des_list.append(waypoints[0][0])
#     y_des_list.append(waypoints[0][1])
#     z_des_list.append(waypoints[0][2])
#     u_des_list.append(u_des)
#     v_des_list.append(0)
#     w_des_list.append(w_des)
#     phi_des_list.append(0)
#     theta_des_list.append(theta_des)
#     psi_des_list.append(psi_des)
#     p_des_list.append(0)
#     q_des_list.append(0)
#     r_des_list.append(0)

#     list_dict = {"x_des_list": x_des_list,
#                 "y_des_list": y_des_list,
#                 "z_des_list": z_des_list,
#                 "u_des_list": u_des_list,
#                 "v_des_list": v_des_list,
#                 "w_des_list": w_des_list,
#                 "phi_des_list": phi_des_list,
#                 "theta_des_list": theta_des_list,
#                 "psi_des_list": psi_des_list,
#                 "p_des_list": p_des_list,
#                 "q_des_list": q_des_list,
#                 "r_des_list": r_des_list,}