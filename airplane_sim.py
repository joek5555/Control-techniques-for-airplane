
import vpython
import numpy as np
import yaml


solution_data = np.load("sol_closed.npy")
predicted_path = np.load("predicted_path.npy")

#solution_data = np.load("project_code/airplane/sol_nonlinear.npy")

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


with open("aerosonde_parameters.yaml", 'r') as file:
            params = yaml.safe_load(file)


#width=1600, height=800
scene = vpython.canvas(title='Aerosonde Simulation',
     width=1200, height=600,
     center=vpython.vector(0,0,0), background=vpython.color.cyan)

scene.camera.pos = vpython.vector(-11.215, 59.1165, 38.8996)
scene.camera.axis = vpython.vector(19.3377, -34.1684, -56.4564)

arrow_length = 0.5
arrow_thickness = 0.05
ground_size_length = 1000
ground_size_width = 250
ground = vpython.box(color=vpython.color.green, length=ground_size_length, height= ground_size_width, width=arrow_thickness/4)
x_arrow_e = vpython.arrow(axis = vpython.vector(1,0,0), color=vpython.color.red, length=arrow_length, shaftwidth=arrow_thickness)
y_arrow_e = vpython.arrow(axis = vpython.vector(0,1,0), color=vpython.color.blue, length=-arrow_length, shaftwidth=arrow_thickness)
z_arrow_e = vpython.arrow(axis = vpython.vector(0,0,1), color=vpython.color.green, length=-arrow_length, shaftwidth=arrow_thickness)


wing_length = params['physical']['c_wing_chord']*2 
wing_height = params['physical']['b_wing_span']
wing_width = params['physical']['b_wing_span']/20

motor_length = params['physical']['c_wing_chord']*4 
motor_radius =params['physical']['b_wing_span']/10

back_length = wing_height/1.5
back_height = wing_height/20
back_width = wing_height/20

back_displacement_y = wing_height/5

rear_angle = np.pi/6
rear_length = wing_length
rear_height = back_displacement_y/np.cos(rear_angle)
rear_width = back_width/2

wings = vpython.box(color=vpython.color.black, length= wing_length, \
                    height = wing_height, width = wing_width, make_trail = True)

# motor = vpython.cylinder(color=vpython.color.white, length=motor_length,\
#                         radius = motor_radius)
# motor_displacement = np.array([[-wing_length],[0], [wing_height/10]])

motor = vpython.box(color=vpython.color.white, length=motor_length*1.25,\
                        height = motor_radius*1.5, width = motor_radius*1.5)
motor_displacement = np.array([[wing_length/2],[0], [wing_height/10]])

back_left = vpython.box(color=vpython.color.black, length=back_length , \
                    height = back_height, width = back_width)
back_left_displacement = np.array([[-wing_height/3],[-back_displacement_y], [wing_height/20]])

back_right = vpython.box(color=vpython.color.black, length=back_length , \
                    height = back_height, width = back_width)
back_right_displacement = np.array([[-wing_height/3],[back_displacement_y], [wing_height/20]])

rear_left = vpython.box(color=vpython.color.blue, length=rear_length , \
                    height = rear_height, width = rear_width)
rear_left.rotate(axis=vpython.vector(1,0,0), angle=-rear_angle)
rear_left_displacement = np.array([[-back_length],[-back_displacement_y/2], [0]])

rear_right = vpython.box(color=vpython.color.red, length=rear_length , \
                    height = rear_height, width = rear_width)
rear_right.rotate(axis=vpython.vector(1,0,0), angle=rear_angle)
rear_right_displacement = np.array([[-back_length],[back_displacement_y/2], [0]])

rotate_world_angle = -np.pi/2
rotate_airplane_angle = np.pi + rotate_world_angle 



world = [ground, x_arrow_e, y_arrow_e, z_arrow_e]
for item in world:
      item.rotate(axis=vpython.vector(1,0,0), angle=rotate_world_angle)

airplane_sections = [wings, motor, back_left, back_right, rear_left, rear_right]
airplane_displacements = [np.array([[0],[0],[0]]), motor_displacement, back_left_displacement, back_right_displacement, \
                          rear_left_displacement, rear_right_displacement]
airplane_lengths = [wing_length, motor_length, back_length, back_length, rear_length, rear_length]
airplane_widths = [wing_width, motor_radius*1.5, back_width, back_width, rear_width, rear_width]
airplane_heights= [wing_height, motor_radius*1.5, back_height, back_height, rear_height, rear_height]
airplane_angle = [0,0,0,0, -rear_angle, rear_angle]






airplane_pos = np.array([[solution_data[0,0]], [solution_data[1,0]], [solution_data[2,0]]])
airplane_rot = vpython.vector(solution_data[6,0], solution_data[7,0], solution_data[8,0])


for count, airplane_part in enumerate(airplane_sections):


    axis_vec_earth = vpython.vector(np.cos(airplane_rot.z)*np.cos(airplane_rot.y), np.sin(airplane_rot.y), \
        np.sin(airplane_rot.z) * np.cos(airplane_rot.y))
    y_vec = vpython.vector(0,1,0)
    s_vec = vpython.cross(axis_vec_earth, y_vec)
    v_vec = vpython.cross(s_vec, axis_vec_earth)
    up_vec_earth = v_vec * vpython.cos(airplane_rot.x+rotate_airplane_angle + airplane_angle[count]) + \
        vpython.cross(axis_vec_earth, v_vec) * vpython.sin(airplane_rot.x+rotate_airplane_angle + airplane_angle[count])

    airplane_part.axis = axis_vec_earth
    airplane_part.up = up_vec_earth

    airplane_part.length = airplane_lengths[count]
    airplane_part.height = airplane_heights[count]
    airplane_part.width = airplane_widths[count]


    displacement =  np.transpose(Rot_z(airplane_rot.z)) @ np.transpose(Rot_y(airplane_rot.y)) @ np.transpose(Rot_x(airplane_rot.x))@ airplane_displacements[count]
    displacement =  np.transpose(Rot_x(rotate_airplane_angle)) @ displacement
    position = np.transpose(Rot_x(rotate_airplane_angle)) @ airplane_pos
    airplane_part.pos=vpython.vector(position[0,0], position[1,0], position[2,0])+ vpython.vector(displacement[0,0], displacement[1,0], displacement[2,0])
    

path_pos = np.transpose(Rot_x(rotate_airplane_angle)) @ airplane_pos
# desired_path = vpython.box(pos = vpython.vector(path_pos[0,0],path_pos[1,0],path_pos[2,0]),\
#                     color=vpython.color.white, length= 500, \
#                     height = 0.5, width = 0.5)
# desired_path = vpython.arrow(pos = vpython.vector(path_pos[0,0],path_pos[1,0],path_pos[2,0]),\
#                     color=vpython.color.white, vector= (1,0,0), \
#                     length = 373.4, shaftwidth=0.5, headwidth = 0.5, headlenght = 0.5)

# desired_path.rotate(axis=vpython.vector(1,0,0), angle=rotate_world_angle)

# desired_path.rotate(axis=vpython.vector(0,0,1), angle=0.3272)
# desired_path.rotate(axis=vpython.vector(0,1,0), angle=-np.pi/4)

# way1 = np.transpose(Rot_x(rotate_airplane_angle)) @np.array([50, 50, -120])
# way2 = np.transpose(Rot_x(rotate_airplane_angle)) @np.array([100, 50, -100])
# way3 = np.transpose(Rot_x(rotate_airplane_angle)) @np.array([50, 50, -80])
# waypoints = [way1, way2, way3]


# wave
way1 = np.array([100, 100, -100])
way2 = np.array([175, 150, -120])
way3 = np.array([250, 150, -140])
way4 = np.array([325, 100, -110])
way5 = np.array([400, 50, -90])
way6 = np.array([475, 50, -120])
way7 = np.array([550, 100, -150])

waypoints = [way1, way2, way3, way4, way5, way6, way7]

# helix
# way1 = np.array([50*2 ,-14*2,  -100])
# way2 = np.array([86*2,-50*2,  -120])
# way3 = np.array([100*2,-100*2,  -140])
# way4 = np.array([86*2,-150*2,  -160])
# way5 = np.array([50*2,-186*2,  -180])
# way6 = np.array([0*2, -200*2,  -200])
# way7 = np.array([-50*2, -186*2,  -220])
# way8 = np.array([-86*2, -150*2, -240])
# way9 = np.array([-100*2, -100*2, -260])
# way10 = np.array([-86*2, -50*2, -280])
# way11 = np.array([-50*2, -14*2, -300])
# way12 = np.array([-0, -0, -320])


# waypoints = [way1, way2, way3, way4, way5, way6, way7, way8, way9, way10, way11, way12]
# way1 = np.array([50 ,-14,  -100])
# way2 = np.array([86,-50,  -120])
# way3 = np.array([100,-100,  -140])
# way4 = np.array([86,-150,  -160])
# way5 = np.array([50,-186,  -180])
# way6 = np.array([0, -200,  -200])
# way7 = np.array([-50, -186,  -220])
# way8 = np.array([-86, -150, -240])
# way9 = np.array([-100, -100, -260])
# way10 = np.array([-86, -50, -280])
# way11 = np.array([-50, -14, -300])
# way12 = np.array([-0, -0, -320])

# waypoints = [way1, way2, way3, way4, way5, way6, way7, way8, way9, way10, way11, way12]

waypoint_spheres = []

waypoint_precision = 5

for waypoint in waypoints:
    waypoint = np.transpose(Rot_x(rotate_airplane_angle)) @ waypoint

    waypoint_spheres.append(vpython.sphere(pos=vpython.vector(waypoint[0],waypoint[1],waypoint[2]), radius = waypoint_precision, \
                           color = vpython.color.magenta, opacity = 0.50))

obstacle1 = np.array([60, 40, -100, 10])
obstacles = [obstacle1]
obstacle_spheres = []

for obstacle in obstacles:
    obstacle_pos = np.transpose(Rot_x(rotate_airplane_angle)) @ obstacle[0:3]

    obstacle_spheres.append(vpython.sphere(pos=vpython.vector(obstacle_pos[0],obstacle_pos[1],obstacle_pos[2]), radius = obstacle[3], \
                           color = vpython.color.red, opacity = 0.50))

scene.camera.follow(wings)
vpython.sleep(1)

#while True:
for i in range(1, solution_data.shape[1]):
    # vpython.rate(20)
    vpython.rate(10) # 2 times speed

    #vpython.rate(500)
    airplane_pos = np.array([[solution_data[0,i]], [solution_data[1,i]], [solution_data[2,i]]])
    airplane_rot = vpython.vector(solution_data[6,i], solution_data[7,i], solution_data[8,i])

    predicted_path_iter = predicted_path[i-1,:,:].tolist()
    predicted_path_vectors = []
    for predicted_path_xyz in predicted_path_iter:
        predicted_path_xyz = np.transpose(Rot_x(rotate_airplane_angle)) @ predicted_path_xyz
        predicted_path_vectors.append(vpython.vector(predicted_path_xyz[0], predicted_path_xyz[1], predicted_path_xyz[2]))
    
    predicted_path_drawn = vpython.curve(predicted_path_vectors)
    print(len(predicted_path_vectors))
    print(predicted_path_vectors[0])
         

    for count, airplane_part in enumerate(airplane_sections):

        axis_vec_earth = vpython.vector(np.cos(airplane_rot.z)*np.cos(airplane_rot.y), np.sin(airplane_rot.y), \
            np.sin(airplane_rot.z) * np.cos(airplane_rot.y))
        y_vec = vpython.vector(0,1,0)
        s_vec = vpython.cross(axis_vec_earth, y_vec)
        v_vec = vpython.cross(s_vec, axis_vec_earth)
        up_vec_earth = v_vec * vpython.cos(airplane_rot.x + rotate_airplane_angle + airplane_angle[count]) + \
            vpython.cross(axis_vec_earth, v_vec) * vpython.sin(airplane_rot.x + rotate_airplane_angle + airplane_angle[count])

        airplane_part.axis = axis_vec_earth
        airplane_part.up = up_vec_earth

        airplane_part.length = airplane_lengths[count]
        airplane_part.height = airplane_heights[count]
        airplane_part.width = airplane_widths[count]


        displacement =  np.transpose(Rot_z(airplane_rot.z)) @ np.transpose(Rot_y(airplane_rot.y)) @ np.transpose(Rot_x(airplane_rot.x))@ airplane_displacements[count]
        displacement =  np.transpose(Rot_x(rotate_airplane_angle)) @ displacement
        position = np.transpose(Rot_x(rotate_airplane_angle)) @ airplane_pos
        airplane_part.pos=vpython.vector(position[0,0], position[1,0], position[2,0])+ vpython.vector(displacement[0,0], displacement[1,0], displacement[2,0])

    if i == 1:
            wings.clear_trail()

scene.camera.follow(None) 
    
    # vec = np.array([[0],[1],[0]])
    # airplane_part_up = vec*np.cos(airplane_rot.x+rotate_airplane_angle) + \
    #     airplane_part_axis*(np.dot(airplane_part_axis.squeeze(), vec.squeeze())) * (1-np.cos(airplane_rot.x+rotate_airplane_angle))

    # hi = np.cross(airplane_part_axis.squeeze(), vec.squeeze()) 



        # arrow_position[0,0] = arrow_position[0,0] + 1
        # arrow.pos.x = arrow_position[0,0]

        

        # # airplane_pos[0,0]= airplane_pos[0,0] + 1
        # airplane_rot.x = airplane_rot.x + np.pi/4

        # for count, airplane_part in enumerate(airplane_sections):
        
        #     airplane_part.rotate(axis=vpython.vector(0,0,1), angle=airplane_rot.z)
        #     airplane_part.rotate(axis=vpython.vector(0,1,0), angle=airplane_rot.y)
        #     airplane_part.rotate(axis=vpython.vector(1,0,0), angle=airplane_rot.x+rotate_airplane_angle)

        #     displacement =  np.transpose(Rot_z(airplane_rot.z)) @ np.transpose(Rot_y(airplane_rot.y)) @ np.transpose(Rot_x(airplane_rot.x))@ airplane_displacements[count]
        #     displacement =  np.transpose(Rot_x(rotate_airplane_angle)) @ displacement
        #     position = np.transpose(Rot_x(rotate_airplane_angle)) @ airplane_pos
        #     airplane_part.pos=vpython.vector(position[0,0], position[1,0], position[2,0])+ vpython.vector(displacement[0,0], displacement[1,0], displacement[2,0])
            

        
        # airplane_sections = [wings, motor, back_left, back_right, rear_left, rear_right]
        # airplane_displacements = [np.array([[0],[0],[0]]), motor_displacement, back_left_displacement, back_right_displacement, \
        #                         rear_left_displacement, rear_right_displacement]
        # for count, airplane_part in enumerate(airplane_sections):

            
        #     airplane_part.rotate(axis=vpython.vector(0,0,1), angle=airplane_rot.z)
        #     airplane_part.rotate(axis=vpython.vector(0,1,0), angle=airplane_rot.y)
        #     airplane_part.rotate(axis=vpython.vector(1,0,0), angle=airplane_rot.x+rotate_airplane_angle)

        #     displacement =  np.transpose(Rot_z(airplane_rot.z)) @ np.transpose(Rot_y(airplane_rot.y)) @ np.transpose(Rot_x(airplane_rot.x))@ airplane_displacements[count]
        #     displacement =  np.transpose(Rot_x(rotate_airplane_angle)) @ displacement
        #     position = np.transpose(Rot_x(rotate_airplane_angle)) @ airplane_pos
        #     airplane_part.pos=vpython.vector(position[0,0], position[1,0], position[2,0])+ vpython.vector(displacement[0,0], displacement[1,0], displacement[2,0])

        # print(rear_right.axis)



# def MPC_controller(state, input_init, dt):
#     Q_lat = 
#     R_lat= 