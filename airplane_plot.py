
import numpy as np
import matplotlib.pyplot as plt

def plot_airplane_data(sol):
    num_ticks = 5

    # Translation kinematics
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 3, 1)
    plt.plot(sol.t, sol.y[0], label='x')
    plt.xlabel('Time')
    plt.ylabel('x')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 2)
    plt.plot(sol.t, sol.y[1], label='y', color='orange')
    plt.xlabel('Time')
    plt.ylabel('y')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 3)
    plt.plot(sol.t, sol.y[2], label='z', color='green')
    plt.xlabel('Time')
    plt.ylabel('z')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    # Translation dynamics

    plt.subplot(4, 3, 4)
    plt.plot(sol.t, sol.y[3], label='u', color='red')
    plt.xlabel('Time')
    plt.ylabel('u')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 5)
    plt.plot(sol.t, sol.y[4], label='v', color='red')
    plt.xlabel('Time')
    plt.ylabel('v')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 6)
    plt.plot(sol.t, sol.y[5], label='w', color='red')
    plt.xlabel('Time')
    plt.ylabel('w')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")
    
    # Rotational kinematics

    plt.subplot(4, 3, 7)
    plt.plot(sol.t, sol.y[6], label='phi')
    plt.xlabel('Time')
    plt.ylabel('phi')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 8)
    plt.plot(sol.t, sol.y[7], label='theta', color='orange')
    plt.xlabel('Time')
    plt.ylabel('theta')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 9)
    plt.plot(sol.t, sol.y[8], label='psi', color='green')
    plt.xlabel('Time')
    plt.ylabel('psi')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    # Rotational Dynamics

    plt.subplot(4, 3, 10)
    plt.plot(sol.t, sol.y[9], label='p', color='red')
    plt.xlabel('Time')
    plt.ylabel('p')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 11)
    plt.plot(sol.t, sol.y[10], label='q', color='red')
    plt.xlabel('Time')
    plt.ylabel('q')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 12)
    plt.plot(sol.t, sol.y[11], label='r', color='red')
    plt.xlabel('Time')
    plt.ylabel('r')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()


    plt.tight_layout()
    plt.show()





    ##################################################3333
def plot_airplane_compare_data(sol_nonlinear, sol_linear):
    num_ticks = 5

    # Translation kinematics
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 3, 1)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[0],'b-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[0],'b--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('x')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 2)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[1],'g-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[1],'g--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('y')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 3)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2],'r-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[2],'r--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('z')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    # Translation dynamics

    plt.subplot(4, 3, 4)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[3],'c-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[3],'c--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('u')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 5)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[4],'m-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[4],'m--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('v')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")

    plt.subplot(4, 3, 6)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[5],'y-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[5],'y--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('w')
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.legend()
    plt.yscale("linear")
    
    # Rotational kinematics

    plt.subplot(4, 3, 7)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[6],'b-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[6],'b--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('phi')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 8)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[7],'g-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[7],'g--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('theta')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 9)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[8],'r-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[8],'r--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('psi')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    # Rotational Dynamics

    plt.subplot(4, 3, 10)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[9],'c-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[9],'c--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('p')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 11)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[10],'m-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[10],'m--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('q')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()

    plt.subplot(4, 3, 12)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[11],'y-', label='non')
    plt.plot(sol_nonlinear.t, sol_linear.y[11],'y--', label='lin')
    plt.xlabel('Time')
    plt.ylabel('r')
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/4
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.legend()


    plt.tight_layout()
    plt.show()


    ####################3
def plot_airplane_long(sol, state_des=[[],[],[],[],[],[],[],[],[],[],[],[]], t_list_des=[]):
    num_ticks = 5

    # Translation kinematics
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 3, 1)
    plt.plot(sol.t, sol.y[0], 'b')
    plt.plot(t_list_des, state_des[0], 'b--')
    plt.title("x vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")

    plt.subplot(2, 3, 2)
    plt.plot(sol.t, sol.y[2], 'r')
    plt.plot(t_list_des, state_des[2], 'r--')
    plt.title("z vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100+0.01
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100-0.01
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")

    # Translation dynamics

    plt.subplot(2, 3, 3)
    plt.plot(sol.t, sol.y[3], 'c')
    plt.plot(t_list_des, state_des[3], 'c--')
    plt.title("u vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")

    plt.subplot(2, 3, 4)
    plt.plot(sol.t, sol.y[5], 'y')
    plt.plot(t_list_des, state_des[5], 'y--')
    plt.title("w vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")
    
    # Rotational kinematics


    plt.subplot(2, 3, 5)
    plt.plot(sol.t, sol.y[7], 'g')
    plt.plot(t_list_des, state_des[7], 'g--')
    plt.title("theta vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])



    # Rotational Dynamics


    plt.subplot(2, 3, 6)
    plt.plot(sol.t, sol.y[10], 'm')
    plt.plot(t_list_des, state_des[10], 'm--')
    plt.title("q vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])



    plt.tight_layout()
    plt.show()


def plot_airplane_lat(sol, state_des=[[],[],[],[],[],[],[],[],[],[],[],[]], t_list_des=[]):
    num_ticks = 5

    # Translation kinematics

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 3, 1)
    plt.plot(sol.t, sol.y[1], 'g')
    plt.plot(t_list_des, state_des[1], 'g--')
    plt.title("y vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")



    # Translation dynamics



    plt.subplot(2, 3, 2)
    plt.plot(sol.t, sol.y[4], 'm')
    plt.plot(t_list_des, state_des[4], 'm--')
    plt.title("v vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.1:
        upper_lim = np.ceil(plt.yticks()[0][-1]*10)/10
        lower_lim = np.floor(plt.yticks()[0][0]*10)/10
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")


    
    # Rotational kinematics

    plt.subplot(2, 3, 3)
    plt.plot(sol.t, sol.y[6], 'b')
    plt.plot(t_list_des, state_des[6], 'b--')
    plt.title("phi vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])



    plt.subplot(2, 3, 4)
    plt.plot(sol.t, sol.y[8], 'r')
    plt.plot(t_list_des, state_des[8], 'r--')
    plt.title("psi vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])


    # Rotational Dynamics

    plt.subplot(2, 3, 5)
    plt.plot(sol.t, sol.y[9], 'c')
    plt.plot(t_list_des, state_des[9], 'c--')
    plt.title("p vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])



    plt.subplot(2, 3, 6)
    plt.plot(sol.t, sol.y[11], 'y')
    plt.plot(t_list_des, state_des[11], 'y--')
    plt.title("r vs time", fontsize = 18)
    step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    if step_size == 0:
        step_size = 1/8
    start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
            labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])



    plt.tight_layout()
    plt.show()




def plot_airplane_inputs(inputs, time):
    num_ticks = 3

    # Translation kinematics

    plt.figure(figsize=(6, 4))
    plt.subplot(2, 2, 1)
    plt.plot(time, inputs[0], 'r')
    plt.title("thrust vs time", fontsize = 18)
    # if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
    #     upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
    #     lower_lim = np.floor(plt.yticks()[0][0]*100)/100
    #     plt.ylim([lower_lim, upper_lim])
    # plt.yscale("linear")
    plt.ylim([0, 60])
    plt.subplot(2, 2, 2)
    plt.plot(time, inputs[1], 'b')
    plt.title("elevator vs time", fontsize = 18)
    
    # step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    # if step_size == 0:
    #     step_size = 1/16
    # step_size = 1/32
    # start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    # stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    # plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
    #         labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.yticks(ticks= np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16), \
                labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16)])
    

    plt.subplot(2, 2, 3)
    plt.plot(time, inputs[2], 'g')
    plt.title("aileron vs time", fontsize = 18)
    
    # step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi))
    # if step_size == 0:
    #     step_size = 1/16
    # step_size = 1/32
    # start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    # stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    # plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
    #         labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.yticks(ticks= np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16), \
                labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16)])
    

    plt.subplot(2, 2, 4)
    plt.plot(time, inputs[3], 'm')
    plt.title("rudder vs time", fontsize = 18)
    
    # step_size = np.round((np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]))/(num_ticks*np.pi) *4)/4
    # print(step_size)
    # if step_size == 0:
    #     step_size = 1/16
    # step_size = 1/32
    
    # start_tick = np.floor(plt.yticks()[0][0] / (np.pi*step_size)) * np.pi*step_size
    # stop_tick = np.ceil(plt.yticks()[0][-1]/ (np.pi*step_size)) * np.pi*step_size
    # plt.yticks(ticks= np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size), \
    #         labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(start_tick, stop_tick+np.pi*step_size, np.pi*step_size)])
    plt.yticks(ticks= np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16), \
                labels= [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-np.pi/8, np.pi/8 + np.pi/16, np.pi/16)])
    

    plt.tight_layout()
    plt.show()




def plot_wind(x_wind_list, y_wind_list, z_wind_list, time):
    plt.figure(figsize=(4, 3))
    plt.plot(time, x_wind_list, 'r', label="x_wind")
    plt.plot(time, y_wind_list, 'g', label="y_wind")
    plt.plot(time, z_wind_list, 'b', label="z_wind")
    plt.legend()
    plt.title("wind vs time", fontsize = 18)
    if np.abs(plt.yticks()[0][-1] - plt.yticks()[0][0]) < 0.01:
        upper_lim = np.ceil(plt.yticks()[0][-1]*100)/100
        lower_lim = np.floor(plt.yticks()[0][0]*100)/100
        plt.ylim([lower_lim, upper_lim])
    plt.yscale("linear")
    plt.xlabel("time")
    plt.ylabel("change in aircraft speed (m/s)")
    plt.show()
