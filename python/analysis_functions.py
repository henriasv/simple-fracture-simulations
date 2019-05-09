from lammps_logfile import File
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.weight' : 'normal'})
import numpy as np
from LammpsUtilities.convenience import RunningMean

from scipy.ndimage import gaussian_filter

v_R = 0.634*np.sqrt(10)

def compute_fracture_velocity(log, N = 200):
    myRunningMean = RunningMean(N)
    print(log.keywords)
    crack_position = log.get("c_crackpos")
    time = log.get("Time")
    #x = myRunningMean(crack_position)
    x = gaussian_filter(crack_position, N)
    v = np.diff(x)/np.diff(time)
    print("hello")
    return time, x, v

def compute_instability_timestep(log, v, N=200, crack_instability_threshold = 8):
    myRunningMean = RunningMean(N)
    crack_position = log.get("c_crackpos")
    crack_dev = log.get("c_crackdev")
    instability_timestep = np.argmin(crack_dev<crack_instability_threshold)
    max_speed_ind = np.argmax(v[:instability_timestep-N])
    return max_speed_ind

def plot_fracture_velocity(log, N=200):
    t, x, v  = compute_fracture_velocity(log,N)
    step = log.get("Step")
    plt.plot(x[:-1:10], v[::10]/v_R)
    plt.plot(np.array([0, 1076]), [1, 1], ":", c="k")
    plt.xlabel("Position")
    plt.ylabel("$v/v_R$")
    plt.xlim([0, 1076])
    plt.ylim([-0.1, 1.2])
    max_speed_ind = compute_instability_timestep(log, v, N)

    plt.plot(x[max_speed_ind], v[max_speed_ind]/v_R, "o")


    
    
if __name__=="__main__": 
    ifile = "/home/henriasv/simulation_data/simple_fracture_simulations_2019-05-08/smoothing_150_rbreak_1.21_k_2.0/log.lammps"
    log = File(ifile)

    #compute_instability_speed(log)
    #plt.show()
    plt.figure(figsize=(8, 1.5))
    plot_fracture_velocity(log, 100)
    plt.tight_layout()
    plt.savefig("figures/crack_speed_single.pdf")
    plt.show()
    """
    
    """