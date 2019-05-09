from regex_file_collector import Collector
from analysis_functions import compute_instability_timestep, compute_fracture_velocity
from lammps_logfile import File
from LammpsUtilities.convenience import getMatlabColor
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 8, 'font.weight' : 'normal'})

def patternmaker(name):
    regexpstring = name + r"\_(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
    return regexpstring

path = "/home/henriasv/simulation_data/simple_fracture_simulations_2019-05-08/"
pattern = patternmaker("smoothing")+"_"+patternmaker("rbreak")+"_"+patternmaker("k")+"/log.lammps"

collection = Collector(path, pattern, fields=("smoothing", "rbreak", "k"))

myDict = collection.get_tree()

v_R = 0.634*np.sqrt(10)

data_dict = {}

plt.figure(figsize=(4, 6))
for smoothing, value in myDict.items():
    for rbreak, value2 in value.items():
        for k, file_name in value2.items():
            colorIndex = int(float(k)-1)
            log = File(file_name)
            t, x, v = compute_fracture_velocity(log)
            ind = compute_instability_timestep(log,v, N=100)

            plt.subplot(2,1,2)
            plt.semilogx(x[:-1:10]-278, v[::10]/v_R,color=getMatlabColor(colorIndex), linewidth=0.75)
            plt.semilogx(x[ind]-278, v[ind]/v_R, "o",color=getMatlabColor(colorIndex))
            vcrit = v[ind]
            if not str(k) in data_dict.keys():
                data_dict[k] = {"rbreak":[], "vcrit" : []}
            if x[ind]<1000:
                data_dict[k]["rbreak"].append(float(rbreak))
                data_dict[k]["vcrit"].append(float(vcrit))

plt.xlabel("Distance along sample")
plt.ylabel("Crack speed$/v_R$")
plt.xlim([1, 1010])
plt.subplot(2,1,1)
ks = [1.0, 2.0, 4.0]

for k in ks:
    value = data_dict[str(k)]
    value["rbreak"] = np.asarray(value["rbreak"])
    value["vcrit"] = np.asarray(value["vcrit"])
    colorIndex = int(float(k)-1)
    indices = np.argsort(value["rbreak"])
    plt.plot(value["rbreak"][indices], value["vcrit"][indices]/v_R, "-o", color=getMatlabColor(colorIndex), label=str(k))
plt.ylabel(r"$v_\mathrm{crit}/v_R$")
plt.xlabel(r"$r_\mathrm{break}$")
plt.plot([1.13, 1.26], [0.73, 0.73], "--", c="k")
plt.legend()
plt.tight_layout()
plt.savefig("figures/instability_speeds.pdf")
plt.show()