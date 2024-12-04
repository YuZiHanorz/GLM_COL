import matplotlib.pyplot as plt
import os
import sys
import shutil
import numpy as np

def read_specific_line(filename, line_number):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if line_number <= len(lines):
            line_content = lines[line_number - 1].strip()
            float_list = [float(value) for value in line_content.split(', ')]
            return float_list
        else:
            print(f"exceed line cnt {len(lines)}")
            return None

def plot_overlap_s(x, data, plt_output_dir):
    medians = np.median(data, axis=1)
    lower_quartile = np.percentile(data, 25, axis=1)
    upper_quartile = np.percentile(data, 75, axis=1)

    yerr = np.array([medians - lower_quartile, upper_quartile - medians])

    plt.errorbar(x, medians, yerr=yerr, fmt='', capsize=5, label=fr"$\alpha_F = {alpha_F}, k = {k}$")
    plt.plot(x, medians, '') 
    plt.xlabel('d')
    plt.ylabel(r"$q_S$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plt_output_dir,"overlap.png"))
    plt.close()

def plot_phi(x, data, plt_output_dir):
    medians = np.median(data, axis=1)
    lower_quartile = np.percentile(data, 25, axis=1)
    upper_quartile = np.percentile(data, 75, axis=1)

    yerr = np.array([medians - lower_quartile, upper_quartile - medians])

    plt.errorbar(x, medians, yerr=yerr, fmt='', capsize=5, label=fr"$\alpha_F = {alpha_F}, k = {k}$")
    plt.plot(x, medians, '') 
    plt.xlabel('d')
    plt.ylabel(r"$\Phi^{\mathrm{B}}$" +"-" +r"$\Phi^{\mathrm{info}}$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plt_output_dir,"phi_B-phi_info.png"))
    plt.close()


alpha_F = float(sys.argv[1])
k = int(sys.argv[2])

"""# for alpha_F=1, k=3
ds = np.arange(1.0, 10.2, 0.2) 
ds = np.sort(ds)
"""

"""# for alpha_F=3, k=3
ds = np.arange(4.0, 6.55, 0.1) 
ds = np.concatenate((ds, np.arange(5.52, 5.59, 0.02)))
ds = np.concatenate((ds, np.arange(5.66, 5.69, 0.02)))
ds = np.concatenate((ds, np.arange(1.8, 3.9, 0.2)))
ds = np.sort(ds)
"""

# for alpha_F=5, k=3
ds = np.arange(1.0, 8.2, 0.2) 
ds = np.concatenate((ds, np.arange(5.1, 5.6, 0.1)))
ds = np.sort(ds)

overlap_s = []
Phi = []

for d in ds:
    plant_init_config = f"plantedgraph{True}_informedinit{False}"    
    problem_config = f"N{3000}_aF{alpha_F:.3f}_k{k}_d{d:.3f}_damping{0.75:.3f}"
    fin_output_dir = os.path.join('test', plant_init_config, problem_config)
    filename = os.path.join(fin_output_dir, 'output.txt')
    line_number = 11
    content = read_specific_line(filename, line_number)
    overlap_s.append(content[3:13])
    Phi.append(content[23:])

x = ds
data = np.array(overlap_s)
plt_config = f"N{3000}_aF{alpha_F:.3f}_k{k}"
plt_output_dir = os.path.join('plots_test', plt_config)
if os.path.exists(plt_output_dir):
    shutil.rmtree(plt_output_dir)
os.makedirs(plt_output_dir, exist_ok=True)
plot_overlap_s(x, data, plt_output_dir)

data = np.array(Phi)
plot_phi(x, data, plt_output_dir)