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

def plot_scatters(x_1, x_2, y_1, y_2, plt_output_dir):
    medians_1 = np.median(y_1, axis=1)
    lower_quartile = np.percentile(y_1, 25, axis=1)
    upper_quartile = np.percentile(y_1, 75, axis=1)
    yerr_1 = np.array([medians_1 - lower_quartile, upper_quartile - medians_1])
    
    medians_2 = np.median(y_2, axis=1)
    lower_quartile = np.percentile(y_2, 25, axis=1)
    upper_quartile = np.percentile(y_2, 75, axis=1)
    yerr_2 = np.array([medians_2 - lower_quartile, upper_quartile - medians_2])

    plt.errorbar(x_1, medians_1, yerr=yerr_1, fmt='', capsize=5, label="random init")
    plt.errorbar(x_2, medians_2, yerr=yerr_2, fmt='', capsize=5, label="informed init")
    plt.plot(x_1, medians_1, '')
    plt.plot(x_2, medians_2, '')
    plt.xlabel(r"$\alpha_F$")
    plt.ylabel(r"$q_w$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plt_output_dir,"overlap_w.png"))
    plt.close()
    
def plot_overlap_w(x_1, x_2, y_1, y_2, plt_output_dir):
    medians_1 = np.median(y_1, axis=1)
    lower_quartile = np.percentile(y_1, 25, axis=1)
    upper_quartile = np.percentile(y_1, 75, axis=1)
    yerr_1 = np.array([medians_1 - lower_quartile, upper_quartile - medians_1])
    
    medians_2 = np.median(y_2, axis=1)
    lower_quartile = np.percentile(y_2, 25, axis=1)
    upper_quartile = np.percentile(y_2, 75, axis=1)
    yerr_2 = np.array([medians_2 - lower_quartile, upper_quartile - medians_2])

    plt.errorbar(x_1, medians_1, yerr=yerr_1, fmt='', capsize=5, label="random init")
    plt.errorbar(x_2, medians_2, yerr=yerr_2, fmt='', capsize=5, label="informed init")
    plt.plot(x_1, medians_1, '')
    plt.plot(x_2, medians_2, '')
    plt.xlabel(r"$\alpha_F$")
    plt.ylabel(r"$q_w$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plt_output_dir,"overlap_w.png"))
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


alpha_Fs = np.arange(0.2, 2.1, 0.1) # for alpha_F=2, k=3
x_1 = np.sort(np.concatenate((alpha_Fs, np.arange(1.45, 1.7, 0.1))))
x_2 = np.sort(np.concatenate((alpha_Fs, np.arange(0.55, 1.1, 0.1))))
overlap_w_uninformed, overlap_w_informed = [], []

for alpha_F in x_1:
       
    problem_config = f"N{3000}_aF{alpha_F:.3f}_k{3}_d{3:.3f}_damping{0.75:.3f}"
    
    plant_init_config = f"plantedgraph{True}_informedinit{False}" 
    fin_output_dir = os.path.join('verify_AMP_Yizhou', plant_init_config, problem_config)
    filename = os.path.join(fin_output_dir, 'output.txt')
    line_number = 11
    content = read_specific_line(filename, line_number)
    overlap_w_uninformed.append(content[13:23])

for alpha_F in x_2:
    problem_config = f"N{3000}_aF{alpha_F:.3f}_k{3}_d{3:.3f}_damping{0.75:.3f}"
    
    plant_init_config = f"plantedgraph{True}_informedinit{True}"    
    fin_output_dir = os.path.join('verify_AMP_Yizhou', plant_init_config, problem_config)
    filename = os.path.join(fin_output_dir, 'output.txt')
    line_number = 11
    content = read_specific_line(filename, line_number)
    overlap_w_informed.append(content[13:23])

y_1 = np.array(overlap_w_uninformed)
y_2 = np.array(overlap_w_informed)
plt_config = f"N{3000}_damping{0.75:.3f}"
plt_output_dir = os.path.join('plots_amp', plt_config)
if os.path.exists(plt_output_dir):
    shutil.rmtree(plt_output_dir)
os.makedirs(plt_output_dir, exist_ok=True)
plot_overlap_w(x_1, x_2, y_1, y_2, plt_output_dir)