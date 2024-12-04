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
    plt.xlabel(r"$\alpha_F$")
    plt.ylabel(r"$steps$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plt_output_dir,"steps.png"))
    plt.close()
    

alpha_Fs = np.arange(0.2, 2.1, 0.1) # for alpha_F=2, k=3
x_1 = np.sort(np.concatenate((alpha_Fs, np.arange(1.45, 1.7, 0.1))))
x_2 = np.sort(np.concatenate((alpha_Fs, np.arange(0.55, 1.1, 0.1))))
converge_steps_uninformed = [\
    [235, 231, 243, 226, 236, 227, 224, 237, 247, 226], \
    [284, 287, 301, 260, 263, 270, 281, 269, 295, 265], \
    [340, 322, 322, 298, 298, 324, 323, 320, 376, 329], \
    [330, 374, 396, 367, 374, 370, 256, 368, 422, 364], \
    [591, 420, 495, 435, 389, 421, 424, 481, 478, 473], \
    [820, 513, 244, 490, 223, 498, 466, 384, 351, 393],\
    [192, 209, 322, 598, 183, 629, 587, 390, 681, 584], \
    [226, 274, 798, 751, 549, 788, 715, 2000, 356, 830], \
    [529, 430, 290, 943, 280, 392, 817, 1588, 497, 380], \
    [425, 264, 426, 1487, 200, 310, 1483, 512, 408, 199], \
    [552, 364, 352, 2000, 263, 783, 2000, 366, 333, 131], \
    [296, 279, 2000, 2000, 2000, 233, 2000, 300, 329, 264], \
    [532, 745, 2000, 2000, 394, 374, 2000, 326, 487, 274], \
    [411, 405, 2000, 2000, 451, 247, 2000, 286, 499, 597], \
    [444, 312, 2000, 2000, 315, 296, 2000, 348, 457, 360], \
    [1179, 266, 2000, 2000, 208, 410, 2000, 500, 424, 542], \
    [193, 123, 798, 704, 123, 225, 381, 189, 193, 201], \
    [124, 102, 232, 233, 102, 115, 137, 122, 102, 108], \
    [102, 102, 131, 102, 102, 102, 110, 102, 102, 102], \
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102], \
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102], \
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102] \
    ]

converge_steps_informed = [\
    [242, 235, 251, 235, 243, 237, 257, 245, 263, 233],\
    [298, 299, 314, 271, 271, 276, 296, 282, 309, 275],\
    [356, 344, 331, 316, 313, 348, 345, 327, 384, 344],\
    [102, 417, 489, 389, 414, 396, 275, 102, 440, 387],\
    [102, 2000, 504, 401, 102, 102, 466, 425, 447, 102],\
    [102, 102, 102, 2000, 2000, 102, 2000, 102, 2000, 102],\
    [102, 102, 697, 2000, 605, 102, 2000, 102, 2000, 102],\
    [102, 102, 102, 2000, 2000, 102, 102, 102, 102, 102],\
    [102, 102, 102, 2000, 2000, 102, 102, 2000, 402, 102],\
    [102, 102, 441, 2000, 2000, 102, 102, 635, 102, 102],\
    [102, 102, 735, 2000, 2000, 102, 102, 722, 102, 102],\
    [102, 102, 874, 2000, 102, 102, 102, 102, 102, 102],\
    [102, 102, 1012, 2000, 2000, 102, 102, 102, 102, 102],\
    [102, 102, 373, 2000, 102, 102, 102, 2000, 102, 102],\
    [102, 102, 381, 2000, 2000, 102, 102, 102, 102, 102],\
    [102, 102, 102, 2000, 2000, 102, 102, 102, 102, 102],\
    [102, 102, 102, 2000, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],\
    [102, 102, 102, 102, 102, 102, 102, 102, 102, 102]\
    ]


y_1 = np.array(converge_steps_uninformed)
y_2 = np.array(converge_steps_informed)
plt_config = f"N{3000}_damping{0.75:.3f}"
plt_output_dir = os.path.join('plots_amp', plt_config)
os.makedirs(plt_output_dir, exist_ok=True)
plot_scatters(x_1, x_2, y_1, y_2, plt_output_dir)