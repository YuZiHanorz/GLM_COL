import os
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

alpha_F = 1
k = 3
ds = np.arange(4, 15.5, 0.5)

for d in ds:
    plant_init_config = f"plantedgraph{True}_informedinit{False}"    
    problem_config = f"N{3000}_aF{alpha_F:.3f}_k{k}_d{d:.3f}"
    fin_output_dir = os.path.join('result', plant_init_config, problem_config)
    filename = os.path.join(fin_output_dir, 'output.txt')
    line_number = 11
    overlap_s = read_specific_line(filename, line_number)[3:13]
    print(f"{d}: ", overlap_s)