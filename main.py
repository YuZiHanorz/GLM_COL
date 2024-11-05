import sys
from util import *
import os
import shutil

'''set parameter'''
alpha_F = float(sys.argv[1])
k = int(sys.argv[2]) # arity
d = float(sys.argv[3]) # average degree

N = 100
M = int(N * alpha_F)

initInformative = False
damping = 0.75  # parametric damping. 0 for no damping, 1 for freezing damping
maxIter, minIter = 40000, 100
err = 1e-20

'''random seeds'''
graph_seed = 42
F_seed = 42
w_seed = 42
ampInit_seed = 42

Nexp = 1

'''set output dir'''
plant_init_config = f"plantedgraph{True}_informedinit{False}"    
problem_config = f"N{N}_aF{alpha_F:.3f}_k{k}_d{d:.3f}"
seed_config = f"G{graph_seed}F{F_seed}w{w_seed}m{ampInit_seed}"
output_dir = os.path.join('result', plant_init_config, problem_config, seed_config)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'output.txt')

'''generate F'''
F_rng = np.random.default_rng(seed=F_seed)
F = F_rng.normal(0,1,(M, N))*np.sqrt(1/(N))

'''generate w* and s*'''
w_rng = np.random.default_rng(seed = w_seed)
w_star = w_rng.choice(2,size = (N))*2 - 1
s_star = (np.sign(np.abs(F@w_star)-tao)).astype(int)

#print(s_star)

factor_edge_num, factor_edge_arr, l_u_to_v_l_arr, u_l_to_lprime_u_arr, u_to_l_u_arr = createGraph(s_star, d, k, graph_seed)

#print(N, M, factor_edge_num)

psi_uu, chi_uu, log_chi_ul, marginal_u, a, v, go = uninformed_init(N, M, F, factor_edge_num, ampInit_seed)

#print(psi_uu.shape, chi_uu.shape, log_chi_ul.shape, len(a), v.shape, V, Q.shape, go.shape)

'''for plotting'''
oSpr, oWpr = [0]*10, [0]*10
a_error_list = []
v_error_list = []
marginal_error_list = []
bethe_free_entropy_log2_ls = []
bethe_free_entropy_per_log2_ls = []
bethe_free_entropy_col_ls = []
q_ls, v_ls, Q_ls, V_ls, go_ls, d_go_ls, Gamma_ls, Lambda_ls, psi_uu_pos_ls, chi_uu_pos_ls,\
chi_ul_pos_ls, psi_lu_pos_ls, marginal_u_pos_ls = [],[],[],[],[],[],[],[],[],[],[],[],[]
w_overlap_ls,s_overlap_ls = [], []
w_coarse_overlap_ls,s_coarse_overlap_ls = [], []
damping_ls = []
sampled_ids_n = np.random.choice(N, 40)
sampled_ids_u = np.random.choice(M, 40)
sampled_ids_ul = np.random.choice(factor_edge_num, 40)

for iter in tqdm(range(maxIter)):
    break_flag = False
    plot_flag = iter % 100 == 1
    
    '''amp-bp-step'''
    log_chi_ul_N, chi_uu_N, marginal_u_N, Phi_BP = step_BP(psi_uu, log_chi_ul, l_u_to_v_l_arr, u_l_to_lprime_u_arr, 
    factor_edge_arr[:,0], u_to_l_u_arr, k, N)
    a_N, v_N, psi_uu_N, go_N, phi_AMP = step_AMP(a, v, chi_uu, go, F)
    
    a_error_list.append(np.linalg.norm(a_N - a))
    v_error_list.append(np.linalg.norm(v_N - v) )
    marginal_error_list.append(np.linalg.norm(marginal_u_N - marginal_u) )
    
    '''damping update'''
    log_chi_ul = damping * log_chi_ul + (1-damping) * log_chi_ul_N
    chi_uu = damping * chi_uu + (1-damping) * chi_uu_N
    marginal_u = damping * marginal_u + (1-damping) * marginal_u_N
    a = damping * a + (1-damping) * a_N
    v = damping * v + (1-damping) * v_N
    psi_uu = damping * psi_uu + (1-damping) * psi_uu_N
    go = damping * go + (1-damping) * go_N
    
    '''compute overlap'''
    oS, oW = get_overlapS(marginal_u, s_star), get_overlapW(a, w_star)
    oSpr.append(oS)
    oSpr.pop(0)
    oWpr.append(oW)
    oWpr.pop(0)
    if iter > minIter and np.std(oSpr) < err and np.std(oWpr) < err:
        break_flag = True

    if plot_flag:
        break
    
    if break_flag:
        break