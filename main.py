import sys
from util import *
import os
import shutil

'''set parameter'''
alpha_F = float(sys.argv[1])
k = int(sys.argv[2]) # arity
d = float(sys.argv[3]) # average degree

N = 3000
M = int(N * alpha_F)

initInformative = False
damping = 0.9  # parametric damping. 0 for no damping, 1 for freezing damping
maxIter, minIter = 2000, 100
err = 1e-5
informed = False

'''number of repeated experiments'''
Nexp = 10

overlapS, overlapW, conv_steps, Phi_minus_Phi_info = [], [], [], []
plant_init_config = f"plantedgraph{True}_informedinit{informed}"    
problem_config = f"N{N}_aF{alpha_F:.3f}_k{k}_d{d:.3f}_damping{damping:.3f}"
fin_output_dir = os.path.join('test', plant_init_config, problem_config)
if os.path.exists(fin_output_dir):
    shutil.rmtree(fin_output_dir)
os.makedirs(fin_output_dir, exist_ok=True)
fin_output_file = os.path.join(fin_output_dir, 'output.txt')

for n in range(Nexp):
    '''random seeds'''
    graph_seed = 4 * n + 1
    F_seed = 4 * n + 2
    w_seed = 4 * n + 3
    ampInit_seed = 4 * n + 4
    
    '''set output dir'''
    seed_config = f"G{graph_seed}F{F_seed}w{w_seed}m{ampInit_seed}"
    output_dir = os.path.join('test', plant_init_config, problem_config, seed_config)
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

    print(np.mean(s_star == -1))

    factor_edge_num, factor_edge_arr, l_u_to_v_l_arr, u_l_to_lprime_u_arr, u_to_l_u_arr = createGraph(s_star, d, k, graph_seed)

    #print(N, M, factor_edge_num)
    if not informed:
        psi_uu, chi_uu, log_chi_ul, marginal_u, a, v, go = uninformed_init(N, M, F, factor_edge_num, ampInit_seed, w_star)
    else:
        psi_uu, chi_uu, log_chi_ul, marginal_u, a, v, go = informed_init(N, M, F, factor_edge_num, factor_edge_arr[:,0], ampInit_seed, s_star, w_star)

    #print(psi_uu.shape, chi_uu.shape, log_chi_ul.shape, len(a), v.shape, V, Q.shape, go.shape)

    '''for plotting'''
    a_error_list = []
    v_error_list = []
    marginal_error_list = []
    Phi_BP_ls = []
    Phi_AMP_ls = []
    Phi_minus_Phi_info_ls = []
    marginal_u_pos_ls = []
    w_overlap_ls,s_overlap_ls = [], []
    sampled_ids_u = np.random.choice(M, 40)

    '''iteration'''
    oSpr, oWpr = [0]*10, [0]*10
    convergence_steps = 0
    for iter in tqdm(range(maxIter)):
        break_flag = False
        plot_flag = iter % 100 == 0
        
        '''amp-bp step'''
        log_chi_ul_N, chi_uu_N, marginal_u_N, Phi_BP = step_BP(psi_uu, log_chi_ul, l_u_to_v_l_arr, u_l_to_lprime_u_arr, 
        factor_edge_arr[:,0], u_to_l_u_arr, k, N)
        a_N, v_N, psi_uu_N, go_N, Phi_AMP = step_AMP(a, v, chi_uu, go, F)
        
        '''for plotting'''
        a_error_list.append(np.linalg.norm(a_N - a))
        v_error_list.append(np.linalg.norm(v_N - v) )
        marginal_error_list.append(np.linalg.norm(marginal_u_N - marginal_u) )
        Phi_BP_ls.append(Phi_BP)
        Phi_AMP_ls.append(Phi_AMP)
        Phi_minus_Phi_info_ls.append(Phi_BP+Phi_AMP+np.log(2))
        
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
            plot_flag = True
            break_flag = True

        '''for plotting'''
        marginal_u_pos_ls.append(marginal_u[:, 0][sampled_ids_u])
        w_overlap_ls.append(oW)
        s_overlap_ls.append(oS)
        if plot_flag:
            '''plot error'''
            plt.plot(marginal_error_list,label="marginal_u")
            plt.plot(a_error_list, label="a")
            plt.plot(v_error_list, label = "v")
            plt.xlabel("iterations")
            plt.ylabel("log error")
            plt.legend()
            plt.yscale('log')
            plt.ylim([10**-25, 10])
            plt.savefig(os.path.join(output_dir,"error_log.png"))
            plt.close()
            
            '''plot entropy'''
            plot_y_min = -3
            plot_y_max = 1
            plt.plot(Phi_BP_ls, label = r"$\Phi^{\mathrm{COL}}$")
            plt.plot(Phi_AMP_ls, label = r"$\Phi^{\mathrm{B}}_{\mathrm{GLM}^\prime}$")
            plt.plot(Phi_minus_Phi_info_ls, label = r"$\Phi^{\mathrm{B}}$" +"+" +r"$\log 2$")
            # Sometimes several of the above lists might be zero-lengthed. Hence, I am doing the following.
            valid_ls = [ls for ls in [Phi_BP_ls, Phi_AMP_ls, Phi_minus_Phi_info_ls] if len(ls)!= 0]
            y_min = np.min(valid_ls)
            y_max = np.max(valid_ls)
            plot_y_min = np.max([y_min,plot_y_min])
            plot_y_max = np.min([y_max,plot_y_max])
            plt.ylim([-10-0.1, 10+0.1])
            plt.xlabel("iterations")
            plt.ylabel(r"$\Phi$")
            plt.legend()
            if len(Phi_BP_ls)>0:
                plt.text(0, -0.1, fr"$\Phi_c=${Phi_BP_ls[-1]:.5f}", transform=plt.gca().transAxes)
            if len(Phi_AMP_ls)>0:
                plt.text(0, -0.15, fr"$\Phi_p=${Phi_AMP_ls[-1]:.5f}", transform=plt.gca().transAxes)
            if len(Phi_minus_Phi_info_ls)>0:
                plt.text(0, -0.2, fr"$\Phi+\log 2=${Phi_minus_Phi_info_ls[-1]:.5f}", transform=plt.gca().transAxes)
            plt.savefig(os.path.join(output_dir,"Phi_bethe.png"),bbox_inches = "tight")
            plt.close()
            
            '''plot marginal'''
            plt.plot(marginal_u_pos_ls)
            plt.xlabel("iterations")
            plt.ylabel(r"$\chi_{\mu}(+1)$")
            plt.savefig(os.path.join(output_dir,"marginal_u.png"))
            plt.close()
            
            '''plot overlap'''
            plt.plot(w_overlap_ls, label = r"$\mathbf{w}$ overlap")
            plt.plot(s_overlap_ls, label = r"$\mathbf{s}$ overlap")
            plt.xlabel("iterations")
            plt.ylabel("overlap")
            plt.legend()
            plt.savefig(os.path.join(output_dir,"overlap.png"))
            plt.close()
        
        if break_flag:
            convergence_steps = iter
            fprint(f"Algorithm converges after {iter+1} iteration", file=output_file)
            fprint(f"experiment {n+1} converges after {iter+1} iteration", file=fin_output_file, timestamp=False)
            fprint(f"The para free entropy plus log2 is {np.log(2)+alpha_F*d/k*np.log(1-1/(2**(k-1)))} and final bethe free entropy plus log2 is {Phi_minus_Phi_info_ls[-1]}, overlap_w is {w_overlap_ls[-1]}, {s_overlap_ls[-1]}", file=output_file)
            break
        
        if iter == maxIter - 1:
            convergence_steps = iter
            fprint(f"Algorithm does not converge!", file=output_file)
            fprint(f"experiment {n+1} does not converge!", file=fin_output_file, timestamp=False)
            fprint(f"The para free entropy plus log2 is {np.log(2)+alpha_F*d/k*np.log(1-1/(2**(k-1)))} and final bethe free entropy plus log2 is {Phi_minus_Phi_info_ls[-1]}, overlap_w is {w_overlap_ls[-1]}, {s_overlap_ls[-1]}", file=output_file)
    
    '''One experiment done'''
    overlapS.append(np.mean(oSpr))
    overlapW.append(np.mean(oWpr))
    conv_steps.append(convergence_steps)
    Phi_minus_Phi_info.append(Phi_minus_Phi_info_ls[-1])

s = "{}, {}, {}, " + ", ".join(['{:.4}']*3*Nexp)
fprint(s.format(alpha_F, k, d, *overlapS, *overlapW, *Phi_minus_Phi_info), file=fin_output_file, timestamp=False)