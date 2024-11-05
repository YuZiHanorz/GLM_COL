from scipy import special
from tqdm import tqdm
import datetime
import sys
from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np

epsilon = 2**(-200)
tao = np.sqrt(2)*special.erfinv(0.5)
beta = np.inf

def createGraph(s_star, d, k, seed):
    """Generates the random graph given colors of the nodes"""
    
    M = len(s_star)
    G_rng = np.random.default_rng(seed = seed)
    L = np.floor(M * d / k).astype(int)

    node_arr = np.arange(M)
    
    cnt = 0
    setEdge_ls = []
    
    pbar = tqdm(total=L)

    # sample edges
    while cnt < L:
        setEdge = set(G_rng.choice(node_arr, k, replace = False))
        edgeNodesList = np.asarray(list(setEdge))
        if not np.all(s_star[edgeNodesList] == s_star[edgeNodesList[0]]) and setEdge not in setEdge_ls:
            setEdge_ls.append(setEdge)
            cnt += 1
            pbar.update(1)
            #print(f"adding {setEdge}")
        else:
            #print(f"discarding {setEdge}")
            continue

    edge_ls = [list(s) for s in setEdge_ls]

    # sanity check: no repetition of the same edge.
    tmp = [tuple(l) for l in edge_ls]
    assert(len(set(tmp)) == len(tmp))

    #print('edges:\n', np.array(edge_ls, dtype = int))
    edge_nodes_arr = np.expand_dims(np.array(edge_ls, dtype = int),axis=2 )# L,k,1
    edge_idx_arr_ = np.arange(L).reshape(-1,1)
    edge_idx_arr = np.expand_dims(np.tile(edge_idx_arr_, (1, k)),axis = 2) # L,k,1
    factor_edge_arr = np.concatenate([edge_nodes_arr, edge_idx_arr], axis = 2).reshape(-1,2) # L* k, 2  all factor edges (u, l) with u in l
    factor_edge_num = L * k
    assert (factor_edge_num == factor_edge_arr.shape[0])
    #print(factor_edge_arr)
    
    # for psi_lu
    l_u_to_v_l_ls = [[] for _ in range(factor_edge_num)]
    for fe_id, fe in enumerate(factor_edge_arr):
        l = fe[1]
        v_l_list = list(np.where(factor_edge_arr[:,1] == l)[0])
        v_l_list.remove(fe_id)
        l_u_to_v_l_ls[fe_id] += v_l_list
    l_u_to_v_l_arr = np.asarray(l_u_to_v_l_ls, dtype = int) # must be k-1, no need to pad
    #print('for psi_lu:\n', l_u_to_v_l_arr)
    
    # for chi_ul
    u_l_to_lprime_u_ls = [[] for _ in range(factor_edge_num)]
    for fe_id, fe in enumerate(factor_edge_arr):
        u = fe[0]
        lprime_u_ls = list(np.where(factor_edge_arr[:,0] == u)[0])
        lprime_u_ls.remove(fe_id)
        u_l_to_lprime_u_ls[fe_id] += lprime_u_ls
    u_l_to_lprime_u_arr = np.asarray(list_padding(u_l_to_lprime_u_ls, factor_edge_num), dtype = int)
    #print('for chi_ul:\n', u_l_to_lprime_u_arr)
    
    # for chi_uu
    u_to_l_u_ls = [[] for _ in range(M)]
    for fe_id, fe in enumerate(factor_edge_arr):
        u = fe[0]
        u_to_l_u_ls[u].append(fe_id)
    u_to_l_u_arr = np.asarray(list_padding(u_to_l_u_ls, factor_edge_num), dtype = int) # padding to max degree with a 'dummy' factor edge
    # u_to_l_u_arr serve as the index to find all relevant phi_lu for every u
    #print('for chi_uu:\n', u_to_l_u_arr)
    
    return factor_edge_num, factor_edge_arr, l_u_to_v_l_arr, u_l_to_lprime_u_arr, u_to_l_u_arr

def uninformed_init(N, M, F, factor_edge_num, ampInit_seed):
    """
    uninformed initialization
    """
    rng = np.random.default_rng(ampInit_seed)
    log_chi_ul = (2 * rng.random((factor_edge_num, 2))-1) * (N**-0.5)
    log_chi_ul -= special.logsumexp(log_chi_ul, axis=1, keepdims=True)
    chi_uu = np.ones((M, 2))/2
    marginal_u = np.ones((M, 2))/2
    
    a = rng.normal(0, 1, N) * (N**-0.5) # difference here
    v = np.ones(N)
    
    V = np.mean(v)
    Q = F @ a
    psi_uu = get_psi_uu(Q, V)
    go, _ = get_go(Q, chi_uu, V)
    
    return psi_uu, chi_uu, log_chi_ul, marginal_u, a, v, go

def step_BP(psi_uu, log_chi_ul, l_u_to_v_l_arr, u_l_to_lprime_u_arr, u_l_to_u_arr, u_to_l_u_arr, k, N):
    
    '''compute psi_lu as a helper variable'''
    log_chi_vl_product = np.sum(log_chi_ul[l_u_to_v_l_arr], axis = 1)
    log_psi_lu_unnorm = np.log( (np.exp(-beta)-1)*np.exp(log_chi_vl_product)+1 )
    log_psi_lu = log_psi_lu_unnorm - special.logsumexp(log_psi_lu_unnorm, axis=1, keepdims=True)
    log_psi_lu_padded = np.vstack((log_psi_lu, np.zeros((1,log_psi_lu.shape[1])))) 
    
    '''compute chi_ul_N'''
    log_chi_ul_N = np.sum(log_psi_lu_padded[u_l_to_lprime_u_arr],axis = 1) + np.log(psi_uu[u_l_to_u_arr])
    log_chi_ul_N -= special.logsumexp(log_chi_ul_N, axis=1, keepdims=True)
    
    '''compute chi_uu_N'''
    chi_uu_unnorm_N = np.exp(np.sum(log_psi_lu_padded[u_to_l_u_arr],axis = 1))
    chi_uu_N =  normalize_2d(chi_uu_unnorm_N)
    
    '''compute marginal_N'''
    marginal_u_unnorm_N = chi_uu_unnorm_N * psi_uu
    marginal_u_N = normalize_2d(marginal_u_unnorm_N)
    
    '''compute Phi_BP after updation'''
    log_chi_vl_N_product = np.sum(log_chi_ul_N[l_u_to_v_l_arr], axis = 1)
    log_psi_lu_unnorm_N = np.log( (np.exp(-beta)-1)*np.exp(log_chi_vl_N_product)+1 )
    log_psi_lu_unnorm_N_padded = np.vstack((log_psi_lu_unnorm_N, np.zeros((1,log_psi_lu_unnorm_N.shape[1]))))
    log_part6_product = np.sum(log_psi_lu_unnorm_N_padded[u_to_l_u_arr],axis = 1)
    part6 = np.sum(np.log(np.sum(np.exp(log_part6_product),axis = 1)))
    
    k_phi_L = np.log(np.sum(np.exp(log_chi_ul_N + log_psi_lu_unnorm_N),axis = 1))
    #print(k_phi_L)
    part7 = (1 - k) * np.sum(k_phi_L) / k
    Phi_BP = (part6 + part7) / N
    
    return log_chi_ul_N, chi_uu_N, marginal_u_N, Phi_BP

def step_AMP(a, v, chi_uu, go, F):
    N = len(a)
    M = len(chi_uu)
    
    V_N = np.mean(v)
    Q_N = F @ a - V_N * go
    
    psi_uu_N = get_psi_uu(Q_N, V_N)
    go_N, _ = get_go(Q_N, chi_uu, V_N)
    d_go_N = - go_N**2
    
    Lambda_N = -np.mean(d_go_N)
    Gamma_N = a * Lambda_N + go_N @ F

    a_N = np.tanh(Gamma_N)
    v_N = np.maximum(epsilon, np.cosh(Gamma_N)**(-2))
    
    '''compute Phi_AMP after updation'''
    part1 = -N*Lambda_N/2 + np.sum(np.logaddexp(Gamma_N, -Gamma_N) - np.log(2))
    part2 = 0.5*np.sum(Lambda_N*(a_N**2 + v_N))
    part3 = -np.sum(Gamma_N*a_N)
    part4 = np.sum((F@a_N - Q_N)**2/2/V_N)
    
    chi_p = chi_uu[:,0].reshape(-1)
    chi_n = chi_uu[:,1].reshape(-1)
    log_integral = np.log(1/2*(chi_n*(special.erf((tao-Q_N)/np.sqrt(2*V_N))+special.erf((tao+Q_N)/np.sqrt(2*V_N))) +\
                                chi_p*(special.erfc((tao-Q_N)/np.sqrt(2*V_N))+special.erfc((tao+Q_N)/np.sqrt(2*V_N)))))     
    part5 = np.sum(log_integral)
    phi_AMP = (part1+part2+part3+part4+part5) / N

    return a_N, v_N, psi_uu_N, go_N, phi_AMP

def get_psi_uu(Q, V):
    psi_uu_n = 0.5 * (special.erf((tao-Q)/np.sqrt(2*V)) + special.erf((tao+Q)/np.sqrt(2*V))).reshape(-1,1)
    psi_uu_p = 1 - psi_uu_n
    psi_uu = np.hstack([psi_uu_p, psi_uu_n])
    psi_uu = np.maximum(epsilon, psi_uu)
    return psi_uu

def get_go(Q, chi_uu, V):
    chis_p = chi_uu[:,0].reshape(-1)
    chis_n = chi_uu[:,1].reshape(-1)
    
    numerator_devided_by_V = (np.exp(-(tao+Q)**2/2/V)-np.exp(-(tao-Q)**2/2/V))*(chis_n - chis_p)
    denominator_part = chis_n*(special.erf((tao-Q)/np.sqrt(2*V))+special.erf((tao+Q)/np.sqrt(2*V))) +\
                        chis_p*(special.erfc((tao-Q)/np.sqrt(2*V))+special.erfc((tao+Q)/np.sqrt(2*V)))
    denominator = np.sqrt(V*np.pi/2)*denominator_part 
    
    denominator = np.maximum(epsilon, denominator)
    go = numerator_devided_by_V / denominator
    return go, denominator

def quantileSpread(xs, q):
    """
    Compute the difference between the q-th quantile and the (1-q)-th quantile of xs.
    """
    q = max(q, 1-q)
    x1 = np.quantile(xs, q)
    x2 = np.quantile(xs, 1-q)
    return x1-x2    

def normalize_2d(arr):
    """normalizing arrays along the innermost dimension and return the normalized array."""
    assert (arr>=0).all()
    return arr / np.sum(arr, axis=1, keepdims=True)

def list_padding(list_of_list, padding_num):
    """Pad the sublists in a list to make the same length, so that they can be easily converted to numpy arrays."""
    len_ls = [len(ls) for ls in list_of_list]
    max_len = max(len_ls)
    padded_list = [ls+[padding_num]*(max_len - len(ls)) for ls in list_of_list]
    return padded_list

def fprint(text, file, timestamp=True,):
    """Print text to screen and write to FPRINT_FILE."""
    # Add timestamp if requested
    if timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = timestamp + ' ' + text
    # Redirect stdout to file
    original = sys.stdout
    if file is not None:
        sys.stdout = open(file, 'a+')
        print(text)
        # Set stdout back to original
        sys.stdout = original
    print(text)
