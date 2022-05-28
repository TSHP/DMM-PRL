import os
import numpy as np
from numpy.random import randn, seed, multinomial

import matplotlib.pyplot as plt

from src.simulate_data import sample_same_trans, sample_same_stick
from src.gibbs_multinomial import *
from src.permute import compute_cost
from src.util import sample_pi_efox, compute_log_marginal_lik_multinomial


seed(42)

def generate_data(file_path, args):
    nof_states = 2
    p_1, p_2, p_3, p_4, p_y, steps, states_ratio = args
    dic_real = np.array([[p_y, 1 - p_y], [1 - p_y, p_y]])
    dic_real = dic_real / dic_real.sum(axis=1, keepdims=True)

    steps_per_state = [int(steps * states_ratio), int(steps * (1 - states_ratio))]
    zt_real, yt_real, wt_real, kappa_real, trans_vec, trans_mat = [[] for i in range(6)]
    p = [[p_1, p_2, p_3, p_4], [p_2, p_1, p_3, p_4]]

    for state in range(nof_states):
        steps = steps_per_state[state]
        p_1, p_2, p_3, p_4 = p[state]
        zt_real_tmp, wt_real_tmp, kappa_real_tmp, trans_vec_tmp = sample_same_trans(K_real=nof_states, p_real1=p_1,
                                                                                    p_real2=p_2,
                                                                                    p_real3=p_3, p_real4=p_4,
                                                                                    T=steps)

        zt_real.append(zt_real_tmp)
        wt_real.append(wt_real_tmp)
        kappa_real.append(kappa_real_tmp)
        trans_vec.append(trans_vec_tmp)

        yt_real_tmp = np.array([multinomial(1, dic_real[zt_real_tmp][ii]) for ii in range(len(zt_real_tmp))])
        # trans_mat_tmp = trans_vec_tmp * np.expand_dims(1 - kappa_real_tmp, axis=-1) + np.diag(
        # kappa_real_tmp)
        if len(yt_real_tmp) != 0:
            yt_real.append(yt_real_tmp[:, 1])
        # trans_mat.append(trans_mat_tmp)

    # np.savez(file_path, zt=flatten(zt_real), wt=flatten(wt_real), kappa=flatten(kappa_real),
    #          yt=flatten(yt_real), dic=dic_real, trans_mat=flatten(trans_mat))
    np.savez(file_path, zt=flatten(zt_real), wt=flatten(wt_real), kappa=flatten(kappa_real),
             yt=flatten(yt_real), dic=dic_real)



def flatten(list):
    return [item for sublist in list for item in sublist]


def main():
    data_path = "./io/data"
    plot_path = "./io/plots"
    results_path = "./io/results"

    filename = "2_states_test"

    file_path = data_path + "/" + filename + ".npz"

    p_1 = 1  # P(staying in state 0 if currently in state 0)
    p_2 = 0  # P(staying in state 1 if currently in state 1)
    p_3 = 1  # P(going to state 0 if state transition occurs)
    p_4 = 0  # P(going to state 1 if state transition occurs)

    p_y = 0.85  # P(observing real hidden state)
    steps = 100
    states_ratio = 1
    args = [p_1, p_2, p_3, p_4, p_y, steps, states_ratio]

    generate_data(file_path, args)

    alpha0_a_pri = 1
    alpha0_b_pri = 0.01
    gamma0_a_pri = 2
    gamma0_b_pri = 1

    iters = 10
    dir0 = 0.1

    path = data_path + "/2_states_test.npz"
    path_test = data_path + "/2_states_test.npz"

    ## load data
    dat = np.load(path)
    zt_real, yt_real = dat['zt'], dat['yt']  ## yt_real & zt_real are 1d length T1 numpy array
    test_dat = np.load(path_test)
    yt_test = test_dat['yt']  ## yt_test is 1d length T2 numpy array

    fig, ax = plt.subplots()
    ax.set_title('zt_real')
    ax.set_xlabel('t')
    ax.set_ylabel('zt_real')
    ax.scatter(range(0, len(zt_real)), zt_real, color='tab:blue')
    fig.savefig(plot_path + "/zt_real.eps", format='eps')

    fig, ax = plt.subplots()
    ax.set_title('yt_real')
    ax.set_xlabel('t')
    ax.set_ylabel('yt_real')
    ax.scatter(range(0, len(yt_real)), yt_real, color='tab:blue')
    fig.savefig(plot_path + "/yt_real.eps", format='eps')

    T = len(yt_real)
    dir0 = dir0 * np.ones(1)

    ### start gibbs
    zt_sample = []
    hyperparam_sample = []
    loglik_test_sample = []

    rho0 = 0
    for it in range(iters):
        if it == 0:
            alpha0, gamma0, dir0, K, zt, beta_vec, beta_new, n_mat, ysum = init_gibbs_full_bayesian_regular(
                alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, dir0, T, yt_real)
        else:
            zt, n_mat, ysum, beta_vec, beta_new, K = sample_zw(zt, yt_real, n_mat, ysum, beta_vec, beta_new,
                                                                     alpha0, gamma0, dir0, rho0, K)
        zt, n_mat, ysum, beta_vec, K = decre_K(zt, n_mat, ysum, beta_vec)
        m_mat = sample_m(n_mat, beta_vec, alpha0, rho0, K)
        m_mat[0, 0] += 1  ## for first time point
        beta_vec, beta_new = sample_beta(m_mat, gamma0)

        ## sample hyperparams
        alpha0 = sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri)
        gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri)

        ## compute loglik
        if it % 10 == 0:
            pi_mat = sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0)
            _, loglik_test = compute_log_marginal_lik_multinomial(K, yt_test, -1, pi_mat, dir0, ysum)
            loglik_test_sample.append(loglik_test)

            zt_sample.append(zt.copy())
            hyperparam_sample.append(np.array([alpha0, gamma0]))

    ## permute result
    mismatch_vec = []
    zt_sample_permute = []
    K_real = len(np.unique(zt_real))
    for ii in range(len(zt_sample)):
        cost, indexes = compute_cost(zt_sample[ii], zt_real)
        dic = dict((v, k) for k, v in indexes)
        tmp = np.array([dic[zt_sample[ii][t]] for t in range(len(zt_sample[0]))])

        zt_sample_permute.append(tmp.copy())
        mismatch_vec.append((tmp != zt_real).sum())

    np.savez(results_path + "/" + filename + "_full_bayesian_gibbs_multinomial_reg.npz", zt=zt_sample,
             hyper=hyperparam_sample,
             hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample)

    test = np.load(results_path + "/" + filename + "_full_bayesian_gibbs_multinomial_reg.npz")

    fig, ax = plt.subplots()
    ax.set_title('zt_test')
    ax.set_xlabel('t')
    ax.set_ylabel('zt_test')
    ax.scatter(range(0, len(np.squeeze(test['zt']))), np.squeeze(test['zt']), color='tab:blue')
    fig.savefig(plot_path + "/zt_test.eps", format='eps')

    print(test['loglik'])


if __name__ == "__main__":
    main()
