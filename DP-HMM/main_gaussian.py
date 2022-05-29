import os
import numpy as np
from numpy.random import randn, seed, multinomial

import matplotlib.pyplot as plt

from src.simulate_data import sample_same_trans, sample_same_stick
from src.gibbs_gaussian import *
from src.permute import compute_cost
from src.util import sample_pi_efox, compute_log_marginal_lik_gaussian


seed(42)

def generate_data(file_path, nof_states, p_1, p_2, p_3, p_4, steps):
    zt_real, wt_real, kappa_real, trans_vec = sample_same_trans(K_real=nof_states, p_real1=p_1, p_real2=p_2,
                                                                p_real3=p_3, p_real4=p_4, T=steps)
    dic_real = np.diag(np.ones(2)) * 0.7 + (0.3 / 2)
    dic_real = dic_real / dic_real.sum(axis=1, keepdims=True)
    yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))])
    trans_mat = trans_vec * np.expand_dims(1 - kappa_real, axis=-1) + np.diag(
        kappa_real)  ## compute the real transtion matrix

    np.savez(file_path, zt=zt_real, wt=wt_real, kappa=kappa_real,
             yt=yt_real, dic=dic_real, trans_mat=trans_mat)


def main():
    data_path = "./io/data"
    plot_path = "./io/plots"
    results_path = "./io/results"

    filename = "2_states_test"

    args = {"nof_states": 2,
            "p_1": 0.96,
            "p_2": 1,
            "p_3": 1,
            "p_4": 0,
            "steps": 1000}

    file_path = data_path + "/" + filename + ".npz"

    generate_data(file_path, **args)

    iters = 10
    sigma0 = 0.1
    alpha0_a_pri = 1
    alpha0_b_pri = 0.01
    gamma0_a_pri = 2
    gamma0_b_pri = 1

    path = data_path + "/fix_8states_gaussian_same_trans_diff_stick.npz"
    path_test = data_path + "/test_fix_8states_gaussian_same_trans_diff_stick.npz"

    ## load data
    dat = np.load(path)
    zt_real, yt_real = dat['zt'], dat['yt']  ## yt_real & zt_real are 1d length T1 numpy array
    test_dat = np.load(path_test)
    yt_test = test_dat['yt']  ## yt_test is 1d length T2 numpy array

    fig, ax = plt.subplots()
    ax.set_title('zt_real')
    ax.set_xlabel('t')
    ax.plot(zt_real, 'tab:blue')
    fig.savefig(plot_path + "/zt_real.eps", format='eps')

    fig, ax = plt.subplots()
    ax.set_title('yt_real')
    ax.set_xlabel('t')
    ax.plot(yt_real, 'tab:blue')
    fig.savefig(plot_path + "/yt_real.eps", format='eps')

    T = len(yt_real)
    mu0 = np.mean(yt_real)
    sigma0_pri = np.std(yt_real)

    ### start gibbs

    rho0 = 0
    zt_sample = []
    hyperparam_sample = []
    loglik_test_sample = []

    for it in range(iters):
        if it == 0:
            alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, beta_vec, beta_new, n_mat, ysum, ycnt = init_gibbs_full_bayesian_regular(
                alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, sigma0, mu0, sigma0_pri, T, yt_real)
        else:
            zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_zw(zt, yt_real, n_mat, ysum, ycnt, beta_vec, beta_new,
                                                                     alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, K)
        zt, n_mat, ysum, ycnt, beta_vec, K = decre_K(zt, n_mat, ysum, ycnt, beta_vec)
        m_mat = sample_m(n_mat, beta_vec, alpha0, rho0, K)
        m_mat[0, 0] += 1  ## for first time point
        beta_vec, beta_new = sample_beta(m_mat, gamma0)

        ## sample hyperparams
        alpha0 = sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri)
        gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri)

        ## compute loglik
        if it % 10 == 0:
            pi_mat = sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0)
            _, loglik_test = compute_log_marginal_lik_gaussian(K, yt_test, -1, pi_mat, mu0, sigma0, sigma0_pri, ysum,
                                                               ycnt)
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

    np.savez(results_path + "/" + filename + "_full_bayesian_gibbs_gaussian_reg.npz", zt=zt_sample,
             hyper=hyperparam_sample,
             hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample)

    test = np.load(results_path + "/" + filename + "_full_bayesian_gibbs_gaussian_reg.npz")

    fig, ax = plt.subplots()
    ax.set_title('zt_test')
    ax.set_xlabel('t')
    ax.plot(np.squeeze(test['zt']), 'tab:blue')
    fig.savefig(plot_path + "/zt_test.eps", format='eps')

    print(test['loglik'])
    
    fig, ax = plt.subplots()
    ax.set_title('yt_test')
    ax.set_xlabel('t')
    ax.plot(np.squeeze(test['loglik']), 'tab:blue')
    fig.savefig(plot_path + "/loglik_test.eps", format='eps')


if __name__ == "__main__":
    main()
