import os
import numpy as np
from numpy.random import randn, seed, multinomial

import matplotlib.pyplot as plt

from src.simulate_data import sample_same_trans, sample_same_stick
from src.gibbs_gaussian import *
from src.permute import compute_cost
from src.util import sample_pi_efox, compute_log_marginal_lik_gaussian

seed(42)


# def generate_data_old(file_path, nof_states, p_1=1, p_2=0, p_3=1, p_4=0, steps=50):
#     zt_real, wt_real, kappa_real, trans_vec = sample_same_trans(K_real=nof_states, p_real1=p_1, p_real2=p_2,
#                                                                 p_real3=p_3, p_real4=p_4, T=steps)
#     dic_real = np.diag(np.ones(2)) * 0.7 + (0.3 / 2)
#     dic_real = dic_real / dic_real.sum(axis=1, keepdims=True)
#     yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))])
#     trans_mat = trans_vec * np.expand_dims(1 - kappa_real, axis=-1) + np.diag(
#         kappa_real)  ## compute the real transtion matrix
#
#     np.savez(file_path, zt=zt_real, wt=wt_real, kappa=kappa_real,
#              yt=yt_real, dic=dic_real, trans_mat=trans_mat)


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


def save_plots(plot_path, zt_real, yt_real, zt_test):
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

    fig, ax = plt.subplots()
    ax.set_title('zt_test')
    ax.set_xlabel('t')
    ax.plot(zt_test, 'tab:blue')
    fig.savefig(plot_path + "/zt_test.eps", format='eps')


def run_gibbs(data, init_args, rho0=0):
    alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, sigma0, mu0, sigma0_pri, T, iters = init_args
    yt_real, zt_real, yt_test = data

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
        m_mat[0, 0] += 1  # for first time point
        beta_vec, beta_new = sample_beta(m_mat, gamma0)

        # sample hyperparams
        alpha0 = sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri)
        gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri)

        # compute loglik
        if it % 10 == 0:
            pi_mat = sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0)
            _, loglik_test = compute_log_marginal_lik_gaussian(K, yt_test, -1, pi_mat, mu0, sigma0, sigma0_pri, ysum,
                                                               ycnt)
            loglik_test_sample.append(loglik_test)

            zt_sample.append(zt.copy())
            hyperparam_sample.append(np.array([alpha0, gamma0]))

    # permute result
    mismatch_vec = []
    zt_sample_permute = []
    for ii in range(len(zt_sample)):
        cost, indexes = compute_cost(zt_sample[ii], zt_real)
        dic = dict((v, k) for k, v in indexes)
        tmp = np.array([dic[zt_sample[ii][t]] for t in range(len(zt_sample[0]))])

        zt_sample_permute.append(tmp.copy())
        mismatch_vec.append((tmp != zt_real).sum())

    return zt_sample, hyperparam_sample, mismatch_vec, zt_sample_permute, loglik_test_sample


def main():
    root = os.getcwd()
    data_path = root + "/io/data"
    plot_path = root + "/io/plots"
    results_path = root + "/io/results"

    filename = "rl_data"

    file_path = data_path + "/" + filename + ".npz"
    test_file_path = data_path + "/" + filename + ".npz"

    # data generation config
    p_1 = 1  # P(staying in state 0 if currently in state 0)
    p_2 = 0  # P(staying in state 1 if currently in state 1)
    p_3 = 1  # P(going to state 0 if state transition occurs)
    p_4 = 0  # P(going to state 1 if state transition occurs)

    p_y = 0.85  # P(observing real hidden state)
    steps = 100
    states_ratio = 1
    args = [p_1, p_2, p_3, p_4, p_y, steps, states_ratio]

    generate_data(file_path, args)

    # load data
    dat = np.load(file_path)
    zt_real, yt_real = dat['zt'], dat['yt']  # yt_real & zt_real are 1d length T1 numpy array
    test_dat = np.load(test_file_path)
    yt_test = test_dat['yt']  # yt_test is 1d length T2 numpy array

    # gibbs config
    iters = 10
    sigma0 = 0.1
    alpha0_a_pri = 1
    alpha0_b_pri = 0.01
    gamma0_a_pri = 2
    gamma0_b_pri = 1

    T = len(yt_real)
    mu0 = np.mean(yt_real)
    sigma0_pri = np.std(yt_real)

    init_args = [alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, sigma0, mu0, sigma0_pri, T, iters]
    data = [yt_real, zt_real, yt_test]

    # start gibbs
    zt_sample, hyperparam_sample, mismatch_vec, zt_sample_permute, loglik_test_sample = run_gibbs(data, init_args)

    # save results
    np.savez(results_path + "/" + filename + "_full_bayesian_gibbs_gaussian_reg.npz", zt=zt_sample,
             hyper=hyperparam_sample,
             hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample)

    test = np.load(results_path + "/" + filename + "_full_bayesian_gibbs_gaussian_reg.npz")

    # save plots
    save_plots(plot_path=plot_path, zt_real=zt_real, yt_real=yt_real, zt_test=np.squeeze(test['zt']))

    print(test['loglik'])
    
    fig, ax = plt.subplots()
    ax.set_title('yt_test')
    ax.set_xlabel('t')
    ax.plot(np.squeeze(test['loglik']), 'tab:blue')
    fig.savefig(plot_path + "/loglik_test.eps", format='eps')


if __name__ == "__main__":
    main()
