# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import scipy.optimize as optimize


# 状態空間モデルにおける尤度計算
#
# 参考書籍(数式)
# カルマンフィルタ Rを使った時系列予測と状態空間モデル
#
# 参考サイト
# https://logics-of-blue.com/kalman-filter-mle/


# ローカルモデルのフィルタリング
def filtering_local_level_model(y, x_pre, p_pre, sigma_w, sigma_v):
    # 1期先予測 a_t 式(2.5)
    x_forecast = x_pre

    # 1期先予測の予測誤差分散 P_t 式(2.5)
    p_forecast = p_pre + sigma_w

    # 観測値の1期先予測誤差 v_t 式(2.8)
    v = y - x_forecast

    # 観測値の1期先予測誤差分散 F_t 式(2.8)
    f = p_forecast + sigma_v

    # カルマンゲイン
    k_gain = p_forecast / (p_forecast + sigma_v)

    # 状態のフィルタ化推定量 a_t|t
    x_filtered = x_forecast + k_gain * (y - x_forecast)

    # 状態のフィルタ化推定量の推定誤差分散 P_t|t
    p_filtered = (1 - k_gain) * p_forecast

    return {
        'x_filtered': x_filtered,
        'p_filtered': p_filtered,
        'f': f,
        'v': v,
    }


# 対数尤度（散漫対数尤度）
def calc_log_likelihood(n, f, v):
    # 式(2.27)
    return -0.5 * n * math.log(2 * math.pi) - 0.5 * np.sum(np.log(f[1:]) + v[1:] ** 2 / f[1:])


# ローカルレベルの対数尤度を計算する
def calc_filtering_local_level_model(data, raw_w, raw_v):
    # 分散(未知数)
    # 最適化の際、負にならないようexpをとる。
    sigma_w = math.exp(raw_w)
    sigma_v = math.exp(raw_v)

    n = len(data)
    # 状態量の1期先予測
    x = np.zeros(n + 1)

    # 状態量の1期先予測誤差分散
    # 信頼区間は 下限 x - 1.96 * sqrt(P)、上限 x + 1.96 * sqrt(P) となる。 p.29
    p = np.zeros(n + 1) + 10000000  # 散漫初期化 p.42

    # 観測値の1期先予測誤差 p.26
    v = np.zeros(n)

    # 観測値の1期先予測誤差分散 p.26
    f = np.zeros(n)

    for i in range(n):
        filtered_data = filtering_local_level_model(data[i], x[i], p[i], sigma_w, sigma_v)
        x[i + 1] = filtered_data['x_filtered']
        p[i + 1] = filtered_data['p_filtered']
        v[i] = filtered_data['v']
        f[i] = filtered_data['f']

    return {
        'x': x[1:],
        'p': p[1:],
        'f': f,
        'v': v,
    }


def optimize_local_level_model_func(data):
    # optimize.minimize()で使うために符号を反転させる and 引数を配列化
    def minimize_func(sigma):
        filtered_values = calc_filtering_local_level_model(data, sigma[0], sigma[1])

        return -1 * calc_log_likelihood(len(data), filtered_values['f'], filtered_values['v'])

    return minimize_func


if __name__ == '__main__':
    print('ナイル川の流量の状態空間モデルにおける尤度計算')

    # ナイル川の流量データ読み込み
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ts = np.loadtxt(os.path.normpath(os.path.join(current_dir, '../original_data/nile.csv')), delimiter=',',
                    skiprows=1)[:, 1]

    # 目的変数
    objective_variable_w = 6.9077  # exp(6.9077) -> 1000
    objective_variable_v = 9.2103  # exp(9.2103) -> 10000
    filtered = calc_filtering_local_level_model(ts, objective_variable_w, objective_variable_v)

    print('対数尤度')
    print(calc_log_likelihood(len(ts), filtered['f'], filtered['v']))  # -646.326482021

    # 最尤法によるパラメタ推定
    # 対数尤度を最大化するsigma_w, sigma_vを見つける
    # 今回の問題ではmethod='BFGS'でやる際は、初期値x0によっては収束しなかったり間違った値に収束した。
    opt = optimize.minimize(fun=optimize_local_level_model_func(ts), x0=np.array([0, 0]), method="Powell",
                            options={'maxiter': 50})

    print('最適化結果')
    print(opt)
    # direc: array([[-1.77986506, 1.20449534],
    #               [-0.26409706, 0.01680339]])
    # fun: 641.58564268944258
    # message: 'Optimization terminated successfully.'
    # nfev: 146
    # nit: 5
    # status: 0
    # success: True
    # x: array([ 7.29202163,  9.62239557])

    print('sigma_w', math.exp(opt.x[0]))  # sigma_w 1468.4684408798098
    print('sigma_v', math.exp(opt.x[1]))  # sigma_v 15099.71130365184

    # 以上より求めたパラメタを使ってフィルタリング、スムージングを行う。
