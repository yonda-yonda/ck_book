import os
import numpy as np
import statsmodels.api as sm
import matplotlib
from matplotlib import pyplot as plt

# 状態空間時系列分析入門(ref.1) 2章のグラフを再現
# 数式はカルマンフィルタ Rを使った時系列予測と状態空間モデル(ref.2)より
matplotlib.use('tkagg')
plt.rcParams['figure.figsize'] = 16, 9

# 英国の交通事故死(killed or seriously injured)データ読み込み
current_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.normpath(os.path.join(current_dir, '../original_data/UKdriversKSI.txt')), skiprows=1,
                  delimiter='\n')
ts = np.log(data)

# ローカルレベル
# y_t = alpha_t + epsilon_t
# alpha_t+1 = alpha_t + eta_t

print('ローカルレベルモデル（確定的レベル）の推定')
# レベル擾乱項が0に固定されたモデル
mod_dconstant = sm.tsa.UnobservedComponents(ts, 'dconstant')
res_dconstant = mod_dconstant.fit(method='bfgs')

print('sigma_e2 観測擾乱項の分散')
print(res_dconstant.params)
print('a_t alpha_tの1期先予測（散漫初期化によりa_1=0）')
print(res_dconstant.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散（散漫初期化によりP_1=10^6）')
print(res_dconstant.predicted_state_cov[:, :, :10])
print('a_t|t alphatのフィルタ化推定量')
print(res_dconstant.filtered_state[:, :10])
print('計算確認：a_1|1 = a_1 + P_1 / (P_1 + sigma_e2) * (y_1 - a_1)')
print(res_dconstant.predicted_state[0, 0] + res_dconstant.predicted_state_cov[0, 0, 0] / (
    res_dconstant.predicted_state_cov[0, 0, 0] + res_dconstant.params[0]) * (
          ts[0] - res_dconstant.predicted_state[0, 0]))
print('P_t|t alpha_tの推定誤差分散')
print(res_dconstant.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量')
print(res_dconstant.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_dconstant.smoothed_state_cov[:, :, :10])
print('forecast = a_t')
print(res_dconstant.forecasts[:, :10])
print('forecast_error = v_t 観測値の1期先予測誤差')
print(res_dconstant.forecasts_error[:, :10])
print('計算確認：v_t = y_t - a_t')
print(ts[:10] - res_dconstant.predicted_state[0, :10])
print('forecast_error_cov = F_t 観測値の1期先予測誤差分散')
print(res_dconstant.forecasts_error_cov[:, :, :10])
print('計算確認：F_t = P_t + sigma_e^2')
print(res_dconstant.predicted_state_cov[:, :, :10] + res_dconstant.params[0])
print('K_t カルマンゲイン')
print(res_dconstant.filter_results.kalman_gain[:, :, :10])
print('計算確認：K_t = P_t/F_t')
print(res_dconstant.predicted_state_cov[:, :, :10] / res_dconstant.forecasts_error_cov[:, :, :10])
print('llf_obs')
print(res_dconstant.llf_obs[:10])
print('計算確認：llf_obs = -0.5 * log(2pi) - 0.5 * (log(F_t) + v_t^2 / F_t)')
print(-0.5 * np.log(2 * np.pi) - 0.5 * (
    np.log(res_dconstant.forecasts_error_cov[:, :, :10]) + res_dconstant.forecasts_error[:, :10] ** 2 /
    res_dconstant.forecasts_error_cov[:, :, :10]))
print('loglikelihood_burn 散漫初期化要素数')
print(res_dconstant.loglikelihood_burn)
print('llf 対数尤度')
print(res_dconstant.llf)  # 123.877629181
print('llf/n ref.1で求める対数尤度')
print(res_dconstant.llf / len(ts))
print('計算確認：llf = sum(llf_obs[loglikelihood_burn:]) ref.1 式8.7より -0.5*log(2pi) 分ずれる')
print(np.sum(res_dconstant.llf_obs[res_dconstant.loglikelihood_burn:]))
print('計算確認：AIC = (-2*llf + 2*(q + w))/n qは散漫初期値の数、wは推定される擾乱分散項の数 ref.2 式3.50')
print((-2 * res_dconstant.llf + 2 * (res_dconstant.loglikelihood_burn + len(res_dconstant.params))) / len(ts))
print('計算確認：UnobservedComponents AIC')
print(-2 * res_dconstant.llf + 2 * len(res_dconstant.params))

print(res_dconstant.summary())

# fig 2.1
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_dconstant.smoothed_state[0], label='deterministic constant')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/2_1.png'))
plt.close("all")

# fig 2.2
plt.plot(ts - res_dconstant.smoothed_state[0], label='irregular')
plt.ylim([-0.5, 0.5])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/2_2.png'))
plt.close("all")

fig = res_dconstant.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_dconstant.png'))
plt.close("all")

print('ローカルレベルモデル（確率的レベル）の推定')
mod_local_level = sm.tsa.UnobservedComponents(ts, 'local level')
res_local_level = mod_local_level.fit(method='bfgs')

print('sigma_e^2 観測擾乱項の分散, sigma_z^2 レベル擾乱項の分散')
print(res_local_level.params)
print('a_t alpha_tの1期先予測（散漫初期化によりa_1=0）')
print(res_local_level.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散（散漫初期化によりP_1=10^6）')
print(res_local_level.predicted_state_cov[:, :, :10])
print('a_t|t alphatのフィルタ化推定量')
print(res_local_level.filtered_state[:, 10])
print('計算確認：a_1|1 = a_1 + P_1 / (P_1 + sigma_e2) * (y_1 - a_1)')
print(res_local_level.predicted_state[0, 0] + res_local_level.predicted_state_cov[0, 0, 0] / (
    res_local_level.predicted_state_cov[0, 0, 0] + res_local_level.params[0]) * (
          ts[0] - res_local_level.predicted_state[0, 0]))
print('P_t|t alpha_tの推定誤差分散')
print(res_local_level.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量')
print(res_local_level.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_local_level.smoothed_state_cov[:, :, :10])
print('forecast = a_t')
print(res_local_level.forecasts[:, :10])
print('forecast_error = v_t 観測値の1期先予測誤差')
print(res_local_level.forecasts_error[:, :10])
print('計算確認：v_t = y_t - a_t')
print(ts[:10] - res_local_level.predicted_state[0, :10])
print('forecast_error_cov = F_t 観測値の1期先予測誤差分散')
print(res_local_level.forecasts_error_cov[:, :, :10])
print('計算確認：F_t = P_t + sigma_e^2')
print(res_local_level.predicted_state_cov[:, :, :10] + res_local_level.params[0])
print('K_t カルマンゲイン')
print(res_local_level.filter_results.kalman_gain[:, :, :10])
print('計算確認：K_t = P_t/F_t')
print(res_local_level.predicted_state_cov[:, :, :10] / res_local_level.forecasts_error_cov[:, :, :10])
print('llf_obs')
print(res_local_level.llf_obs[:10])
print('計算確認：llf_obs = -0.5 * log(2pi) - 0.5 * (log(F_t) + v_t^2 / F_t)')
print(-0.5 * np.log(2 * np.pi) - 0.5 * (
    np.log(res_local_level.forecasts_error_cov[:, :, :10]) + res_local_level.forecasts_error[:, :10] ** 2 /
    res_local_level.forecasts_error_cov[:, :, :10]))
print('loglikelihood_burn 散漫初期化要素数')
print(res_local_level.loglikelihood_burn)
print('llf 対数尤度 ref.1ではllf/n')
print(res_local_level.llf)  # 123.877629181
print('llf/n ref.1で求める対数尤度')
print(res_local_level.llf / len(ts))
print('計算確認：llf = sum(llf_obs[loglikelihood_burn:]) ref.1 式8.7より-0.5*log(2pi)分ずれる')
print(np.sum(res_local_level.llf_obs[res_local_level.loglikelihood_burn:]))
print('計算確認：AIC = (-2*llf + 2*(q + w))/n qは散漫初期値の数、wは推定される擾乱分散項の数 ref.2 式3.50')
print((-2 * res_local_level.llf + 2 * (res_local_level.loglikelihood_burn + len(res_local_level.params))) / len(ts))
print('計算確認：UnobservedComponents AIC')
print(-2 * res_local_level.llf + 2 * len(res_local_level.params))

print(res_local_level.summary())

# fig 2.3
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_local_level.smoothed_state[0], label='stochastic level')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/2_3.png'))
plt.close("all")

# fig 2.4
plt.plot(ts - res_local_level.smoothed_state[0], label='irregular')
plt.ylim([-0.07, 0.07])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/2_4.png'))
plt.close("all")

fig = res_local_level.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_local_level.png'))
plt.close("all")
