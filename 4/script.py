import os
import numpy as np
import statsmodels.api as sm
import matplotlib
from matplotlib import pyplot as plt

# 状態空間時系列分析入門(ref.1) 3章のグラフを再現
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

print('ローカルレベルモデル（確定的レベルと確定的季節要素）の推定')
# レベル擾乱項が0に固定されたモデル
mod_dconstant_seasonal = sm.tsa.UnobservedComponents(ts, 'dconstant', seasonal=12, stochastic_seasonal=False)
res_dconstant_seasonal = mod_dconstant_seasonal.fit(method='powell')

print('観測擾乱項の分散')
print(res_dconstant_seasonal.params)
print('a_t alpha_tの1期先予測')
print(res_dconstant_seasonal.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散')
print(res_dconstant_seasonal.predicted_state_cov[:, :, :10])
print('a_t|t alphatのフィルタ化推定量')
print(res_dconstant_seasonal.filtered_state[:, :10])
print('P_t|t alpha_tの推定誤差分散')
print(res_dconstant_seasonal.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量')
print(res_dconstant_seasonal.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_dconstant_seasonal.smoothed_state_cov[:, :, :10])
print('forecast = a_t')
print(res_dconstant_seasonal.forecasts[:, :10])
print('forecast_error = v_t 観測値の1期先予測誤差')
print(res_dconstant_seasonal.forecasts_error[:, :10])
print('forecast_error_cov = F_t 観測値の1期先予測誤差分散')
print(res_dconstant_seasonal.forecasts_error_cov[:, :, :10])
print('K_t カルマンゲイン')
print(res_dconstant_seasonal.filter_results.kalman_gain[:, :, :10])
print('llf_obs')
print(res_dconstant_seasonal.llf_obs[:20])
print('loglikelihood_burn 散漫初期化要素数')
print(res_dconstant_seasonal.loglikelihood_burn)
print('llf 対数尤度')
print(res_dconstant_seasonal.llf)
print('llf/n')
print(res_dconstant_seasonal.llf / len(ts))


# fig 4.3
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_dconstant_seasonal.smoothed_state[0], label='deterministic constant')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_3.png'))
plt.close("all")

# fig 4.4
plt.plot(res_dconstant_seasonal.seasonal.smoothed, label='deterministic seasonal')
plt.ylim([-0.15, 0.25])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_4.png'))
plt.close("all")

# fig 4.5
plt.plot(ts - res_dconstant_seasonal.smoothed_state[0] - res_dconstant_seasonal.seasonal.smoothed, label='irregular')
plt.ylim([-0.35, 0.35])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_5.png'))
plt.close("all")

print(res_dconstant_seasonal.summary())
fig = res_dconstant_seasonal.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_dconstant_seasonal.png'))
plt.close("all")


print('ローカルレベルモデル（確率的レベルと確率的季節要素）の推定')
mod_local_level_seasonal = sm.tsa.UnobservedComponents(ts, 'local level', seasonal=12, stochastic_seasonal=True)
res_local_level_seasonal = mod_local_level_seasonal.fit(method='powell')

print('観測擾乱項の分散')
print(res_local_level_seasonal.params)
print('a_t alpha_tの1期先予測')
print(res_local_level_seasonal.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散')
print(res_local_level_seasonal.predicted_state_cov[:, :, :10])
print('a_t|t alphatのフィルタ化推定量')
print(res_local_level_seasonal.filtered_state[:, :10])
print('P_t|t alpha_tの推定誤差分散')
print(res_local_level_seasonal.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量')
print(res_local_level_seasonal.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_local_level_seasonal.smoothed_state_cov[:, :, :10])
print('forecast = a_t')
print(res_local_level_seasonal.forecasts[:, :10])
print('forecast_error = v_t 観測値の1期先予測誤差')
print(res_local_level_seasonal.forecasts_error[:, :10])
print('forecast_error_cov = F_t 観測値の1期先予測誤差分散')
print(res_local_level_seasonal.forecasts_error_cov[:, :, :10])
print('K_t カルマンゲイン')
print(res_local_level_seasonal.filter_results.kalman_gain[:, :, :10])
print('llf_obs')
print(res_local_level_seasonal.llf_obs[:20])
print('loglikelihood_burn 散漫初期化要素数')
print(res_local_level_seasonal.loglikelihood_burn)
print('llf 対数尤度')
print(res_local_level_seasonal.llf)
print('llf/n')
print(res_local_level_seasonal.llf / len(ts))

# fig 4.6
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_local_level_seasonal.smoothed_state[0], label='stochastic constant')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_6.png'))
plt.close("all")

# # fig 4.7
plt.plot(res_local_level_seasonal.seasonal.smoothed, label='stochastic seasonal')
plt.ylim([-0.15, 0.25])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_7.png'))
plt.close("all")

# fig 4.9
plt.plot(ts - res_local_level_seasonal.smoothed_state[0] - res_local_level_seasonal.seasonal.smoothed, label='irregular')
plt.ylim([-0.15, 0.15])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/4_9.png'))
plt.close("all")

print(res_local_level_seasonal.summary())
fig = res_local_level_seasonal.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_local_level_seasonal.png'))
plt.close("all")
