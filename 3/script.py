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

# ローカル線形トレンドモデル
# y_t = mu_t + epsilon_t
# mu_t+1 = mu_t + nu_t + eta_t
# nu_t+1 = nu_t + theta_t


print('ローカル線形トレンドモデル（確率的レベルと確率的傾き）の推定')
mod_local_linear_trend = sm.tsa.UnobservedComponents(ts, 'local linear trend')
res_local_linear_trend = mod_local_linear_trend.fit(method='powell')

print('擾乱項の分散')
print(res_local_linear_trend.params)
print('a_t alpha_tの1期先予測')
print(res_local_linear_trend.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散')
print(res_local_linear_trend.predicted_state_cov[:, :, :10])
print('a_t|t alphatのフィルタ化推定量')
print(res_local_linear_trend.filtered_state[:, :10])
print('P_t|t alpha_tの推定誤差分散')
print(res_local_linear_trend.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量')
print(res_local_linear_trend.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_local_linear_trend.smoothed_state_cov[:, :, :10])
print('llf 対数尤度')
print(res_local_linear_trend.llf)
print('計算確認：AIC = (-2*llf + 2*(q + w))/n qは散漫初期値の数、wは推定される擾乱分散項の数 ref.2 式3.50')
print((-2 * res_local_linear_trend.llf + 2 * (
    res_local_linear_trend.loglikelihood_burn + len(res_local_linear_trend.params))) / len(ts))
print(res_local_linear_trend.summary())

# fig 3.1
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_local_linear_trend.smoothed_state[0], label='stochastic level and slope')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/3_1.png'))
plt.close("all")

# fig 3.2
# tsueta_tの分散の推定値がref.1よりも小さいため、ref.1の図のように変化しない。
plt.plot(res_local_linear_trend.smoothed_state[1][1:], label='stochastic slope')
plt.ylim([0.0002888, 0.0002890])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/3_2.png'))
plt.close("all")

# fig 3.3
plt.plot(ts - res_local_linear_trend.smoothed_state[0], label='irregular')
plt.ylim([-0.07, 0.07])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/3_3.png'))
plt.close("all")

fig = res_local_linear_trend.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_local_linear_trend.png'))
plt.close("all")

print('ローカル線形トレンドモデル（確率的レベルと確定的傾き）の推定')
mod_local_linear_deterministic_trend = sm.tsa.UnobservedComponents(ts, 'local linear deterministic trend')
res_local_linear_deterministic_trend = mod_local_linear_deterministic_trend.fit(method='bfgs')

print('sigma 各擾乱項の分散')
print(res_local_linear_deterministic_trend.params, )
print('a_t alpha_tの1期先予測')
print(res_local_linear_deterministic_trend.predicted_state[:, :10])
print('P_t alpha_tの1期先予測誤差分散')
print(res_local_linear_deterministic_trend.predicted_state_cov[:, :, :10])
print('a_t|t alpha_tのフィルタ化推定量')
print(res_local_linear_deterministic_trend.filtered_state[:, :10])
print('P_t|t alpha_tの推定誤差分散')
print(res_local_linear_deterministic_trend.filtered_state_cov[:, :, :10])
print('a_t_hat alpha_tの平滑化状態量 ')
print(res_local_linear_deterministic_trend.smoothed_state[:, :10])
print('P_t_hat alpha_tの平滑化状態量分散')
print(res_local_linear_deterministic_trend.smoothed_state_cov[:, :, :10])
print('llf 対数尤度')
print(res_local_linear_deterministic_trend.llf)
print('AIC = (-2*llf + 2*(q + w))/n')
print((-2 * (res_local_linear_deterministic_trend.llf) + 2 * (
    res_local_linear_deterministic_trend.loglikelihood_burn + len(res_local_linear_deterministic_trend.params))) / len(
    ts))

# fig 3.4
plt.plot(ts, label='log UK driver KSI')
plt.plot(res_local_linear_deterministic_trend.smoothed_state[0], label='stochastic level, deterministic slope')
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/3_4.png'))
plt.close("all")

print(res_local_linear_deterministic_trend.summary())
fig = res_local_linear_deterministic_trend.plot_components()
fig.savefig(os.path.join(current_dir, 'result/statsmodels_local_linear_deterministic_trend.png'))
plt.close("all")
