import os
import numpy as np
import statsmodels.api as sm
import matplotlib
from matplotlib import pyplot as plt

# 状態空間時系列分析入門1章のグラフを再現
matplotlib.use('tkagg')
plt.rcParams['figure.figsize'] = 16, 9

# 英国の交通事故死(killed or seriously injured)データ読み込み
current_dir = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(os.path.normpath(os.path.join(current_dir, '../original_data/UKdriversKSI.txt')), skiprows=1,
                  delimiter='\n')
ts = np.log(data)

# 対数の線形回帰
x_ts = np.array(range(len(ts)))
x_prop = np.polyfit(x_ts, ts, 1)
ts_ols = x_prop[0] * x_ts + x_prop[1]

print('fig1.1 ドライバーの死者数の対数と線形回帰線')
ts_legend = 'the logarithm of monthly UK drivers KSI 1969-1984'
ts_ols_legend = 'regression'
plt.scatter(x_ts, ts, label=ts_legend, s=50, marker='+')
plt.plot(x_ts, ts_ols, label=ts_ols_legend)
plt.legend()
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.plot(x_ts, ts_ols, label=ts_ols_legend)
plt.savefig(os.path.join(current_dir, 'result/drivers_ksi_regression.png'))
plt.close("all")

print('fig1.2 ドライバーの死者数の対数の時系列')
plt.plot(ts, label=ts_legend)
plt.ylim([6.95, 7.9])
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/drivers_ksi.png'))
plt.close("all")

# 線形回帰との残差
ts_residual = ts - ts_ols

print('fig1.3 ドライバーの死者数の対数の線形回帰残差')
ts_residual_legend = 'residuals'
plt.plot(x_ts, ts_residual, label=ts_residual_legend)
plt.ylim([-0.34, 0.42])
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.xlim([0, 200])
plt.xticks(range(0, 200, 20))
plt.savefig(os.path.join(current_dir, 'result/drivers_ksi_residual.png'))
plt.close("all")

# 偏自己相関
ts_acf = sm.tsa.stattools.acf(ts_residual, nlags=14)[1:]
x_acf = range(1, len(ts_acf) + 1)

print('fig1.5 コレログラム')
plt.bar(x_acf, ts_acf, width=0.5)
plt.ylim([-1, 1])
plt.xlim([0, 15])
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.5)
plt.hlines([-0.144, 0.144], 0, 15, linestyles="dashed", color='black', label='ACF regression residuals', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(current_dir, 'result/drivers_ksi_acf.png'))
plt.close("all")
