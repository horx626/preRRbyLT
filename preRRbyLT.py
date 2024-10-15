import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# # 读取Excel文件作粗处理
# file_path = 'COF3月180天.xlsx'
# sheet_name = 'COF3月180天'
file_path = 'COF1月到4月180天.xlsx'
sheet_name = 'COF1月到4月'
# data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2, usecols='F:GB')
data=pd.read_excel(file_path, sheet_name=sheet_name, usecols='E:GB',nrows=4)
true_data=np.append(data.iloc[1], 0)
data.iloc[1] = data.iloc[1].cumsum()
days = np.arange(1, 181)
def power_func(x, a, b):
    return a * np.power(x, b)
params, covariance = curve_fit(power_func, days, data.iloc[1])
a, b = params
fitted_lt_values = power_func(days, *params)
print(f"拟合幂函数的参数: a = {a:.4f}, b = {b:.4f}")
# 使用sympy定义变量和幂函数
x = sp.symbols('x')
fitted_func = a * x ** b
# 对拟合的幂函数求导
derivative_func = sp.diff(fitted_func, x)
print(f"导函数: {derivative_func}")
# 将导函数转换为可计算的数值形式
derivative_func_num = sp.lambdify(x, derivative_func, 'numpy')
predicted_retention_rate = derivative_func_num(days)

# 原始数据和拟合曲线
plt.subplot(2, 1, 1)
plt.plot(days, data.iloc[1], 'bo', label='实际累积和数据')
plt.plot(days, power_func(days, *params), 'r-', label=f'拟合曲线: y = {a:.4f} * x^{b:.4f}')
plt.xlabel('天数')
plt.ylabel('LT累积和值')
plt.legend()
plt.title('幂函数拟合累积和数据')

#  导数（留存率）曲线
plt.subplot(2, 1, 2)
plt.plot(days,true_data[1:] , 'r-', label='真实留存率')
plt.plot(days, predicted_retention_rate, 'g-', label='预测留存率 (导数)')
plt.xlabel('天数')
plt.ylabel('瞬时留存率')
plt.legend()
plt.title('拟合幂函数的导数（瞬时留存率）')
plt.tight_layout()
plt.show()
print(predicted_retention_rate)