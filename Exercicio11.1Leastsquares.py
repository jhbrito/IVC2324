# Least squares solving
# y = m x + b
# y0 = m x0 + b
# y1 = m x1 + b
# y2 = m x2 + b
# y3 = m x3 + b
# y4 = m x4 + b
# equivalent to A p = y
# with
# A = [x0 1
#      x1 1
#      x2 1
#      x3 1
#      x4 1]
#  p = [m
#       b]

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng(1)

N = 100
x = np.array(range(N))
m = 0.5
b = 1

noise_std = 0.1
y = x * m + b
noise = rng.standard_normal(len(y)) * noise_std
y_with_noise = y + noise

A = np.array([x, np.ones(len(x))]).T
solution = np.linalg.lstsq(a=A, b=y_with_noise)
# m_est = solution[0][0]
# b_est = solution[0][1]
m_est, b_est = solution[0]

y_est = x * m_est + b_est


plt.plot(x, y, 'k', label='Original data')
plt.plot(x, y_with_noise, 'rx', label='Noisy data')
plt.plot(x, y_est, 'g', label='Fitted line')
plt.legend()
plt.xlim([-1, N])
plt.ylim([0, N/2 + b])
plt.show()
