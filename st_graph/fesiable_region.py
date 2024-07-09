#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np


t = np.arange(0, 8, 0.1)
s0 = 5.0
v0 = 20.0
a0 = 0.0
a_min = -6.0
a_max = 4.0

t_zero_speed = v0 / -a_min
s_lower_bound = np.zeros_like(t)
mask = t < t_zero_speed

s_lower_bound = np.where(mask, 
                         s0 + v0 * t + 0.5 * a_min * t ** 2, 
                         s0 + v0 ** 2 / (-a_min * 2))
s_upper_bound = s0 + v0 * t + 0.5 * a_max * t ** 2

plt.plot(t, s_lower_bound, 'r')
plt.plot(t, s_upper_bound, 'b')
plt.show()