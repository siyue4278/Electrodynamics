import numpy as np
from scipy.optimize import brentq

def electric_field_at_point(point, a, n):
    Ex, Ey = 0.0, 0.0
    for k in range(n):
        theta = 2 * np.pi * k / n
        rx, ry = a * np.cos(theta), a * np.sin(theta)

        dx, dy = point[0] - rx, point[1] - ry
        dist_sq = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_sq)

        factor = 1.0 / (dist_sq * dist)
        Ex += dx * factor
        Ey += dy * factor
    return np.array([Ex, Ey])

def find_zero_field_eadius(n, a):
    angle = np.pi/n
    def radial_field(r):
        if r <= 0 or r >= a:
            return 1e9
        point = (r * np.cos(angle), r * np.sin(angle))
        E = electric_field_at_point(point, a, n)
        if E is None:
            return 1e9
        er = np.array([np.cos(angle), np.sin(angle)])
        return np.dot(E, er)

    try:
            r_zero = brentq(radial_field, 1e-4,a - 1e-4)
            return r_zero
    except ValueError:
         return None

for n in [420]:
     r = find_zero_field_eadius(n, 1.0)
     print(f"对于{n}边形, r = {r}")