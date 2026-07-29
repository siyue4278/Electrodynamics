import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt
N = 21
V0 = 100.0
V = np.zeros((N, N, N))
V[:, :, -1] = V0

omega = 2.0 / (1.0 + np.sin(np.pi / N))
print(f"理论最优超松弛因子 omega = {omega:.4f}")

@jit(nopython=True)
def solve_laplace_sor(V, omega, max_iter, tol):
    for i in range(max_iter):
        max_diff = 0.0
        for x in range(1, N-1):
            for y in range(1, N-1):
                for z in range(1, N-1):
                    v_old = V[x, y, z]
                    
                    v_new = (V[x-1, y, z] + V[x+1, y, z] +
                             V[x, y-1, z] + V[x, y+1, z] +
                             V[x, y, z-1] + V[x, y, z+1]) / 6.0
                    
                    V[x, y, z] = (1.0 - omega) * v_old + omega * v_new
                    
                    diff = abs(V[x, y, z] - v_old)
                    if diff > max_diff:
                        max_diff = diff
                        
        if max_diff < tol:
            return i, max_diff
            
    return max_iter, max_diff

print("Numba + SOR 加速计算中...")
start = time.time()
iters, err = solve_laplace_sor(V, omega, 10000, 1e-5)
end = time.time()

print(f"在 {iters} 次迭代后收敛,最大误差 {err:.6e}")
print(f"总耗时: {end - start:.4f} 秒")

center_idx = N // 2
print("物理结论检验")
print(f"数值解法: {V[center_idx, center_idx, center_idx]:.4f} V")
print(f"理论上的电势: {V0 / 6:.4f} V")

#可视化部分


slice_V = V[:, center_idx, :]

plt.figure(figsize=(8, 6))

plt.imshow(slice_V.T, cmap = "jet", origin = "lower", extent = [0, 1, 0, 1])
plt.colorbar(label="Potential(V)")
plt.title("Potential cross-section  at y = a/2")
plt.xlabel("x/a")
plt.ylabel("z/a")

plt.axhline(1.0, color="red", linewidth=3, label="V = V0")
plt.plot(0.5, 0.5, "w*", markersize=10, label="Center (V0/6)")
plt.legend(loc="lower right")

plt.show()