import torch
import sys
sys.path.insert(0, '.')

stats = torch.load('data/statistics_new.pth', weights_only=False)

print("=== 卫星数据统计 (56维) ===\n")

# 索引 0-19
print("索引 0-19 (惯性+质量+质心+轨道+太阳能板):")
for i in range(20):
    print(f"[{i:2d}] mean={stats.constellation_mean[i]:10.4f}  std={stats.constellation_std[i]:10.4f}")

# 索引 20-43
print("\n索引 20-43 (传感器+电池+反作用轮):")
for i in range(20, 44):
    print(f"[{i:2d}] mean={stats.constellation_mean[i]:10.4f}  std={stats.constellation_std[i]:10.4f}")

# 索引 44-55
print("\n索引 44-55 (MRP控制器+动态数据):")
for i in range(44, 56):
    print(f"[{i:2d}] mean={stats.constellation_mean[i]:10.4f}  std={stats.constellation_std[i]:10.4f}")

print("\n=== 关键物理参数 ===")
print(f"惯性矩 Ixx [0]: mean={stats.constellation_mean[0]:.2f}, std={stats.constellation_std[0]:.2f}")
print(f"惯性矩 Iyy [4]: mean={stats.constellation_mean[4]:.2f}, std={stats.constellation_std[4]:.2f}")
print(f"惯性矩 Izz [8]: mean={stats.constellation_mean[8]:.2f}, std={stats.constellation_std[8]:.2f}")
print(f"质量 [9]: mean={stats.constellation_mean[9]:.2f}, std={stats.constellation_std[9]:.2f}")
print(f"反作用轮1最大角动量 [29]: mean={stats.constellation_mean[29]:.4f}, std={stats.constellation_std[29]:.4f}")
print(f"反作用轮2最大角动量 [35]: mean={stats.constellation_mean[35]:.4f}, std={stats.constellation_std[35]:.4f}")
print(f"反作用轮3最大角动量 [41]: mean={stats.constellation_mean[41]:.4f}, std={stats.constellation_std[41]:.4f}")
print(f"MRP姿态 [53]: mean={stats.constellation_mean[53]:.6f}, std={stats.constellation_std[53]:.6f}")
print(f"MRP姿态 [54]: mean={stats.constellation_mean[54]:.6f}, std={stats.constellation_std[54]:.6f}")
print(f"MRP姿态 [55]: mean={stats.constellation_mean[55]:.6f}, std={stats.constellation_std[55]:.6f}")
