import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

# 🔍 读取 LiDAR 数据
# npy_path = "/home/yixin/Off-road-Benchmark/data/dataset1/lidar_points.npy"
npy_path = "/home/yixin/Off-road-Benchmark/data/dataset1/points_camera.npy"


if not os.path.exists(npy_path):
    raise FileNotFoundError(f"File not found: {npy_path}")

lidar_data = np.load(npy_path)

# 🧹 只保留 XYZ
if lidar_data.shape[1] < 3:
    raise ValueError("LiDAR data must have at least 3 columns (X, Y, Z).")
points = lidar_data[:, :3]

print("💡 Loaded points:", points.shape)
print(f"📏 X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
print(f"📏 Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
print(f"📏 Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")

# ⚠️ 自动缩放（如果点太小）
scale_applied = False
if np.abs(points).max() < 2:
    print("⚠️ Points too small, scaling by 50")
    points *= 50
    scale_applied = True

# 🧱 构建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

# 🎨 根据 Z 值上色
z = points[:, 2]
z_min, z_max = z.min(), z.max()
if z_max - z_min > 1e-5:
    norm_z = (z - z_min) / (z_max - z_min)
else:
    norm_z = np.zeros_like(z)
colors = plt.get_cmap("viridis")(norm_z)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# 🪞 可视化
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="LiDAR Viewer", width=1024, height=768)
vis.add_geometry(pcd)
vis.add_geometry(axis)
render_opt = vis.get_render_option()
render_opt.point_size = 2.5
render_opt.background_color = np.asarray([0, 0, 0])
vis.run()
vis.destroy_window()

# ✅ 提示缩放是否应用
if scale_applied:
    print("🔧 Data was scaled for visibility.")
