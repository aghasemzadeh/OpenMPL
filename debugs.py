import matplotlib.pyplot as plt
fig = plt.figure()
body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
body_edges = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), (5, 6), (11, 12),
             (0, 5), (0, 6), (5, 11), (6, 12))
ax = fig.add_subplot(111)

# Add background rectangle
rect = plt.Rectangle((0, 0), 1280, 720, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
to_plot = points_image_amass
ax.scatter(to_plot[:,0], to_plot[:,1], s=1)
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], color='g', linewidth=1)


ax.axis('equal')
plt.savefig('i2d_.png')






coords = points_image_amass.copy()
coords[:, 0] = (coords[:, 0] - cx) / fx * 1.0
coords[:, 1] = (coords[:, 1] - cy) / fy * 1.0

coords_cam_05 = np.concatenate((0.5*coords, .5 * np.ones((coords.shape[0], 1))), axis=1)
coords_cam = np.concatenate((coords, 1.0 * np.ones((coords.shape[0], 1))), axis=1)
coords_cam_2 = np.concatenate((2*coords, 2.0 * np.ones((coords.shape[0], 1))), axis=1)
coords_cam_3 = np.concatenate((3*coords, 3.0 * np.ones((coords.shape[0], 1))), axis=1)
coords_cam_4 = np.concatenate((4*coords, 4.0 * np.ones((coords.shape[0], 1))), axis=1)
coords_cam_5 = np.concatenate((5*coords, 5.0 * np.ones((coords.shape[0], 1))), axis=1)


T = cameras[view][camera_setup_to_use]['T']
coords_world_05 = np.array((R.T @ coords_cam_05.T + T).T)
coords_world = np.array((R.T @ coords_cam.T + T).T)
coords_world_2 = np.array((R.T @ coords_cam_2.T + T).T)
coords_world_3 = np.array((R.T @ coords_cam_3.T + T).T)
coords_world_4 = np.array((R.T @ coords_cam_4.T + T).T)
coords_world_5 = np.array((R.T @ coords_cam_5.T + T).T)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
body_edges = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
          (13, 15), (5, 6), (11, 12),
            (0, 5), (0, 6), (5, 11), (6, 12))
fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111, projection='3d')
to_plot = joints_3d
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')

to_plot = coords_world_05
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

to_plot = coords_world
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

to_plot = coords_world_2
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

to_plot = coords_world_3
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

to_plot = coords_world_4
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

to_plot = coords_world_5
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='orange')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='r')

ax.scatter(T[0], T[1], T[2], c='black')

ax.scatter(0,0,0, c='gray')

max_range = np.array([to_plot[:, 0].max() - to_plot[:, 0].min(), to_plot[:, 1].max() - to_plot[:, 1].min(), to_plot[:, 2].max() - to_plot[:, 2].min()]).max() / 2.0
x_mean = to_plot[:, 0].mean()
y_mean = to_plot[:, 1].mean()
z_mean = to_plot[:, 2].mean()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.view_init(0, -90, 0)
#ax.view_init(90, 90)
plt.savefig('rays_4.png')
plt.close()

from PIL import Image
im = Image.fromarray(body_image)
im.save("body_image_2.png")



import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16],[8,9],[9,10]])
body_edges = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
          (13, 15), (5, 6), (11, 12),
            (0, 5), (0, 6), (5, 11), (6, 12))
fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111, projection='3d')
to_plot = joints_3d
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], marker='o', s=20, c='b')
for edge in body_edges:
    ax.plot(to_plot[edge,0], to_plot[edge,1], to_plot[edge,2], color='g')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.view_init(0, -90, 0)
#ax.view_init(90, 90)
plt.savefig('j3d.png')
plt.close()