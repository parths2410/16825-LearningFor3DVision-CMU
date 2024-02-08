import numpy as np
import imageio
from matplotlib import pyplot as plt

import torch
import pytorch3d


cow_path = "data/cow.obj"
image_size = 256

# 1. Practicing with Cameras
print("1. Practicing with Cameras")
## 1.1 360-degree renders
print("1.1 360-degree renders")
from starter.render_mesh import render360
output_path = "output/cow_360_render.gif"
images = render360(obj_path=cow_path, image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

## 1.2 Recreating the dolly zoom
print("1.2 Recreating the dolly zoom")
from starter.dolly_zoom import dolly_zoom
output_path = "output/dolly.gif"
dolly_zoom(image_size=image_size, num_frames=120, duration=3, output_file=output_path)

# 2. Practicing with Meshes
print("2. Practicing with Meshes")
## 2.1 Construcitng a Tetrahedron
print("2.1 Construcitng a Tetrahedron")
from starter.mesh_practice import render360
output_path = "output/tetrahedron_360.gif"
tetrahedron = {
    "vertices": torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.float32,
    ),
    "faces": torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=torch.int64,
    ),
}
images = render360(tetrahedron, image_size=image_size)
print(len(images), images[0].shape, np.max(images[0]))
imageio.mimsave(output_path, images, fps=30)

## 2.1 Construcitng a Cube
print("2.1 Construcitng a Cube")
output_path = "output/cube_360.gif"
cube = {
    "vertices": torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=torch.float32,
    ),
    "faces": torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.int64,
    ),
} 
images = render360(tetrahedron, image_size=image_size)
print(len(images), images[0].shape, np.max(images[0]))
imageio.mimsave(output_path, images, fps=30)

# 3. Retexturing the mesh
print("3. Retexturing the mesh")
from starter.render_mesh import render_cow_retextured_360
output_path = "output/textured_cow_360.gif"
images = render_cow_retextured_360(cow_path=cow_path, image_size=image_size, color1=[0.7, 0.7, 1], color2=[0.7, 1, 0.7])
imageio.mimsave(output_path, images, fps=30)

# 4. Camera Transforms
print("4. Camera Transforms")
from starter.camera_transforms import render_cow

# 0
print("4.1 Camera Transforms: 0")
output_path = "output/transform_cow_0.png"
R_rel = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ"
).numpy()
T_rel = [0, 0, 0]
plt.imsave(output_path, 
            render_cow(cow_path="data/cow_with_axis.obj", 
                        image_size=image_size, 
                        R_relative=R_rel,
                        T_relative=T_rel))

# 1
output_path = "output/transform_cow_1.png"
print("4.1 Camera Transforms: 1")
R_rel = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, np.pi/2]), "XYZ"
).numpy()
T_rel = [0, 0, 0]
plt.imsave(output_path, 
            render_cow(cow_path="data/cow_with_axis.obj", 
                        image_size=image_size, 
                        R_relative=R_rel,
                        T_relative=T_rel))

# 2
print("4.1 Camera Transforms: 2")
output_path = "output/transform_cow_2.png"
R_rel = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ"
).numpy()
T_rel = [0, 0, 2]
plt.imsave(output_path, 
            render_cow(cow_path="data/cow_with_axis.obj", 
                        image_size=image_size, 
                        R_relative=R_rel,
                        T_relative=T_rel))

# 3
print("4.1 Camera Transforms: 3")
output_path = "output/transform_cow_3.png"
R_rel = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ"
).numpy()
T_rel = [0.5, -0.5, 0]
plt.imsave(output_path, 
            render_cow(cow_path="data/cow_with_axis.obj", 
                        image_size=image_size, 
                        R_relative=R_rel,
                        T_relative=T_rel))

# 4
print("4.1 Camera Transforms: 4")
output_path = "output/transform_cow_4.png"
R_rel = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, -np.pi/2, 0]), "XYZ"
).numpy()
T_rel = [3, 0, 3]
plt.imsave(output_path, 
            render_cow(cow_path="data/cow_with_axis.obj", 
                        image_size=image_size, 
                        R_relative=R_rel,
                        T_relative=T_rel))

# 5. Rendering generic 3D Representations
print("5. Rendering generic 3D Representations")
# 5.1 Render a point-cloud from RGBD
print("5.1 Render a point-cloud from RGBD")
from starter.render_3d_repr import render_plant
output_path = "output/plant1_360.gif"
images = render_plant(num=1)
imageio.mimsave(output_path, images, fps=30)
output_path = "output/plant2_360.gif"
images = render_plant(num=2)
imageio.mimsave(output_path, images, fps=30)
output_path = "output/plant3_360.gif"
images = render_plant(num=3)
imageio.mimsave(output_path, images, fps=30)

# 5.2 Parametric Fucntions
print("5.2 Parametric Fucntions")
print("torus")
from starter.render_3d_repr import render_torus
output_path = "output/torus_360.gif"
images = render_torus(image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

print("mobius")
from starter.render_3d_repr import render_klein
output_path = "output/mobius_360.gif"
images = render_klein(image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

# 5.3 Implicit Functions
print("5.3 Implicit Functions")
print("torus mesh")
from starter.render_3d_repr import render_torus_mesh
output_path = "output/torus_mesh_360.gif"
images = render_torus_mesh(image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

print("spring mesh")
from starter.render_3d_repr import render_spring_mesh
output_path = "output/spring_mesh_360.gif"
images = render_spring_mesh(image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

# 6. Do Something fun
print("6. Do Something fun")
from starter.something_fun import render_torus
output_path = "output/fun.gif"
images = render_torus(image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

# 7. Samping points on a mesh
print("7. Sampling points on a mesh")
print("10")
from starter.sampling_mesh import render_cow
output_path = "output/sampled_cow_10.gif"
images = render_cow(cow_path=cow_path, num_samples=10, image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

print("100")
output_path = "output/sampled_cow_100.gif"
images = render_cow(cow_path=cow_path, num_samples=100, image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

print("1000")
output_path = "output/sampled_cow_1000.gif"
images = render_cow(cow_path=cow_path, num_samples=1000, image_size=image_size)
imageio.mimsave(output_path, images, fps=30)

print("10000")
output_path = "output/sampled_cow_10000.gif"
images = render_cow(cow_path=cow_path, num_samples=10000, image_size=image_size)
imageio.mimsave(output_path, images, fps=30)