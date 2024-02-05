"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

import numpy as np
import imageio

def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def render_cow_retextured_360(
    cow_path="data/cow.obj", image_size=256, color1=[0.7, 0.7, 1], color2=[0.7, 1, 0.7], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    
    z_min = vertices[:, :, 2].min()
    z_max = vertices[:, :, 2].max()

    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    alpha = alpha.unsqueeze(-1)

    textures = alpha * torch.tensor(color1) + (1 - alpha) * torch.tensor(color2)
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    num_views = 120
    radius = 3
    num_rotations = 4
    dist, elev, azim = generate_spiral_path(num_views, radius, num_rotations)
    
    renders = []
    for d, e, a in zip(dist, elev, azim):
        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = rend * 255
        renders.append(rend.astype(np.uint8))
    return renders

def generate_spiral_path(num_views, radius, num_rotations):
    """
    Generate a spiral path around an object.
    
    Args:
        num_views (int): Number of views to generate.
        radius (float): Radius of the spiral.
        num_rotations (int): Number of rotations to complete the spiral.
    
    Returns:
        Tuple: Arrays of distance, elevation, and azimuth values.
    """
    t = np.linspace(0, num_rotations * 2 * np.pi, num_views)
    dist = np.linspace(radius, radius, num_views)
    elev = np.concatenate([np.linspace(0, 60, int(num_views/2)), np.linspace(59, 0, int(num_views/2))])
    azim = np.linspace(-180, 180, num_views)
    
    # Spiral motion
    dist_offset = 1
    dist += dist_offset * np.sin(t / 25)
    
    return dist, elev, azim

def render360(
    obj_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(obj_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)


    num_views = 120
    radius = 3
    num_rotations = 4
    dist, elev, azim = generate_spiral_path(num_views, radius, num_rotations)
    
    renders = []
    for d, e, a in zip(dist, elev, azim):
        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = rend * 255
        renders.append(rend.astype(np.uint8))
    return renders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    # image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    # plt.imsave(args.output_path, image)

    # images = render360(obj_path=args.cow_path, image_size=args.image_size)
    images = render_cow_retextured_360(cow_path=args.cow_path, image_size=args.image_size, color1=[0.7, 0.7, 1], color2=[0.7, 1, 0.7])
    print(len(images), images[0].shape, np.max(images[0]))
    imageio.mimsave(args.output_path, images, fps=30)