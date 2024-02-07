"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer

import numpy as np
import imageio

def render_cow(
    cow_path="data/cow.obj", num_samples = 10000, image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_points_renderer(image_size=image_size, device=device)
    
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

    # print(faces.shape)
    areas = mesh.faces_areas_packed()
    faces_probs = areas / areas.sum()
    faces_probs = faces_probs.cpu().numpy()
    
    sampled_pts = []
    # sample faces
    sampled_faces_idxs = np.random.choice(len(faces_probs), num_samples, p=faces_probs)
    for idx in sampled_faces_idxs:
        face = faces[0][idx]
        verts = vertices[0][face]
        verts = verts.cpu().numpy()

        # computing barycentric coordinates
        a, a2 = np.random.rand(2)
        a1 = 1 - np.sqrt(a)

        pt = a1 * verts[0] + (1 - a1) * a2 * verts[1] + (1 - a1) * (1 - a2) * verts[2]
        sampled_pts.append(pt)

    sampled_pts = torch.tensor(np.array(sampled_pts)).to(device)

    color = (sampled_pts - sampled_pts.min()) / (sampled_pts.max() - sampled_pts.min())


    point_cloud = pytorch3d.structures.Pointclouds(
        points=[sampled_pts], features=[color],
    ).to(device)
    
    # Prepare the camera:
    R, T = pytorch3d.renderer.look_at_view_transform(3, 0, 170)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device
    )

    num_views = 120
    dist = np.linspace(3, 3, num_views)
    elev = np.linspace(0, 0, num_views)
    azim = np.linspace(-180, 180, num_views)

    renders = []
    for d, e, a in zip(dist, elev, azim):
        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        
        rend = renderer(point_cloud, cameras=cameras)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/sampled_cow_10.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = render_cow(cow_path=args.cow_path, num_samples=10, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=30)