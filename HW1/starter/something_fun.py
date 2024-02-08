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

def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R1 = 0.4
    R2 = 0.1
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v)

    x = (R1 + R2 * torch.cos(U)) * torch.cos(V)
    y = (R1 + R2 * torch.cos(U)) * torch.sin(V)
    z = R2 * torch.sin(U)

    points_torus = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).to(device)

    color = [0.7, 0.7, 1]

    vertices, faces = load_cow_mesh("data/cow.obj")
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
    sampled_faces_idxs = np.random.choice(len(faces_probs), 10000, p=faces_probs)
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

    # rotate the torus_points by 30 degrees in the z-axis
    R = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([np.pi / 3, 0, 0]), "XYZ").to(device)
    points_torus = torch.matmul(points_torus, R.T)
    T = torch.tensor([0, 0.15, 0]).to(device)
    points_torus = points_torus + T

    # give the cow a random color but every point the same color
    color_cow = torch.ones_like(sampled_pts) * torch.tensor([0.7, 0.7, 1]).to(device)

    points = torch.cat([sampled_pts, points_torus], dim=0)
    

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    
    num_views = 120
    radius = 3
    num_rotations = 4
    dist, elev, azim = generate_spiral_path(num_views, radius, num_rotations)
    
    renders = []
    disco_color = torch.rand(3).to(device)
    for d, e, a in zip(dist, elev, azim):
        if len(renders) % 20 == 0:
            disco_color = torch.rand(3).to(device)

        complementory_color = torch.tensor([1, 1, 1]).to(device) - disco_color
        complementory_color = complementory_color * 0.85

        color_cow = torch.ones_like(sampled_pts) * complementory_color
        color_torus = torch.ones_like(points_torus) * disco_color
        colors = torch.cat([color_cow, color_torus], dim=0)

        point_cloud = pytorch3d.structures.Pointclouds(
            points=[points], features=[colors],
        ).to(device)

        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
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
    elev = np.linspace(0, 0, num_views)
    azim = np.linspace(-180, 180, num_views)
    
    # Spiral motion
    dist_offset = 0
    dist += dist_offset * np.sin(t / 25)
    
    return dist, elev, azim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/fun.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = render_torus(num_samples=500, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=30)