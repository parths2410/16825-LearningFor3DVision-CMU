import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

import imageio

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image
from starter.render_mesh import generate_spiral_path

from starter.render_generic import load_rgbd_data

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
    elev = np.concatenate([np.linspace(0, -60, int(num_views/2)), np.linspace(-59, 0, int(num_views/2))])
    azim = np.linspace(-180, 180, num_views)
    
    # Spiral motion
    dist_offset = 0
    dist += dist_offset * np.sin(t / 25)
    elev *= 0
    return dist, elev, azim

def render_pcd(
    point_cloud,
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    
    point_cloud = pytorch3d.structures.Pointclouds(
        points=verts, 
        features=rgb)
    
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
    num_views = 120
    radius = 6
    num_rotations = 4
    dist, elev, azim = generate_spiral_path(num_views, radius, num_rotations)
    
    renders = []
    for d, e, a in zip(dist, elev, azim):
        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = rend * 255
        renders.append(rend.astype(np.uint8))
    return renders

def render_plant():
    data = load_rgbd_data()

    image1 = torch.tensor(data["rgb1"]).to(torch.float32)
    mask1 = torch.tensor(data["mask1"]).to(torch.float32)
    depth1 = torch.tensor(data["depth1"]).to(torch.float32)
    camera1 = data["cameras2"]
    pcd_verts1, pcd_rgba1 = unproject_depth_image(image1, mask1, depth1, camera1)
    
    image2 = torch.tensor(data["rgb2"]).to(torch.float32)
    mask2 = torch.tensor(data["mask2"]).to(torch.float32)
    depth2 = torch.tensor(data["depth2"]).to(torch.float32)
    camera2 = data["cameras2"]
    pcd_verts2, pcd_rgba2 = unproject_depth_image(image2, mask2, depth2, camera2)
    
    pcd_verts = torch.cat([pcd_verts1, pcd_verts2], dim=0)
    pcd_rgba = torch.cat([pcd_rgba1, pcd_rgba2], dim=0)

    point_cloud = {"verts": pcd_verts, "rgb": pcd_rgba}
    return render_pcd(point_cloud)
    # plt.imshow(rend)
    # plt.show()
    
def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    R1 = 1
    R2 = 0.5
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v)

    x = (R1 + R2 * torch.cos(V)) * torch.cos(U)
    y = (R1 + R2 * torch.cos(V)) * torch.sin(U)
    z = R2 * torch.sin(V)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    
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

        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        
        renders.append(rend.astype(np.uint8))

    return renders

def render_klein(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    a = 3
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v)

    x = (a + torch.cos(U / 2) * torch.sin(V) - torch.sin(U / 2) * torch.sin(2 * V)) * torch.cos(U)
    y = (a + torch.cos(U / 2) * torch.sin(V) - torch.sin(U / 2) * torch.sin(2 * V)) * torch.sin(U)
    z = torch.sin(U / 2) * torch.sin(V) + torch.cos(U / 2) * torch.sin(2 * V)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    
    num_views = 120
    radius = 10
    num_rotations = 4
    dist, elev, azim = generate_spiral_path(num_views, radius, num_rotations)
    
    renders = []
    for d, e, a in zip(dist, elev, azim):
        R, T = pytorch3d.renderer.look_at_view_transform(d, e, a)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        
        renders.append(rend.astype(np.uint8))

    return renders

def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    min_value = -1.1
    max_value = 1.1

    R1 = 0.75
    R2 = 0.25

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (R1 - np.sqrt(X ** 2 + Y ** 2)) ** 2 + Z ** 2 - R2 ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
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
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        
        renders.append(rend.astype(np.uint8))

    return renders

def render_spring_mesh(image_size=256, voxel_size=64, device=None):

    if device is None:
        device = get_device()

    min_value = -1.1
    max_value = 1.1

    R1 = 0.75
    R2 = 0.2

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (R1 - np.sqrt(X ** 2 + Y ** 2)) ** 2 + (Z + np.arctan2(X, Y) / np.pi)** 2 - R2 ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
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
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = rend * 255
        
        renders.append(rend.astype(np.uint8))

    return renders

if __name__ == "__main__":
    # images = render_plant()
    # images = render_torus()
    # images = render_klein()
    images = render_torus_mesh()
    # images = render_spring_mesh()
    imageio.mimsave("output/torus_mesh_360.gif", images, fps=30)

    # images = render_sphere()
    # plt.imshow(images)
    # plt.show()