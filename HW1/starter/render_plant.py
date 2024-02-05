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
    dist_offset = 1
    dist += dist_offset * np.sin(t / 25)
    
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
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
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

if __name__ == "__main__":
    images = render_plant()
    imageio.mimsave("output/plant3_360.gif", images, fps=30)