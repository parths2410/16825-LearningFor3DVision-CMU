import numpy as np
import torch
import pytorch3d

from utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer, vox2mesh

def render(data, type):
    renders = []
    if type == "mesh":
        renders = render360(data, type="mesh")
    elif type == "point":
        renders = render360(data, type="point")
    elif type == "vox":
        renders = render360(data, type="vox")
    return


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
    dist_offset = 1
    dist += dist_offset * np.sin(t / 25)
    
    return dist, elev, azim


def render360(
    obj, image_size=256, color=[0.7, 0.7, 1], device=None, type = "mesh"
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    if type == "point":
        renderer = get_points_renderer(image_size=image_size, device=device)
    
        sampled_pts = obj.to(device).squeeze(0)
        
        # red color for all points
        # color = [1, 0, 0]
        color = torch.ones_like(sampled_pts) * torch.tensor(color).to(device)

        obj = pytorch3d.structures.Pointclouds(
            points=[sampled_pts], features=[color],
        ).to(device)

    elif type == "mesh":
        renderer = get_mesh_renderer(image_size=image_size)

        vertices, faces = obj.verts_packed(), obj.faces_packed()
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        
        textures = torch.ones_like(vertices).to('cuda')  # (1, N_v, 3)
        textures = textures * torch.tensor(color).to('cuda')  # (1, N_v, 3)
        obj = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        obj = obj.to(device)

    elif type == "vox":
        renderer = get_mesh_renderer(image_size=image_size)

        obj = vox2mesh(obj, color=color)
        obj = obj.to(device)
    
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

        rend = renderer(obj.detach(), cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = rend * 255
        renders.append(rend.astype(np.uint8))
    return renders