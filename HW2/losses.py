import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	
	voxel_src = torch.sigmoid(voxel_src)	
	loss = torch.nn.BCELoss()
	loss = loss(voxel_src,voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	
	dists_src_to_tgt = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1).dists
	dists_tgt_to_src = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1).dists
	
	# loss_chamfer = torch.mean(dists_src_to_tgt) + torch.mean(dists_tgt_to_src)
	# loss_chamfer = loss_chamfer/2

	loss_chamfer = torch.sum(dists_src_to_tgt) / point_cloud_src.shape[0] + torch.sum(dists_tgt_to_src) / point_cloud_tgt.shape[0]
	loss_chamfer = loss_chamfer/2

	loss_chamfer_2 = torch.sum((dists_src_to_tgt + dists_tgt_to_src)) / point_cloud_src.shape[0]
	# print("Chamfer Loss 1: ", loss_chamfer)
	
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	loss_laplacian = mesh_laplacian_smoothing(mesh_src)

	return loss_laplacian