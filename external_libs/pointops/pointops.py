import fpsample
import numpy as np
from torch_geometric.nn import fps, knn, knn_interpolate
import torch

def furthestsampling_cpu(xyz, offset, new_offset):
    """
    furthest point sampling -> numpy implementation (deterministic)

    Args:
        xyz (np.array): shape [batch_size x num_points, 3]
        offset (np.array): shape [batch_size,] -> [N_o1, N_o1+N_o2, ..., sum_{i=1}^N N_oi] where N_oi denotes the number of points before sampling for the ith cloud
        new_offset (np.array): shape [batch_size,] -> [N_s1, N_s1+N_s2, ..., sum_{i=1}^N N_si] where N_si denotes the number of points after sampling for the ith cloud
    """
    idx = []
    total_samples = 0
    
    for i in range(len(offset)):
        old_offset = 0 if i == 0 else offset[i-1]
        new_offset = offset[i]
        num_points = new_offset[i] if i == 0 else new_offset[i] - new_offset[i-1]
        batch_xyz = xyz[old_offset:new_offset]
        batch_idx = fpsample.fps_sampling(batch_xyz, num_points, start_idx=0)
        idx.append(batch_idx + total_samples)
        total_samples += len(batch_idx)
        
    idx = np.stack(idx)
    return idx
        
def furthestsampling_gpu(xyz, batch_size, num_samples):
    """
    furthest point sampling -> pytorch implementation (deterministic)

    Args:
        xyz (torch.Tensor): Node feature matrix, shape [batch_size x num_points, Dim]
        batch_size (int)
        num_samples (int)
    """
    num_points = xyz.shape[0] // batch_size
    batch = torch.from_numpy(np.repeat(np.arange(batch_size).reshape((-1, 1)), repeats=num_points)).to(torch.int64)
    batch = batch.to(xyz.device)
    ratio = float(num_samples) / num_points
    idx = fps(xyz, batch=batch, ratio=ratio, random_start=False)
    
    return idx

def knnquery_gpu(source_xyz, target_xyz, batch_size, num_samples):
    """
    KNN query search -> pytorch implementation

    Args:
        source_xyz (torch.Tensor): [batch_size x num_source_points, Dim]
        target_xyz (torch.Tensor): [batch_size x num_target_points, Dim]
        batch_size (int)
        num_samples (int): k
    """
    num_source_points = source_xyz.shape[0] // batch_size
    source_batch = torch.from_numpy(np.repeat(np.arange(batch_size).reshape((-1, 1)), repeats=num_source_points)).to(torch.int64)
    source_batch = source_batch.to(source_xyz.device)
    
    num_target_points = target_xyz.shape[0] // batch_size
    target_batch = torch.from_numpy(np.repeat(np.arange(batch_size).reshape((-1, 1)), repeats=num_target_points)).to(torch.int64)
    target_batch = target_batch.to(target_xyz.device)
    
    num_samples = min(num_samples, num_source_points)
    
    idx = knn(source_xyz, target_xyz, num_samples, source_batch, target_batch)
    idx = idx[1].reshape((-1, num_samples))
    return idx


def queryandgroup(source_xyz, target_xyz, source_feat, batch_size, num_samples, use_xyz=True):
    """
    Group features for knn neighbors

    Args:
        source_xyz (torch.Tensor): [batch_size x num_source_points, 3] (n, 3)
        target_xyz (torch.Tensor): [batch_size x num_target_points, 3] (m, 3)
        source_feat (torch.Tensor): [batch_size x num_source_points, Dim_feature] (n, c)
        batch_size (int)
        num_samples (int): k
        idx (torch.Tensor): KNN neighbor indices (m, num_samples)
        use_xyz (bool)
        
    Returns:
        grouped_feat: (m, num_samples, c+3) if use_xyz else (m, num_samples, c)
    """
    if target_xyz is None:
        target_xyz = source_xyz
    
    idx = knnquery_gpu(source_xyz, target_xyz, batch_size, num_samples) # (m, num_samples)
    
    max_num_samples = source_xyz.shape[0] // batch_size
    num_samples = min(max_num_samples, num_samples)

    m, c = target_xyz.shape[0], source_feat.shape[1]
    grouped_xyz = source_xyz[idx.view(-1).long(), :].view(m, num_samples, 3) # (m, num_samples, 3)
    grouped_xyz -= target_xyz.unsqueeze(1) # (m, num_samples, 3)
    grouped_feat = source_feat[idx.view(-1).long(), :].view(m, num_samples, c) # (m, num_samples, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, num_samples, 3+c)
    else:
        return grouped_feat # (m, num_samples, c)
    

def interpolation(source_xyz, target_xyz, source_feat, batch_size, num_samples=3):
    """
    Weighted interpolation of features for knn neighbors
    
    Args:
        source_xyz (torch.Tensor): [batch_size x num_source_points, 3] (n, 3)
        target_xyz (torch.Tensor): [batch_size x num_target_points, 3] (m, 3)
        source_feat (torch.Tensor): [batch_size x num_source_points, Dim_feature] (n, c)
        batch_size (int)
        num_samples (int): k
        
    Returns:
        grouped_feat: (m, c)
    """
    num_source_points = source_xyz.shape[0] // batch_size
    source_batch = torch.from_numpy(np.repeat(np.arange(batch_size).reshape((-1, 1)), repeats=num_source_points)).to(torch.int64)
    source_batch = source_batch.to(source_xyz.device)
    
    num_target_points = target_xyz.shape[0] // batch_size
    target_batch = torch.from_numpy(np.repeat(np.arange(batch_size).reshape((-1, 1)), repeats=num_target_points)).to(torch.int64)
    target_batch = target_batch.to(target_xyz.device)
    
    new_feat = knn_interpolate(source_feat, source_xyz, target_xyz, source_batch, target_batch, num_samples)
    return new_feat
    