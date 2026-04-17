import torch
import torch.nn.functional as nnF
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]

    x = nnF.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = nnF.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    return torch.stack((x, y, z), dim=-1)


def quat_to_6d(quats):
    quats = np.asarray(quats, dtype=np.float32)
    orig_shape = quats.shape
    quats_flat = quats.reshape(-1, 4)

    matrices = R.from_quat(quats_flat, scalar_first=True).as_matrix()
    matrices = matrices.reshape(orig_shape[:-1] + (3, 3))

    x = matrices[..., :, 0]
    y = matrices[..., :, 1]
    rot_6d = np.concatenate((x, y), axis=-1)

    return rot_6d.astype(np.float32)


def ortho6d_to_quat(poses_6d):
    poses_6d = np.asarray(poses_6d, dtype=np.float32)
    orig_shape = poses_6d.shape
    poses = poses_6d.reshape(-1, 6)

    x = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
    z = np.cross(x, y_raw)
    z = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8, None)
    y = np.cross(z, x)

    matrices = np.stack((x, y, z), axis=-1)
    quats = R.from_matrix(matrices).as_quat(scalar_first=True)

    return quats.reshape(orig_shape[:-1] + (4,)).astype(np.float32)


def orthogonalize_6d(rot_6d):
    matrices = compute_rotation_matrix_from_ortho6d(rot_6d)
    
    # Extract the X and Y column vectors [..., 3]
    x = matrices[..., :, 0]
    y = matrices[..., :, 1]
    
    # Concatenate to form proper [..., 6] format
    if isinstance(matrices, torch.Tensor):
        return torch.cat((x, y), dim=-1)
    else:
        import numpy as np
        return np.concatenate((x, y), axis=-1)