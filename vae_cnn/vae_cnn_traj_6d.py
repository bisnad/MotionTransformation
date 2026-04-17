import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os, time
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer

# -------------------------------------------------------------------------------------------------
# Utility: 6D Conversions
# -------------------------------------------------------------------------------------------------
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    
    x = nnF.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = nnF.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    
    return torch.stack((x, y, z), dim=-1)

def quat_to_6d(quats):
    orig_shape = quats.shape
    quats_flat = quats.reshape(-1, 4)
    
    matrices = R.from_quat(quats_flat, scalar_first=True).as_matrix()
    matrices = matrices.reshape(orig_shape[:-1] + (3, 3))
    
    x = matrices[..., :, 0]
    y = matrices[..., :, 1]
    rot_6d = np.concatenate((x, y), axis=-1)
    
    return rot_6d.astype(np.float32)

def ortho6d_to_quat(poses_6d):
    orig_shape = poses_6d.shape
    poses = poses_6d.reshape(-1, 6)
    
    x = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    
    matrices = np.stack((x, y, z), axis=-1)
    quats = R.from_matrix(matrices).as_quat(scalar_first=True)
    
    return quats.reshape(orig_shape[:-1] + (4,)).astype(np.float32)

# -------------------------------------------------------------------------------------------------
# Compute Unit
# -------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# -------------------------------------------------------------------------------------------------
# Mocap Settings
# -------------------------------------------------------------------------------------------------
mocap_file_path = "E:/Data/mocap/stocos/Solos/Canal_14-08-2023/fbx_50hz/"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None

train_root_trajectory = True

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------
save_path = "results/"
save_weights_path = save_path + "weights/"
save_history_path = save_path + "history/"
save_anims_path = save_path + "anims/"

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_anims_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------
vae_input_dim = None 
vae_latent_dim = 16 
vae_conv_channel_counts = [64, 64, 64] 
vae_conv_kernel_sizes = [3, 3, 3, 4] 
vae_conv_strides = [2, 2, 2, 2] 
vae_conv_dilations = [1, 2, 4, 1] 

mocap_window_length = 64 
mocap_window_offset = 1 
test_percentage = 0.1
batch_size = 128
epochs = 200
learning_rate = 1e-3

pos_rec_loss_scale = 0.1
rot_rec_loss_scale = 1.0

target_beta = 0.25
target_beta_cycles = 4
target_beta_ratio = 0.5

save_weights = True
save_weights_interval = 50
load_weights = False
load_weights_file = "motion_vae_Training/results/weights/vae_epoch_0200.pt"

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

# -------------------------------------------------------------------------------------------------
# Load Mocap Data
# -------------------------------------------------------------------------------------------------
bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path + mocap_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path + mocap_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] 
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    if not train_root_trajectory:
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 0] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 2] = 0.0

    mocap_data["motion"]["rot_local"] = quat_to_6d(mocap_data["motion"]["rot_local"])
    all_mocap_data.append(mocap_data)

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = 6 
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

def get_edge_list(children):
    edge_list = []
    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    return edge_list

edge_list = get_edge_list(children)

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file) as f:
        joint_loss_weights = json.load(f)
    joint_loss_weights = joint_loss_weights["joint_loss_weights"]
else:
    joint_loss_weights = [1.0] * joint_count

# -------------------------------------------------------------------------------------------------
# Create Dataset
# -------------------------------------------------------------------------------------------------
pose_sequence_excerpts = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    
    if train_root_trajectory:
        root_positions = mocap_data["motion"]["pos_local"][:, 0, :]
        pose_sequence = np.concatenate((root_positions, pose_sequence), axis=1)
    
    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]

    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - mocap_window_length, mocap_window_offset):
        pose_sequence_excerpt = pose_sequence[seq_excerpt_start:seq_excerpt_start + mocap_window_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)

pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)

if train_root_trajectory:
    root_traj_excerpts = pose_sequence_excerpts[:, :, :3]
    
    root_pos_mean = np.mean(root_traj_excerpts, axis=(0, 1), keepdims=True)
    root_pos_std = np.std(root_traj_excerpts, axis=(0, 1), keepdims=True)
    root_pos_std[root_pos_std == 0] = 1.0 
    
    pose_sequence_excerpts[:, :, :3] = (root_traj_excerpts - root_pos_mean) / root_pos_std
    
    root_pos_mean_tensor = torch.from_numpy(root_pos_mean).to(device)
    root_pos_std_tensor = torch.from_numpy(root_pos_std).to(device)

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts

    def __len__(self):
        return self.sequence_excerpts.shape[0]

    def __getitem__(self, idx):
        return self.sequence_excerpts[idx, ...]

full_dataset = SequenceDataset(pose_sequence_excerpts)
test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------------------------------
# Create Models
# -------------------------------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)

    def forward(self, x):
        x_padded = nnF.pad(x, (self.pad, 0))
        return self.conv(x_padded)

class MotionEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, channel_counts, kernel_sizes, strides, dilations):
        super().__init__()
        layers = []
        current_channels = in_channels

        for i in range(len(kernel_sizes)):
            out_channels = channel_counts[i] if i < len(channel_counts) else latent_channels * 2
            layers.append(CausalConv1d(current_channels, out_channels, kernel_size=kernel_sizes[i], stride=strides[i], dilation=dilations[i]))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
            current_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MotionDecoder(nn.Module):
    def __init__(self, out_channels, latent_channels, channel_counts, kernel_sizes, strides, dilations):
        super().__init__()
        layers = []
        current_channels = latent_channels

        for i in range(len(kernel_sizes)):
            if strides[i] > 1:
                layers.append(nn.Upsample(scale_factor=strides[i], mode="nearest"))
            
            next_channels = channel_counts[i] if i < len(channel_counts) else out_channels
            layers.append(CausalConv1d(current_channels, next_channels, kernel_size=kernel_sizes[i], stride=1, dilation=dilations[i]))
            
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
            current_channels = next_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CausalMotionVAE(nn.Module):
    def __init__(self, input_dim=63, latent_dim=16, conv_channel_counts=[128, 128, 128], conv_kernel_sizes=[3, 3, 3, 4], conv_strides=[1, 1, 1, 2], conv_dilations=[1, 2, 4, 1]):
        super().__init__()
        self.encoder = MotionEncoder(in_channels=input_dim, latent_channels=latent_dim, channel_counts=conv_channel_counts, kernel_sizes=conv_kernel_sizes, strides=conv_strides, dilations=conv_dilations)
        
        rev_channels = list(reversed(conv_channel_counts))
        rev_kernels = list(reversed(conv_kernel_sizes))
        rev_strides = list(reversed(conv_strides))
        rev_dilations = list(reversed(conv_dilations))
        
        self.decoder = MotionDecoder(out_channels=input_dim, latent_channels=latent_dim, channel_counts=rev_channels, kernel_sizes=rev_kernels, strides=rev_strides, dilations=rev_dilations)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

vae_input_dim = pose_dim + 3 if train_root_trajectory else pose_dim

vae = CausalMotionVAE(input_dim=vae_input_dim, latent_dim=vae_latent_dim, conv_channel_counts=vae_conv_channel_counts, conv_kernel_sizes=vae_conv_kernel_sizes, conv_strides=vae_conv_strides, conv_dilations=vae_conv_dilations).to(device)

if load_weights and load_weights_file:
    vae_state_dict = torch.load(load_weights_file, map_location=device)
    vae.load_state_dict(vae_state_dict["model_state_dict"])
    
mocap_batch = next(iter(train_dataloader)).to(device)
mocap_batch = mocap_batch.permute(0, 2, 1)

recon, mu, logvar = vae(mocap_batch)
vae_latent_count = mu.shape[-1]

print(f"mocap_batch s : {mocap_batch.shape}")
print(f"recon s : {recon.shape}")
print(f"mu s : {mu.shape}")
print(f"logvar s : {logvar.shape}")

# -------------------------------------------------------------------------------------------------
# Training Losses
# -------------------------------------------------------------------------------------------------
def create_beta_schedule(epochs, target_beta=4.0, cycles=4, ratio=0.5):
    schedule = []
    period = epochs / cycles
    for epoch in range(epochs):
        cycle_step = epoch % period
        phase = cycle_step / period
        if phase < ratio:
            beta = target_beta * (phase / ratio)
        else:
            beta = target_beta
        schedule.append(beta)
    return schedule

beta_schedule = create_beta_schedule(epochs=epochs, target_beta=target_beta, cycles=target_beta_cycles, ratio=target_beta_ratio)
optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)

cycle_length = epochs / target_beta_cycles
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(cycle_length), T_mult=1, eta_min=1e-6)

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32).reshape(1, 1, -1).to(device)

"""
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
"""

def kl_loss(mu, logvar, free_bits=0.5):
    # Per-dimension KL: shape [batch, latent_dim, latent_steps]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Clamp: ensure each dim contributes at least free_bits nats
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()

def forward_kinematics(rotation_matrices, root_positions):
    t_offsets = torch.tensor(offsets).to(device)
    expanded_offsets = t_offsets.expand(rotation_matrices.shape[0], rotation_matrices.shape[1], offsets.shape[0], offsets.shape[1]).unsqueeze(-1)

    positions_world = []
    rotations_world = []

    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotation_matrices[:, :, 0])
        else:
            parent_rot = rotations_world[parents[jI]]
            local_offset = expanded_offsets[:, :, jI]
            
            rotated_offset = torch.matmul(parent_rot, local_offset).squeeze(-1)
            positions_world.append(rotated_offset + positions_world[parents[jI]])

            if len(children[jI]) > 0:
                new_world_rot = torch.matmul(parent_rot, rotation_matrices[:, :, jI])
                rotations_world.append(new_world_rot)
            else:
                rotations_world.append(parent_rot)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def pos_rec_loss(y, yhat):
    if train_root_trajectory:
        y_root_traj = (y[:, :, :3] * root_pos_std_tensor) + root_pos_mean_tensor
        yhat_root_traj = (yhat[:, :, :3] * root_pos_std_tensor) + root_pos_mean_tensor
        y_rot_6d = y[:, :, 3:].reshape(y.shape[0], y.shape[1], joint_count, 6)
        yhat_rot_6d = yhat[:, :, 3:].reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)
    else:
        y_root_traj = torch.zeros((y.shape[0], y.shape[1], 3)).to(device)
        yhat_root_traj = torch.zeros((y.shape[0], y.shape[1], 3)).to(device)
        y_rot_6d = y.reshape(y.shape[0], y.shape[1], joint_count, 6)
        yhat_rot_6d = yhat.reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)

    y_mat = compute_rotation_matrix_from_ortho6d(y_rot_6d)
    yhat_mat = compute_rotation_matrix_from_ortho6d(yhat_rot_6d)

    y_pos = forward_kinematics(y_mat, y_root_traj)
    yhat_pos = forward_kinematics(yhat_mat, yhat_root_traj)
    
    pos_diff = torch.norm(y_pos - yhat_pos, dim=3)
    return torch.mean(pos_diff * joint_loss_weights)

def rot_rec_loss(y, yhat):
    if train_root_trajectory:
        y = y[:, :, 3:]
        yhat = yhat[:, :, 3:]
        
    y_rot_6d = y.reshape(y.shape[0], y.shape[1], joint_count, 6)
    yhat_rot_6d = yhat.reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)
    
    y_mat = compute_rotation_matrix_from_ortho6d(y_rot_6d)
    yhat_mat = compute_rotation_matrix_from_ortho6d(yhat_rot_6d)
    
    trace = torch.diagonal(torch.matmul(y_mat.transpose(-1, -2), yhat_mat), dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -0.9999, 0.9999))
    
    return torch.mean(angle * joint_loss_weights)

def ae_loss(y, yhat, mu, std, beta):
    y_permuted = y.permute(0, 2, 1)
    yhat_permuted = yhat.permute(0, 2, 1)
    
    pos_loss = pos_rec_loss(y_permuted, yhat_permuted)
    rot_loss = rot_rec_loss(y_permuted, yhat_permuted)
    kld_loss = kl_loss(mu, std)

    total_loss = (pos_loss * pos_rec_loss_scale) + (rot_loss * rot_rec_loss_scale) + (kld_loss * beta)
    
    if train_root_trajectory:
        traj_mse = torch.mean((y_permuted[:, :, :3] - yhat_permuted[:, :, :3]) ** 2)
        total_loss += (traj_mse * pos_rec_loss_scale)
        
    return total_loss, pos_loss, rot_loss, kld_loss

def vae_train_step(y, beta):
    y_hat, mu, logvar = vae(y)
    loss, pos_rec_loss_val, rot_rec_loss_val, klloss = ae_loss(y, y_hat, mu, logvar, beta)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vae.parameters(), 0.01)
    optimizer.step()
    return loss, pos_rec_loss_val, rot_rec_loss_val, klloss

@torch.no_grad()
def vae_test_step(y, beta):
    y_hat, mu, logvar = vae(y)
    loss, pos_rec_loss_val, rot_rec_loss_val, klloss = ae_loss(y, y_hat, mu, logvar, beta)
    return loss, pos_rec_loss_val, rot_rec_loss_val, klloss

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {"lr": [], "beta": [], "train": [], "test": [], "pos": [], "rot": [], "kl": []}
    
    for epoch in range(epochs):
        start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        current_beta = beta_schedule[epoch] if epoch < len(beta_schedule) else beta_schedule[-1]
        
        loss_history["lr"].append(current_lr)
        loss_history["beta"].append(current_beta)
        
        train_loss_per_epoch = []
        test_loss_per_epoch = []
        pos_rec_loss_per_epoch = []
        rot_rec_loss_per_epoch = []
        kl_loss_per_epoch = []
        
        for train_batch in train_dataloader:
            train_batch = train_batch.to(device).permute(0, 2, 1)
            loss, pos_rec_loss_val, rot_rec_loss_val, kl_loss_val = vae_train_step(train_batch, current_beta)
            
            train_loss_per_epoch.append(loss.detach().cpu().numpy())
            pos_rec_loss_per_epoch.append(pos_rec_loss_val.detach().cpu().numpy())
            rot_rec_loss_per_epoch.append(rot_rec_loss_val.detach().cpu().numpy())
            kl_loss_per_epoch.append(kl_loss_val.detach().cpu().numpy())

        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        pos_rec_loss_per_epoch = np.mean(np.array(pos_rec_loss_per_epoch))
        rot_rec_loss_per_epoch = np.mean(np.array(rot_rec_loss_per_epoch))
        kl_loss_per_epoch = np.mean(np.array(kl_loss_per_epoch))
        
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device).permute(0, 2, 1)
            loss, _, _, _ = vae_test_step(test_batch, current_beta)
            test_loss_per_epoch.append(loss.detach().cpu().numpy())
            
        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))

        if epoch % save_weights_interval == 0 and save_weights:
            torch.save(vae.state_dict(), f"{save_weights_path}vae_weight_epoch_{epoch}.pt")

        loss_history["train"].append(train_loss_per_epoch)
        loss_history["test"].append(test_loss_per_epoch)
        loss_history["pos"].append(pos_rec_loss_per_epoch)
        loss_history["rot"].append(rot_rec_loss_per_epoch)
        loss_history["kl"].append(kl_loss_per_epoch)

        print(f"epoch: {epoch:03d} | train: {train_loss_per_epoch:01.4f} | test: {test_loss_per_epoch:01.4f} | pos: {pos_rec_loss_per_epoch:01.4f} | rot: {rot_rec_loss_per_epoch:01.4f} | kl: {kl_loss_per_epoch:01.4f}")

        if epoch > 1:
            scheduler.step()
            
    return loss_history

# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------
def save_loss_as_csv(loss_history, save_path):
    epochs = len(loss_history["train"])
    csv_data = np.zeros((epochs, 7))
    csv_data[:, 0] = np.arange(1, epochs + 1)
    csv_data[:, 1] = loss_history["train"]
    csv_data[:, 2] = loss_history["test"]
    csv_data[:, 3] = loss_history["pos"]
    csv_data[:, 4] = loss_history["rot"]
    csv_data[:, 5] = loss_history["kl"]
    csv_data[:, 6] = loss_history["lr"]
    np.savetxt(save_path, csv_data, delimiter=",", header="epoch,train,test,pos,rot,kl,lr", comments="")

def plot_training_history(loss_history, save_path):
    epochs = len(loss_history["train"])
    x = np.arange(1, epochs + 1)
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('VAE Training History', fontsize=18, y=0.98)

    axs[0, 0].plot(x, loss_history["train"], label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(x, loss_history["test"], label='Test Loss', color='#ff7f0e', linewidth=2)
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(x, loss_history["pos"], label='Pos Rec', color='#2ca02c')
    axs[0, 1].plot(x, loss_history["rot"], label='Rot Rec', color='#d62728')
    axs[0, 1].set_title('Reconstruction Losses')
    axs[0, 1].legend()

    axs[1, 0].plot(x, loss_history["kl"], label='KL Loss', color='#8c564b', linewidth=2)
    axs[1, 0].set_title('KL Divergence')
    axs[1, 0].legend()

    axs[1, 1].plot(x, loss_history["beta"], label='Beta Value', color='#e377c2', linewidth=2)
    axs[1, 1].set_title('Cyclical Beta Schedule')
    axs[1, 1].legend()

    axs[2, 0].plot(x, loss_history["lr"], label='Learning Rate', color='#17becf', linewidth=2)
    axs[2, 0].set_title('Learning Rate')
    axs[2, 0].legend()

    axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------------------------------
# Inference and Rendering
# -------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_sequences(orig_sequence, frame_indices):
    vae.eval()
    latent_vectors = []
    seq_excerpt_count = len(frame_indices)
    
    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + mocap_window_length
        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame]
        
        if train_root_trajectory:
            excerpt[:, :3] = (excerpt[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()
            
        excerpt = np.expand_dims(excerpt, axis=0)
        excerpt = torch.from_numpy(excerpt).reshape(1, mocap_window_length, vae_input_dim).permute(0, 2, 1).to(device)
        
        mu, logvar = vae.encode(excerpt)
        latent_vector = vae.reparameterize(mu, logvar)
        latent_vectors.append(torch.squeeze(latent_vector).detach().cpu().numpy())
    
    vae.train()
    return latent_vectors

@torch.no_grad()
def decode_sequence_encodings(sequence_encodings, seq_overlap, base_pose):
    vae.eval()
    seq_env = np.hanning(mocap_window_length) + 0.01
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + mocap_window_length

    gen_sequence_accum = np.zeros(shape=(gen_seq_length, joint_count, 3, 3), dtype=np.float32)
    weight_accum = np.zeros(shape=(gen_seq_length, 1, 1), dtype=np.float32)
    
    if train_root_trajectory:
        root_traj_accum = np.zeros(shape=(gen_seq_length, 3), dtype=np.float32)
        root_weight_accum = np.zeros(shape=(gen_seq_length, 1), dtype=np.float32)

    for excerpt_index in range(seq_excerpt_count):
        latent_vector = torch.from_numpy(np.expand_dims(sequence_encodings[excerpt_index], axis=0)).to(device)

        excerpt_dec = vae.decode(latent_vector).permute(0, 2, 1).squeeze().detach().cpu().numpy()
        
        if train_root_trajectory:
            root_traj_dec = excerpt_dec[:, :3]
            excerpt_dec = excerpt_dec[:, 3:]

        excerpt_dec = np.reshape(excerpt_dec, (-1, joint_count, 6))
        
        x_raw = excerpt_dec[..., 0:3]
        y_raw = excerpt_dec[..., 3:6]
        
        x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        y = np.cross(z, x)
        frame_matrices = np.stack((x, y, z), axis=-1)

        gen_frame = excerpt_index * seq_overlap

        for si in range(mocap_window_length):
            mix_weight = seq_env[si]
            gen_sequence_accum[gen_frame + si] += frame_matrices[si] * mix_weight
            weight_accum[gen_frame + si] += mix_weight
            
            if train_root_trajectory:
                root_traj_accum[gen_frame + si] += root_traj_dec[si] * mix_weight
                root_weight_accum[gen_frame + si] += mix_weight

    weight_accum[weight_accum == 0.0] = 1.0 
    avg_matrices = gen_sequence_accum / np.expand_dims(weight_accum, axis=-1)

    x_avg = avg_matrices[..., :, 0]
    y_avg = avg_matrices[..., :, 1]
    rot_6d = np.concatenate((x_avg, y_avg), axis=-1).reshape(gen_seq_length, -1)
    
    if train_root_trajectory:
        root_weight_accum[root_weight_accum == 0.0] = 1.0
        avg_root_traj = root_traj_accum / root_weight_accum
        avg_root_traj = (avg_root_traj * root_pos_std.flatten()) + root_pos_mean.flatten()
        gen_sequence = np.concatenate((avg_root_traj, rot_6d), axis=1)
    else:
        gen_sequence = rot_6d
    
    vae.train()
    return gen_sequence

def create_2d_latent_space_representation(sequence_excerpts):
    encodings = []
    excerpt_count = sequence_excerpts.shape[0]
    
    for e_I in range(0, excerpt_count, batch_size):
        excerpt_batch = sequence_excerpts[e_I:e_I+batch_size]
        excerpt_batch = torch.from_numpy(excerpt_batch).to(device)
        excerpt_batch = excerpt_batch.permute(0, 2, 1)
        
        with torch.no_grad():
            mu, logvar = vae.encode(excerpt_batch)
            encoding_batch = vae.reparameterize(mu, logvar)
            
        encoding_batch = encoding_batch.detach().cpu()
        encoding_batch = encoding_batch.reshape(encoding_batch.size(0), -1)
        encodings.append(encoding_batch)
        
    encodings = torch.cat(encodings, dim=0).numpy()
    
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)
    Z_tsne = tsne.fit_transform(encodings)
    
    return Z_tsne

def create_2d_latent_space_image(Z_tsne, highlight_excerpt_ranges, filename):
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors = ["green", "red", "blue", "magenta", "orange"]
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c='grey',linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c='grey', alpha=0.5)

    for h_I, h_R in enumerate(highlight_excerpt_ranges):
        #ax.plot(Z_tsne_x[h_R[0]:h_R[1]], Z_tsne_y[h_R[0]:h_R[1]], '-', c=plot_colors[h_I],linewidth=0.6)
        ax.scatter(Z_tsne_x[h_R[0]:h_R[1]], Z_tsne_y[h_R[0]:h_R[1]], s=0.8, c=plot_colors[h_I], alpha=0.5)

    ax.set_xlabel('c1')
    ax.set_ylabel('c2')

    fig.savefig(filename, dpi=300)
    plt.close()

poseRenderer = PoseRenderer(edge_list)

poseRenderer = PoseRenderer(edge_list)

def export_sequence_anim(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence
        
    rot_sequence = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    
    rot_sequence_tensor = torch.tensor(np.expand_dims(rot_sequence, axis=0)).to(device)
    rot_matrices = compute_rotation_matrix_from_ortho6d(rot_sequence_tensor)
    
    root_trajectory = torch.tensor(np.expand_dims(root_trajectory, axis=0)).to(device)
    
    skel_sequence = forward_kinematics(rot_matrices, root_trajectory)
    skel_sequence = skel_sequence.detach().cpu().numpy().squeeze()
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_data["frame_rate"],
        "rot_sequence": mocap_data["rot_sequence"],
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }
    
    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = ortho6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_bvh = mocap_tools.mocap_to_bvh([pred_dataset])
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_data["frame_rate"],
        "rot_sequence": mocap_data["rot_sequence"],
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }
    
    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = ortho6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    fbx_tools.write(pred_fbx, file_name)

# -------------------------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------------------------


# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
save_loss_as_csv(loss_history, f"{save_history_path}history_{epochs}.csv")

# plot history
plot_training_history(loss_history, f"{save_history_path}history_{epochs}.png")

# save model weights
torch.save(vae.state_dict(), f"{save_weights_path}vae_weight_epoch_{epochs}.pt")

# create latent space plot
Z_tsne = create_2d_latent_space_representation(pose_sequence_excerpts)
create_2d_latent_space_image(Z_tsne, [], f"{save_path}latent_space_plot_epoch_{epochs}.png")

# create original sequence
orig_rot = all_mocap_data[0]["motion"]["rot_local"].astype(np.float32)
orig_rot = np.reshape(orig_rot, (-1, pose_dim))

if train_root_trajectory:
    orig_pos = all_mocap_data[0]["motion"]["pos_local"][:, 0, :].astype(np.float32)
    orig_sequence = np.concatenate((orig_pos, orig_rot), axis=1)
else:
    orig_sequence = orig_rot

seq_start = 1000
seq_length = 1000

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(orig_sequence[seq_start:seq_start+seq_length], f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.fbx")

# reconstruct original sequence
seq_start = 1000
seq_length = 1000
seq_overlap = mocap_window_length // 4
#seq_overlap = 1

base_pose = orig_sequence[0].copy() 
if train_root_trajectory:
    base_pose[:3] = (base_pose[:3] - root_pos_mean.flatten()) / root_pos_std.flatten()


seq_indices = [frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]
seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)

export_sequence_anim(gen_sequence, f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")

# interpolate two original sequences
seq_1_start = 1000
seq_2_start = 2000
seq_length = 1000

seq_1_indices = [seq_index for seq_index in range(seq_1_start, seq_1_start + seq_length, seq_overlap)]
seq_2_indices = [seq_index for seq_index in range(seq_2_start, seq_2_start + seq_length, seq_overlap)]

seq_1_encodings = encode_sequences(orig_sequence, seq_1_indices)
seq_2_encodings = encode_sequences(orig_sequence, seq_2_indices)

mix_encodings = []
for index in range(len(seq_1_encodings)):
    mix_factor = index / (len(seq_1_indices) - 1)
    mix_encoding = (seq_1_encodings[index] * (1.0 - mix_factor)) + (seq_2_encodings[index] * mix_factor)
    mix_encodings.append(mix_encoding)

gen_sequence = decode_sequence_encodings(mix_encodings, seq_overlap, base_pose)

export_sequence_anim(gen_sequence, f"{save_anims_path}seq_mix_epoch_{epochs}_seq_1_start_{seq_1_start}_seq_2_start_{seq_2_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_mix_epoch_{epochs}_seq_1_start_{seq_1_start}_seq_2_start_{seq_2_start}_length_{seq_length}.fbx")

# sequence offset following sine wave
seq_start = 1000
seq_length = 1000

seq_indices = [seq_index for seq_index in range(seq_start, seq_start + seq_length, seq_overlap)]
seq_encodings = encode_sequences(orig_sequence, seq_indices)

offset_seq_encodings = []
for index in range(len(seq_encodings)):
    sin_value = np.sin((index / (len(seq_encodings) - 1)) * np.pi * 4.0)
    offset = np.ones(shape=(vae_latent_dim, vae_latent_count), dtype=np.float32) * sin_value * 4.0
    offset_seq_encoding = seq_encodings[index] + offset
    offset_seq_encodings.append(offset_seq_encoding)

gen_sequence = decode_sequence_encodings(offset_seq_encodings, seq_overlap, base_pose)

export_sequence_anim(gen_sequence, f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")

# random walk in latent space
seq_start = 1000
seq_length = 1000

seq_indices = [seq_start]
seq_encodings = encode_sequences(orig_sequence, seq_indices)

for index in range(0, seq_length // seq_overlap):
    random_step = (np.random.random((vae_latent_dim, vae_latent_count)).astype(np.float32) - 0.5) * 2.0
    seq_encodings.append(seq_encodings[index] + random_step)

gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)

export_sequence_anim(gen_sequence, f"{save_anims_path}seq_rand_walk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_rand_walk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")