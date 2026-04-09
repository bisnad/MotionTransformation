"""
A variational autoencoder for motion capture data of a solo dancer
this is for motion capture data that stores joint rotations and recorded in BVH or FBX format
"""

import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import json

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

"""
Compute Unit
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings

important: the skeleton needs to be identical in all mocap recordings
"""

mocap_file_path = "E:/Data/mocap/stocos/Solos/Canal_14-08-2023/fbx_50hz"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None

"""
Save Paths Settings
"""

save_path = "results/"
save_weights_path = save_path + "weights/"
save_history_path = save_path + "history/"
save_anims_path = save_path + "anims/"

"""
Model Settings
"""

vae_input_dim = None
vae_latent_dim = 16
vae_latent_count = None 
vae_conv_channel_counts = [256, 256, 256]
vae_conv_kernel_sizes = [3, 3, 3, 4]
vae_conv_strides = [2, 2, 2, 2]
vae_conv_dilations = [1, 2, 4, 1]

"""
Training Settings
"""

mocap_window_length = 64 # mocap sequence excerpt length in frames that the VAE operates on.
mocap_window_offset = 1 # how many frames to shift the window for the next excerpt
test_percentage = 0.1
batch_size = 128
epochs = 200
learning_rate = 1e-3
learning_rate_cycle_decay = 0.5 
learning_rate_step_size = 40
learning_rate_gamma = 0.5

quat_norm_loss_scale = 0.1
pos_rec_loss_scale = 0.1
rot_rec_loss_scale = 1.0

target_beta = 0.25
target_beta_cycles = 4
target_beta_ratio = 0.5

save_weights = True
save_weights_interval = 50
load_weights = False
load_weights_file = "motion_vae/Training/results/weights/vae_epoch_0200.pt"

"""
Visualization Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Create Save Directories
"""

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_anims_path, exist_ok=True)

"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only
        
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

# set joint loss weigths 

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file) as f:
        joint_loss_weights = json.load(f)
        joint_loss_weights = joint_loss_weights["joint_loss_weights"]
else:
    joint_loss_weights = [1.0]
    joint_loss_weights *= joint_count
        

"""
Create Dataset
"""

# gather pose sequence excerpts

pose_sequence_excerpts = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    
    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - mocap_window_length, mocap_window_offset):
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + sequence_length) )
        pose_sequence_excerpt =  pose_sequence[seq_excerpt_start:seq_excerpt_start + mocap_window_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)
    
pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)

"""
# calc mocap mean and std

mocap_mean = np.mean(pose_sequence_excerpts, axis=(0,1))
mocap_std = np.std(pose_sequence_excerpts, axis=(0,1))

mocap_mean = torch.from_numpy(mocap_mean).reshape((1, 1, -1)).to(device)
mocap_std = torch.from_numpy(mocap_std).reshape((1, 1, -1)).to(device)

print("mocap_mean s ", mocap_mean.shape)
print("mocap_std s ", mocap_std.shape)
"""

# create dataset

sequence_excerpts_count = pose_sequence_excerpts.shape[0]

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts
    
    def __len__(self):
        return self.sequence_excerpts.shape[0]
    
    def __getitem__(self, idx):
        return self.sequence_excerpts[idx, ...]
        

full_dataset = SequenceDataset(pose_sequence_excerpts)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_batch = next(iter(train_dataloader))

print("test_batch s ", test_batch.shape)

"""
Create Models
"""

class CausalConv1d(nn.Module):
    """
    A 1D Convolution that strictly pads the 'past' (left side) 
    so the network cannot see future frames.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # Calculate padding needed for the left side
        self.pad = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=0, dilation=dilation)

    def forward(self, x):
        # Pad only the left side: (pad_left, pad_right)
        x_padded = nnF.pad(x, (self.pad, 0))
        return self.conv(x_padded)


class MotionEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, channel_counts, kernel_sizes, strides, dilations):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        for i in range(len(kernel_sizes)):
            # The final layer outputs latent_channels * 2 (half for mu, half for logvar)
            out_channels = channel_counts[i] if i < len(channel_counts) else latent_channels * 2
            
            layers.append(CausalConv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilations[i]
            ))
            
            # Add GELU activation for all but the final projection layer
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
            # Replace strided convolutions with Upsample + stride-1 conv to avoid checkerboarding
            if strides[i] > 1:
                layers.append(nn.Upsample(scale_factor=strides[i], mode='nearest'))
                
            # The final layer outputs the original raw mocap joints
            next_channels = channel_counts[i] if i < len(channel_counts) else out_channels
            
            layers.append(CausalConv1d(
                in_channels=current_channels,
                out_channels=next_channels,
                kernel_size=kernel_sizes[i],
                stride=1, 
                dilation=dilations[i]
            ))
            
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
                
            current_channels = next_channels
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CausalMotionVAE(nn.Module):
    def __init__(self, 
                 input_dim=63, 
                 latent_dim=16, 
                 conv_channel_counts=[128, 128, 128],
                 conv_kernel_sizes=[3, 3, 3, 4],
                 conv_strides=[1, 1, 1, 2],
                 conv_dilations=[1, 2, 4, 1]):
        super().__init__()
        
        self.encoder = MotionEncoder(
            in_channels=input_dim,
            latent_channels=latent_dim,
            channel_counts=conv_channel_counts,
            kernel_sizes=conv_kernel_sizes,
            strides=conv_strides,
            dilations=conv_dilations
        )

        # Reverse the lists to ensure architectural symmetry
        rev_channels = list(reversed(conv_channel_counts))
        rev_kernels = list(reversed(conv_kernel_sizes))
        rev_strides = list(reversed(conv_strides))
        rev_dilations = list(reversed(conv_dilations))
        
        self.decoder = MotionDecoder(
            out_channels=input_dim,
            latent_channels=latent_dim,
            channel_counts=rev_channels,
            kernel_sizes=rev_kernels,
            strides=rev_strides,
            dilations=rev_dilations
        )
        

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
        recon = self.decoder(z)
        return recon
    
    def forward(self, x):

        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        return recon, mu, logvar


vae_input_dim = pose_dim

vae = CausalMotionVAE(
                 input_dim=vae_input_dim, 
                 latent_dim=vae_latent_dim, 
                 conv_channel_counts= vae_conv_channel_counts,
                 conv_kernel_sizes=vae_conv_kernel_sizes,
                 conv_strides=vae_conv_strides,
                 conv_dilations=vae_conv_dilations).to(device)

print(vae)

if load_weights and load_weights_file:
    vae_state_dict = torch.load(load_weights_file, map_location=device) 
    vae.load_state_dict(vae_state_dict["model_state_dict"])

# test vae

mocap_batch = next(iter(train_dataloader)).to(device)
mocap_batch = mocap_batch.permute(0, 2, 1) # shape (B, pose_dim, 64)
recon, mu, logvar = vae(mocap_batch)
vae_latent_count = mu.shape[-1]

print("Test VAE")
print("mocap_batch s: ", mocap_batch.shape)
print("recon s: ", recon.shape)
print("mu s: ", mu.shape)
print("logvar s: ", logvar.shape)

    
"""
Training
"""

# Cyclical Beta Annealing Schedule
def create_beta_schedule(epochs, target_beta=4.0, cycles=4, ratio=0.5):
    """Generates a cyclical beta annealing schedule."""
    schedule = []
    period = epochs / cycles
    
    for epoch in range(epochs):
        cycle_step = epoch % period
        phase = cycle_step / period
        
        if phase <= ratio:
            beta = target_beta * (phase / ratio)
        else:
            beta = target_beta
            
        schedule.append(beta)
    return schedule

beta_schedule = create_beta_schedule(epochs=epochs, target_beta=target_beta, cycles=target_beta_cycles, ratio=target_beta_ratio)

optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
cycle_length = epochs // target_beta_cycles
scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cycle_length,   # The LR will restart every cycle_length epochs
        T_mult=1,           # Keep cycle length constant
        eta_min=1e-6
    )


mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# joint loss weights
joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

# KL Divergence

def kl_loss(mu, logvar):
    _loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return _loss

def quat_norm_loss(yhat):
    yhat = yhat.permute(0, 2, 1) # Permute back to (Batch, Seq, Channels)
    _yhat = yhat.reshape(-1, 4)  # Use reshape instead of view
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def pos_rec_loss(y, yhat):
    y = y.permute(0, 2, 1)
    yhat = yhat.permute(0, 2, 1)
    
    _yhat = yhat.reshape(-1, 4)
    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    _y_rot = y.reshape((y.shape[0], y.shape[1], -1, 4))
    # FIX: use _yhat_norm instead of _yhat
    _yhat_rot = _yhat_norm.reshape((y.shape[0], y.shape[1], -1, 4))

    # FIX: requires_grad=False is safer here
    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=False).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    _loss = torch.mean(_pos_diff_weighted)
    return _loss

def rot_rec_loss(y, yhat):
    y = y.permute(0, 2, 1)       # Permute back
    yhat = yhat.permute(0, 2, 1) # Permute back
    
    _y = y.reshape((-1, 4))      # Use reshape instead of view
    _yhat = yhat.reshape((-1, 4))
    
    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)
    
    _diff = qmul(_yhat_inv, _y)
    _len = torch.norm(_diff[:, 1:], dim=1)
    _atan = torch.atan2(_len, _diff[:, 0])
    _abs = torch.abs(_atan)
    
    _abs = _abs.reshape(-1, mocap_window_length, joint_count)
    _abs_weighted = _abs * joint_loss_weights
    _loss = torch.mean(_abs_weighted) 
    return _loss

# autoencoder loss function
def ae_loss(y, yhat, mu, std, beta):
    # function parameters
    # y: encoder input
    # yhat: decoder output (i.e. reconstructed encoder input)
    # disc_fake_output: discriminator output for encoder generated prior
    
    _norm_loss = quat_norm_loss(yhat)
    _pos_loss = pos_rec_loss(y, yhat)
    _rot_loss = rot_rec_loss(y, yhat)

    # kld loss
    _kl_loss = kl_loss(mu, std)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * quat_norm_loss_scale
    _total_loss += _pos_loss * pos_rec_loss_scale
    _total_loss += _rot_loss * rot_rec_loss_scale
    _total_loss += _kl_loss * beta
    
    return _total_loss, _norm_loss, _pos_loss, _rot_loss, _kl_loss

def vae_train_step(y, beta):
    
    yhat, mu, logvar = vae(y)

    _loss, _quat_norm_loss, _pos_rec_loss, _rot_rec_loss, _kl_loss = ae_loss(y, yhat, mu, logvar, beta) 

    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    
    torch.nn.utils.clip_grad_norm_(vae.parameters(), 0.01)

    optimizer.step()
    
    return _loss, _quat_norm_loss, _pos_rec_loss, _rot_rec_loss, _kl_loss

@torch.no_grad()
def vae_test_step(y, beta):
    
    yhat, mu, logvar = vae(y)

    _loss, _quat_norm_loss, _pos_rec_loss, _rot_rec_loss, _kl_loss = ae_loss(y, yhat, mu, logvar, beta) 
    
    return _loss, _quat_norm_loss, _pos_rec_loss, _rot_rec_loss, _kl_loss

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {}
    loss_history["lr"] = []
    loss_history["beta"] = []
    loss_history["train"] = []
    loss_history["test"] = []
    loss_history["norm"] = []
    loss_history["pos"] = []
    loss_history["rot"] = []
    loss_history["kl"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        current_beta = beta_schedule[epoch - 1]
        
        loss_history['lr'].append(current_lr)
        loss_history['beta'].append(current_beta)
        
        _train_loss_per_epoch = []
        _test_loss_per_epoch = []
        _quat_norm_loss_per_epoch = []
        _pos_rec_loss_per_epoch = []
        _rot_rec_loss_per_epoch = []
        _kl_loss_per_epoch = []
        
        for train_batch in train_dataloader:
            train_batch = train_batch.to(device)
            train_batch = train_batch.permute(0, 2, 1) # shape (B, pose_dim, 64)
            
            _loss, _quat_norm_loss, _pos_rec_loss, _quat_rec_loss, _kl_loss = vae_train_step(train_batch, current_beta)
            
            _loss = _loss.detach().cpu().numpy()
            _quat_norm_loss = _quat_norm_loss.detach().cpu().numpy()
            _pos_rec_loss = _pos_rec_loss.detach().cpu().numpy()
            _quat_rec_loss = _quat_rec_loss.detach().cpu().numpy()
            _kl_loss = _kl_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            _train_loss_per_epoch.append(_loss)
            _quat_norm_loss_per_epoch.append(_quat_norm_loss)
            _pos_rec_loss_per_epoch.append(_pos_rec_loss)
            _rot_rec_loss_per_epoch.append(_quat_rec_loss)
            _kl_loss_per_epoch.append(_kl_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _quat_norm_loss_per_epoch = np.mean(np.array(_quat_norm_loss_per_epoch))
        _pos_rec_loss_per_epoch = np.mean(np.array(_pos_rec_loss_per_epoch))
        _rot_rec_loss_per_epoch = np.mean(np.array(_rot_rec_loss_per_epoch))
        _kl_loss_per_epoch = np.mean(np.array(_kl_loss_per_epoch))

        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            test_batch = test_batch.permute(0, 2, 1) # shape (B, pose_dim, 64)
            
            _loss, _, _, _, _ = vae_test_step(test_batch, current_beta)
            
            _loss = _loss.detach().cpu().numpy()
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % save_weights_interval == 0 and save_weights == True:
            torch.save(vae.state_dict(), f"{save_weights_path}vae_weights_epoch_{epoch}")
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["norm"].append(_quat_norm_loss_per_epoch)
        loss_history["pos"].append(_pos_rec_loss_per_epoch)
        loss_history["rot"].append(_rot_rec_loss_per_epoch)
        loss_history["kl"].append(_kl_loss_per_epoch)
        
        print (f"epoch {epoch:03d}" + 
               f" : vae train: {_train_loss_per_epoch:01.4f}" +
               f" : vae test: {_test_loss_per_epoch:01.4f}" +
               f" : norm {_quat_norm_loss_per_epoch:01.4f}" +
               f" : pos {_pos_rec_loss_per_epoch:01.4f}" +
               f" : rot {_rot_rec_loss_per_epoch:01.4f}" +
               f" : kl {_kl_loss_per_epoch:01.4f}" +
               f" : lr {current_lr:01.4f}" +
               f" : beta {current_beta:01.4f}" +
               f" : time {(time.time()-start):01.2f}") 
        
        if (epoch + 1) % cycle_length == 0:
            scheduler.base_lrs = [base_lr * learning_rate_cycle_decay for base_lr in scheduler.base_lrs]
        
        scheduler.step()
        
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, f"{save_history_path}history_{epochs}.csv")

# plot history

def plot_training_history(loss_history, save_path):
    epochs = len(loss_history['train'])
    x = np.arange(1, epochs + 1)
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('VAE Training History', fontsize=18, y=0.98)
    
    # 1. Total Loss
    axs[0, 0].plot(x, loss_history['train'], label='Train Loss', color='#1f77b4', linewidth=2)
    axs[0, 0].plot(x, loss_history['test'], label='Test Loss', color='#ff7f0e', linewidth=2)
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()
    
    # 2. Reconstruction Losses
    axs[0, 1].plot(x, loss_history['pos'], label='Pos Rec', color='#2ca02c')
    axs[0, 1].plot(x, loss_history['rot'], label='Rot Rec', color='#d62728')
    axs[0, 1].plot(x, loss_history['norm'], label='Quat Norm', color='#9467bd')
    axs[0, 1].set_title('Reconstruction Losses')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()
    
    # 3. KL Divergence
    axs[1, 0].plot(x, loss_history['kl'], label='KL Loss', color='#8c564b', linewidth=2)
    axs[1, 0].set_title('KL Divergence')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()
    
    # 4. Beta Schedule
    axs[1, 1].plot(x, loss_history['beta'], label='Beta Value', color='#e377c2', linewidth=2)
    axs[1, 1].set_title('Cyclical Beta Schedule')
    axs[1, 1].set_ylabel('Beta')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()
    
    # 5. Learning Rate
    axs[2, 0].plot(x, loss_history['lr'], label='Learning Rate', color='#17becf', linewidth=2)
    axs[2, 0].set_title('Learning Rate (Cosine Annealing)')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('LR')
    axs[2, 0].grid(True, linestyle='--', alpha=0.6)
    axs[2, 0].legend()
    
    # Empty subplot for layout balance
    axs[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_training_history(loss_history, f"{save_history_path}history_{epochs}.png")

# save model weights
torch.save(vae.state_dict(), f"{save_weights_path}vae_weights_epoch_{epochs}")

# inference and rendering 

poseRenderer = PoseRenderer(edge_list)

def export_sequence_anim(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]

    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    
    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    
    fbx_tools.write(pred_fbx, file_name)

@torch.no_grad()
def encode_sequences(orig_sequence, frame_indices):
    vae.eval()
    latent_vectors = []
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + mocap_window_length

        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame]
        excerpt = np.expand_dims(excerpt, axis=0)
        # ADD PERMUTE HERE
        excerpt = torch.from_numpy(excerpt).reshape(1, mocap_window_length, pose_dim).permute(0, 2, 1).to(device)

        with torch.no_grad():
            mu, logvar = vae.encode(excerpt)
            latent_vector = vae.reparameterize(mu, logvar)

        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()
        latent_vectors.append(latent_vector)

    vae.train()
    return latent_vectors

@torch.no_grad()
def decode_sequence_encodings(sequence_encodings, seq_overlap, base_pose):
    vae.eval()

    seq_env = np.hanning(mocap_window_length)
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + mocap_window_length
    gen_sequence = np.full(shape=(gen_seq_length, joint_count, joint_dim), fill_value=base_pose)

    for excerpt_index in range(len(sequence_encodings)):
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)

        with torch.no_grad():
            excerpt_dec = vae.decode(latent_vector)

        # ADD PERMUTE HERE before squeezing and reshaping
        excerpt_dec = excerpt_dec.permute(0, 2, 1)
        excerpt_dec = torch.squeeze(excerpt_dec)
        excerpt_dec = excerpt_dec.detach().cpu().numpy()
        excerpt_dec = np.reshape(excerpt_dec, (-1, joint_count, joint_dim))

        gen_frame = excerpt_index * seq_overlap

        for si in range(mocap_window_length):
            for ji in range(joint_count): 
                current_quat = gen_sequence[gen_frame + si, ji, :]
                target_quat = excerpt_dec[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix)
                gen_sequence[gen_frame + si, ji, :] = mix_quat

    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((gen_seq_length, joint_count, joint_dim))
    gen_sequence = qfix(gen_sequence)

    vae.train()
    return gen_sequence
    
@torch.no_grad()
def decode_sequence_encodings_6d(sequence_encodings, seq_overlap, base_pose):
    vae.eval()
    
    seq_env = np.hanning(mocap_window_length) + 0.01
    
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + mocap_window_length

    # 1. Accumulators: We accumulate 3x3 matrices (9 values per joint)
    gen_sequence_accum = np.zeros(shape=(gen_seq_length, joint_count, 3, 3), dtype=np.float32)
    weight_accum = np.zeros(shape=(gen_seq_length, 1, 1), dtype=np.float32)

    for excerpt_index in range(seq_excerpt_count):
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)

        with torch.no_grad():
            excerpt_dec = vae.decode(latent_vector)
            
        excerpt_dec = excerpt_dec.permute(0, 2, 1)
        excerpt_dec = torch.squeeze(excerpt_dec).detach().cpu().numpy()
        excerpt_dec = np.reshape(excerpt_dec, (-1, joint_count, joint_dim))
        
        gen_frame = excerpt_index * seq_overlap

        for si in range(mocap_window_length):
            frame_quats = excerpt_dec[si]  
            
            # --- FIX 1: Enforce scalar_first=True for your (w,x,y,z) format ---
            frame_matrices = R.from_quat(frame_quats, scalar_first=True).as_matrix() 
            
            mix_weight = seq_env[si]
            
            gen_sequence_accum[gen_frame + si] += frame_matrices * mix_weight
            weight_accum[gen_frame + si] += mix_weight

    weight_accum[weight_accum == 0.0] = 1.0 
    
    # Expand weight accumulator so it broadcasts across the 3x3 matrix shape
    weight_accum_expanded = np.expand_dims(weight_accum, axis=-1)
    avg_matrices = gen_sequence_accum / weight_accum_expanded

    flat_matrices = avg_matrices.reshape(-1, 3, 3)
    
    # --- FIX 2: Convert back using scalar_first=True and cast to float32 ---
    gen_sequence = R.from_matrix(flat_matrices).as_quat(scalar_first=True)
    gen_sequence = gen_sequence.astype(np.float32) # Cast to single-precision float
    
    gen_sequence = gen_sequence.reshape(gen_seq_length, joint_count, joint_dim)
    gen_sequence = qfix(gen_sequence)
    gen_sequence = gen_sequence.reshape(gen_seq_length, -1)
    
    vae.train()
    return gen_sequence

def create_2d_latent_space_representation(sequence_excerpts):
    encodings = []
    excerpt_count = sequence_excerpts.shape[0]

    for eI in range(0, excerpt_count, batch_size):
        excerpt_batch = sequence_excerpts[eI:eI+batch_size]
        excerpt_batch = torch.from_numpy(excerpt_batch).to(device)
        # ADD PERMUTE HERE
        excerpt_batch = excerpt_batch.permute(0, 2, 1)

        mu, logvar = vae.encode(excerpt_batch)
        encoding_batch = vae.reparameterize(mu, logvar)
        
        # FLATTEN TEMPORAL AXIS FOR t-SNE
        encoding_batch = encoding_batch.detach().cpu()
        encoding_batch = encoding_batch.reshape(encoding_batch.size(0), -1)

        encodings.append(encoding_batch)

    encodings = torch.cat(encodings, dim=0).numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1) 
    Z_tsne = tsne.fit_transform(encodings)
    return Z_tsne

def create_2d_latent_space_image(Z_tsne, highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]

    plot_colors = ["green", "red", "blue", "magenta", "orange"]
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)
    
    for hI, hR in enumerate(highlight_excerpt_ranges):
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], s=0.8, c=plot_colors[hI], alpha=0.5)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=300)
    plt.close()
    
# create latent space plot

Z_tsne = create_2d_latent_space_representation(pose_sequence_excerpts)
create_2d_latent_space_image(Z_tsne, [], f"{save_path}latent_space_plot_epoch_{epochs}.png")

# create original sequence

orig_sequence = all_mocap_data[0]["motion"]["rot_local"].astype(np.float32)

orig_sequence.shape

seq_start = 1000
seq_length = 1000

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(orig_sequence[seq_start:seq_start+seq_length], f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.fbx")


# recontruct original sequence

seq_start = 1000
seq_length = 1000
seq_overlap = mocap_window_length // vae_latent_count // 4
#seq_overlap = 1
base_pose = np.reshape(orig_sequence[0], (joint_count, joint_dim))

seq_indices = [ frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings_6d(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")


# random walk in latent space
seq_start = 1000
seq_length = 1000

seq_indices = [seq_start]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

for index in range(0, seq_length // seq_overlap):
    # Centered random walk around 0 (values from -1.0 to 1.0)
    random_step = (np.random.random((vae_latent_dim, vae_latent_count)).astype(np.float32) - 0.5) * 2.0
    seq_encodings.append(seq_encodings[index] + random_step)
    
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")

# sequence offset following

seq_start = 1000
seq_length = 1000
    
seq_indices = [ seq_index for seq_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

offset_seq_encodings = []

for index in range(len(seq_encodings)):
    sin_value = np.sin(index / (len(seq_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(vae_latent_dim, vae_latent_count), dtype=np.float32) * sin_value * 4.0
    offset_seq_encoding = seq_encodings[index] + offset
    offset_seq_encodings.append(offset_seq_encoding)
    
gen_sequence = decode_sequence_encodings(offset_seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx")



# interpolate two original sequences

seq1_start = 1000
seq2_start = 2000
seq_length = 1000

seq1_indices = [ seq_index for seq_index in range(seq1_start, seq1_start + seq_length, seq_overlap)]
seq2_indices = [ seq_index for seq_index in range(seq2_start, seq2_start + seq_length, seq_overlap)]

seq1_encodings = encode_sequences(orig_sequence, seq1_indices)
seq2_encodings = encode_sequences(orig_sequence, seq2_indices)

mix_encodings = []

for index in range(len(seq1_encodings)):
    mix_factor = index / (len(seq1_indices) - 1)
    mix_encoding = seq1_encodings[index] * (1.0 - mix_factor) + seq2_encodings[index] * mix_factor
    mix_encodings.append(mix_encoding)

gen_sequence = decode_sequence_encodings(mix_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.gif")
export_sequence_fbx(gen_sequence, f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.fbx")

