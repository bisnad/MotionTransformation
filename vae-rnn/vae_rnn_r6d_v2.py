"""
A variational autoencoder for motion capture data of a solo dancer
this is for motion capture data that stores joint rotations and recorded in BVH or FBX format
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
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

mocap_file_path = "data/mocap"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None

"""
Model Settings (IMPROVED FOR SMALL DATASETS)
"""

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 256       # Reduced from 512 to prevent overfitting
ae_dense_layer_sizes = [ 256 ] # Reduced from 512 to prevent overfitting
ae_dropout = 0.3              # Added dropout for regularization

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
encoder_weights_file = "results/weights/encoder_weights_epoch_600"
decoder_weights_file = "results/weights/decoder_weights_epoch_600"

"""
Training Settings
"""

sequence_offset = 2 
batch_size = 16
test_percentage = 0.2
ae_learning_rate = 1e-4

# IMPROVED LOSS WEIGHTS
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_vel_loss_scale = 0.5       # New: Penalizes jitter (first derivative)
ae_acc_loss_scale = 0.2       # New: Penalizes jitter (second derivative)
ae_kld_loss_scale = 0.0       

kld_scale_cycle_duration = 100
kld_scale_min_const_duration = 20
kld_scale_max_const_duration = 20
min_kld_scale = 0.0
max_kld_scale = 0.1

epochs = 600
model_save_interval = 50
save_history = True

"""
Visualization Settings
"""
view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

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
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] 

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

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
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
        joint_loss_weights = json.load(f)["joint_loss_weights"]
else:
    joint_loss_weights = [1.0] * joint_count

"""
Create Dataset
"""

pose_sequence_excerpts = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]

    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - sequence_length, sequence_offset):
        pose_sequence_excerpt = pose_sequence[seq_excerpt_start:seq_excerpt_start + sequence_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)

pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts

    def __len__(self):
        return self.sequence_excerpts.shape[0]

    def __getitem__(self, idx):
        # Adding a tiny amount of noise during fetching acts as data augmentation for small datasets
        data = self.sequence_excerpts[idx, ...]
        noise = np.random.normal(0, 1e-4, data.shape).astype(np.float32)
        return data + noise

full_dataset = SequenceDataset(pose_sequence_excerpts)
dataset_size = len(full_dataset)
test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Create Models (IMPROVED ARCHITECTURE)
"""

class Encoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout=0.3):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes

        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True, dropout=dropout)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))

        dense_layers = []
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        dense_layers.append(("encoder_dropout_0", nn.Dropout(dropout)))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            dense_layers.append(("encoder_dropout_{}".format(layer_index), nn.Dropout(dropout)))

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)

    def forward(self, x):
        x, (_, _) = self.rnn_layers(x)
        x = x[:, -1, :] 
        x = self.dense_layers(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        return mu, std

encoder = Encoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes, ae_dropout).to(device)

class Decoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout=0.3):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        dense_layers = []
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))
        dense_layers.append(("decoder_dropout_0", nn.Dropout(dropout)))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            dense_layers.append(("decoder_dropout_{}".format(layer_index), nn.Dropout(dropout)))

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))

        rnn_layers = []
        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True, dropout=dropout)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))

        final_layers = []
        final_layers.append(("decoder_dense_final", nn.Linear(self.rnn_layer_size, self.pose_dim)))
        self.final_layers = nn.Sequential(OrderedDict(final_layers))

    def forward(self, x):
        x = self.dense_layers(x)
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.sequence_length, 1)
        x, (_, _) = self.rnn_layers(x)
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size) 
        yhat = self.final_layers(x_reshaped)
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        return yhat

ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()
decoder = Decoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes_reversed, ae_dropout).to(device)

if load_weights:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

"""
Training (IMPROVED LOSS FUNCTIONS)
"""

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32).reshape(1, 1, -1).to(device)

def variational_loss(mu, std):
    return -0.5 * torch.mean(1 + 2 * torch.log(std) - mu.pow(2) - (std.pow(2)))

def reparameterize(mu, std):
    return mu + std * torch.randn_like(std)

def ae_norm_loss(yhat):
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    return torch.mean((_norm - 1.0) ** 2)

def forward_kinematics(rotations, root_positions):
    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4

    toffsets = torch.tensor(offsets).to(device)
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def ae_quat_loss(y, yhat):
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))
    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)
    
    _diff = qmul(_yhat_inv, _y)
    _len = torch.norm(_diff[:, 1:], dim=1)
    _atan = torch.atan2(_len, _diff[:, 0])
    _abs = torch.abs(_atan).reshape(-1, sequence_length, joint_count)
    return torch.mean(_abs * joint_loss_weights) 

def ae_loss(y, yhat, mu, std):
    _norm_loss = ae_norm_loss(yhat)
    _quat_loss = ae_quat_loss(y, yhat)
    _ae_kld_loss = variational_loss(mu, std)

    # Prepare for positional and temporal losses (FK calculated only once per batch)
    _yhat_norm = nn.functional.normalize(yhat.view(-1, 4), p=2, dim=1).view((y.shape[0], y.shape[1], -1, 4))
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    
    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)
    
    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_norm, zero_trajectory)

    # Positional Loss
    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    _pos_loss = torch.mean(_pos_diff * joint_loss_weights)

    # Velocity Loss (1st derivative)
    _y_vel = _y_pos[:, 1:, :, :] - _y_pos[:, :-1, :, :]
    _yhat_vel = _yhat_pos[:, 1:, :, :] - _yhat_pos[:, :-1, :, :]
    _vel_diff = torch.norm((_y_vel - _yhat_vel), dim=3)
    _vel_loss = torch.mean(_vel_diff * joint_loss_weights)

    # Acceleration Loss (2nd derivative)
    _y_acc = _y_pos[:, 2:, :, :] - 2 * _y_pos[:, 1:-1, :, :] + _y_pos[:, :-2, :, :]
    _yhat_acc = _yhat_pos[:, 2:, :, :] - 2 * _yhat_pos[:, 1:-1, :, :] + _yhat_pos[:, :-2, :, :]
    _acc_diff = torch.norm((_y_acc - _yhat_acc), dim=3)
    _acc_loss = torch.mean(_acc_diff * joint_loss_weights)

    _total_loss = (_norm_loss * ae_norm_loss_scale) + \
                  (_pos_loss * ae_pos_loss_scale) + \
                  (_quat_loss * ae_quat_loss_scale) + \
                  (_vel_loss * ae_vel_loss_scale) + \
                  (_acc_loss * ae_acc_loss_scale) + \
                  (_ae_kld_loss * ae_kld_loss_scale)

    return _total_loss, _norm_loss, _pos_loss, _vel_loss, _acc_loss, _quat_loss, _ae_kld_loss

# ---------------------------------------------------------
# Training Loop Logic
# ---------------------------------------------------------

def calc_kld_scales():
    kld_scales = []
    for e in range(epochs):
        cycle_step = e % kld_scale_cycle_duration
        if cycle_step < kld_scale_min_const_duration:
            kld_scales.append(min_kld_scale)
        elif cycle_step > kld_scale_cycle_duration - kld_scale_max_const_duration:
            kld_scales.append(max_kld_scale)
        else:
            lin_step = cycle_step - kld_scale_min_const_duration
            kld_scale = min_kld_scale + (max_kld_scale - min_kld_scale) * lin_step / (kld_scale_cycle_duration - kld_scale_min_const_duration - kld_scale_max_const_duration)
            kld_scales.append(kld_scale)
    return kld_scales

kld_scales = calc_kld_scales()

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=100, gamma=0.316) 

def ae_train_step(target_poses):
    encoder_output = encoder(target_poses)
    mu = torch.tanh(encoder_output[0])
    std = torch.abs(torch.tanh(encoder_output[1])) + 0.00001
    decoder_input = reparameterize(mu, std)
    pred_poses = decoder(decoder_input)

    _ae_loss, _ae_norm, _ae_pos, _ae_vel, _ae_acc, _ae_quat, _ae_kld = ae_loss(target_poses, pred_poses, mu, std) 

    ae_optimizer.zero_grad()
    _ae_loss.backward()
    ae_optimizer.step()

    return _ae_loss, _ae_norm, _ae_pos, _ae_vel, _ae_acc, _ae_quat, _ae_kld

def ae_test_step(target_poses):
    with torch.no_grad():
        encoder_output = encoder(target_poses)
        mu = torch.tanh(encoder_output[0])
        std = torch.abs(torch.tanh(encoder_output[1])) + 0.00001
        decoder_input = reparameterize(mu, std)
        pred_poses = decoder(decoder_input)

        _ae_loss, _ae_norm, _ae_pos, _ae_vel, _ae_acc, _ae_quat, _ae_kld = ae_loss(target_poses, pred_poses, mu, std) 
        return _ae_loss, _ae_norm, _ae_pos, _ae_vel, _ae_acc, _ae_quat, _ae_kld

def train(train_dataloader, test_dataloader, epochs):
    global ae_kld_loss_scale

    loss_history = {"ae train": [], "ae test": [], "ae pos": [], "ae vel": [], "ae acc": [], "ae quat": [], "ae kld": []}

    for epoch in range(epochs):
        start = time.time()
        ae_kld_loss_scale = kld_scales[epoch]

        ae_train_loss, ae_pos_loss, ae_vel_loss, ae_acc_loss, ae_quat_loss, ae_kld_loss = [], [], [], [], [], []

        for train_batch in train_dataloader:
            train_batch = train_batch.to(device)
            _ae_loss, _, _ae_pos, _ae_vel, _ae_acc, _ae_quat, _ae_kld = ae_train_step(train_batch)

            ae_train_loss.append(_ae_loss.item())
            ae_pos_loss.append(_ae_pos.item())
            ae_vel_loss.append(_ae_vel.item())
            ae_acc_loss.append(_ae_acc.item())
            ae_quat_loss.append(_ae_quat.item())
            ae_kld_loss.append(_ae_kld.item())

        ae_test_loss = []
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            _ae_loss, _, _, _, _, _, _ = ae_test_step(test_batch)
            ae_test_loss.append(_ae_loss.item())

        if epoch % model_save_interval == 0 and save_weights:
            torch.save(encoder.state_dict(), f"results/weights/encoder_weights_epoch_{epoch}")
            torch.save(decoder.state_dict(), f"results/weights/decoder_weights_epoch_{epoch}")

        print(f'epoch {epoch + 1} : ae train: {np.mean(ae_train_loss):01.4f} test: {np.mean(ae_test_loss):01.4f} pos: {np.mean(ae_pos_loss):01.4f} vel: {np.mean(ae_vel_loss):01.4f} acc: {np.mean(ae_acc_loss):01.4f} time: {time.time()-start:01.2f}')
        ae_scheduler.step()

    return loss_history

loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

if save_weights:
    torch.save(encoder.state_dict(), f"results/weights/encoder_weights_epoch_{epochs}")
    torch.save(decoder.state_dict(), f"results/weights/decoder_weights_epoch_{epochs}")

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

def encode_sequences(orig_sequence, frame_indices):
    
    encoder.eval()
    
    latent_vectors = []
    
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + sequence_length

        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame]
        excerpt = np.expand_dims(excerpt, axis=0)
        excerpt = torch.from_numpy(excerpt).reshape(1, sequence_length, pose_dim).to(device)

        with torch.no_grad():

            encoder_output = encoder(excerpt)

            encoder_output_mu = encoder_output[0]
            encoder_output_std = encoder_output[1]
            mu = torch.tanh(encoder_output_mu)
            std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
            
            latent_vector = reparameterize(mu, std)
            
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
        
    encoder.train()
        
    return latent_vectors

def decode_sequence_encodings(sequence_encodings, seq_overlap, base_pose):
    
    decoder.eval()
    
    seq_env = np.hanning(sequence_length)
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + sequence_length

    gen_sequence = np.full(shape=(gen_seq_length, joint_count, joint_dim), fill_value=base_pose)
    
    for excerpt_index in range(len(sequence_encodings)):
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        with torch.no_grad():
            excerpt_dec = decoder(latent_vector)
        
        excerpt_dec = torch.squeeze(excerpt_dec)
        excerpt_dec = excerpt_dec.detach().cpu().numpy()
        excerpt_dec = np.reshape(excerpt_dec, (-1, joint_count, joint_dim))
        
        gen_frame = excerpt_index * seq_overlap
        
        for si in range(sequence_length):
            for ji in range(joint_count): 
                current_quat = gen_sequence[gen_frame + si, ji, :]
                target_quat = excerpt_dec[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix )
                gen_sequence[gen_frame + si, ji, :] = mix_quat
        
    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((gen_seq_length, joint_count, joint_dim))
    gen_sequence = qfix(gen_sequence)

    decoder.train()
    
    return gen_sequence
    
def create_2d_latent_space_representation(sequence_excerpts):

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt_batch = sequence_excerpts[eI:eI+batch_size]
        
        #print("excerpt_batch s ", excerpt_batch.shape)
        
        excerpt_batch = torch.from_numpy(excerpt_batch).to(device)
        
        encoder_output = encoder(excerpt_batch)

        encoder_output_mu = encoder_output[0]
        encoder_output_std = encoder_output[1]
        mu = torch.tanh(encoder_output_mu)
        std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
        
        encoding_batch = reparameterize(mu, std)
        
        #print("encoding_batch s ", encoding_batch.shape)
        
        encoding_batch = encoding_batch.detach().cpu()

        encodings.append(encoding_batch)
        
    encodings = torch.cat(encodings, dim=0)
    
    #print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

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
    ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)
    
    for hI, hR in enumerate(highlight_excerpt_ranges):
        ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], s=0.8, c=plot_colors[hI], alpha=0.5)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=300)
    plt.close()
    
# create latent space plot

Z_tsne = create_2d_latent_space_representation(pose_sequence_excerpts)
create_2d_latent_space_image(Z_tsne, [], "latent_space_plot_epoch_{}.png".format(epochs))

# create original sequence

orig_sequence = all_mocap_data[0]["motion"]["rot_local"].astype(np.float32)

seq_start = 7000
seq_length = 2000

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.gif".format(seq_start, seq_length))
export_sequence_fbx(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.fbx".format(seq_start, seq_length))


# recontruct original sequence

seq_start = 1000
seq_length = 1000
seq_overlap = 16 # 2 for 8, 32 for 128
base_pose = np.reshape(orig_sequence[0], (joint_count, joint_dim))

seq_indices = [ frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))


# random walk in latent space
seq_start = 1000
seq_length = 1000

seq_indices = [seq_start]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

for index in range(0, seq_length // seq_overlap):
    random_step = np.random.random((latent_dim)).astype(np.float32) * 2.0
    seq_encodings.append(seq_encodings[index] + random_step)
    
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/seq_randwalk_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_randwalk_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))

# sequence offset following

seq_start = 1000
seq_length = 1000
    
seq_indices = [ seq_index for seq_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

offset_seq_encodings = []

for index in range(len(seq_encodings)):
    sin_value = np.sin(index / (len(seq_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_seq_encoding = seq_encodings[index] + offset
    offset_seq_encodings.append(offset_seq_encoding)
    
gen_sequence = decode_sequence_encodings(offset_seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/seq_offset_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_offset_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))



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
export_sequence_anim(gen_sequence, "results/anims/seq_mix_epoch_{}_seq1_start_{}_seq2_start_{}_length_{}.gif".format(epochs, seq1_start, seq2_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_mix_epoch_{}_seq1_start_{}_seq2_start_{}_length_{}.fbx".format(epochs, seq1_start, seq2_start, seq_length))

