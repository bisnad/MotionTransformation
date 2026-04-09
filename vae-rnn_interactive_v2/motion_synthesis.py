import torch
from torch import nn
import numpy as np
import torch.nn.functional as nnF

from common.quaternion import qmul, qrot, qnormalize_np
from common.quaternion_torch import slerp, qfix

config = {"skeleton": None,
          "model_encoder": None,
          "model_decoder": None,
          "device": "cuda",
          "seq_window_length": 8,
          "seq_window_overlap": 2,
          "orig_sequences": [],
          "orig_seq1_index": 0,
          "orig_seq2_index": 1
          }

def quaternion_to_yaw(q):
    # q: (..., 4)
    # Returns yaw angle around vertical axis
    # Assumes w,x,y,z order (PyTorch default)
    # Convert quaternion to yaw (rotation about y/up axis)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t3 = 2.0 * (w * y + z * x)
    t4 = 1.0 - 2.0 * (y * y + x * x)
    yaw = torch.atan2(t3, t4)  # shape (...,)
    return yaw

def yaw_to_quaternion(yaw):
    # Converts yaw angle to quaternion (w,x,y,z)
    half_yaw = 0.5 * yaw
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)
    # Only rotation about y axis: [w, x, y, z]
    zeros = torch.zeros_like(cy)
    return torch.stack([cy, zeros, sy, zeros], dim=-1)

def ensure_quaternion_continuity(q):
    """
    Ensures quaternion continuity along the time axis by flipping quaternions
    whose dot product with the previous frame is negative.
    q: torch tensor of shape [frames, joints, 4]
    Returns: tensor on the same device, same shape, corrected for continuity.
    """
    # Compute dot products between consecutive frames, for every joint
    dots = torch.sum(q[1:] * q[:-1], dim=2)
    # Mask for where to flip sign
    mask = dots < 0
    # Cumulative flips (parity): flip all odd-parity segments
    flips = mask.cumsum(dim=0) % 2 == 1
    # Make a copy so the input is not changed
    q_out = q.clone()
    # Expand flips to shape [frames-1, joints, 1] for broadcasting
    flips_expand = flips.unsqueeze(2).expand_as(q_out[1:])
    # Flip signs in all frames/joints where parity is odd
    q_out[1:] = torch.where(flips_expand, -q_out[1:], q_out[1:])
    return q_out

def slerp_batch(q0, q1, t):
    """
    Vectorized slerp with antipodal fix for a batch of quaternions.

    q0, q1: arrays of shape [frames, joints, 4]
    t: arrays of shape [frames, joints] - blend weight (0=start, 1=end)
    Returns: blended array, [frames, joints, 4]
    """
    # Normalize input quaternions
    q0 = nnF.normalize(q0, p=2, dim=2)
    q1 = nnF.normalize(q1, p=2, dim=2)
    # Dot product of [frames, joints], unsqueeze to [frames, joints, 1]
    dot = torch.sum(q0 * q1, dim=2, keepdim=True)
    # Antipodal fix: flip q1 where dot < 0
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp(dot, -1.0, 1.0)
    # Omega and sin(omega)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Linear where angle small
    use_linear = sin_omega < 1e-4
    t = t.unsqueeze(2)  # [frames, joints, 1]
    s0 = torch.where(use_linear, 1.0 - t, torch.sin((1.0 - t) * omega) / (sin_omega + 1e-8))
    s1 = torch.where(use_linear, t, torch.sin(t * omega) / (sin_omega + 1e-8))
    blended = s0 * q0 + s1 * q1
    # Normalize result
    blended = nnF.normalize(blended, p=2, dim=2)
    return blended

def smooth_1d(data_1d, window_length, window_type="hanning"):
    """
    Smooth 1D signal with specified window on a torch tensor.
    data_1d: shape [N] or [L], torch tensor
    Returns: smoothed tensor, shape [N]
    """
    if data_1d.dim() != 1:
        raise ValueError("smooth only accepts 1 dimension tensors.")
    if data_1d.numel() < window_length:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_length < 3:
        return data_1d

    # Select window function
    if window_type == 'flat':
        window = torch.ones(window_length, device=data_1d.device, dtype=data_1d.dtype)
    elif window_type == 'hanning':
        window = torch.hann_window(window_length, device=data_1d.device, dtype=data_1d.dtype)
    elif window_type == 'hamming':
        window = torch.hamming_window(window_length, device=data_1d.device, dtype=data_1d.dtype)
    elif window_type == 'bartlett':
        window = torch.bartlett_window(window_length, device=data_1d.device, dtype=data_1d.dtype)
    elif window_type == 'blackman':
        window = torch.blackman_window(window_length, device=data_1d.device, dtype=data_1d.dtype)
    else:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.")

    # Normalize window sum
    window = window / window.sum()
    # Pad input symmetrically (replicating boundary, mimicking numpy flip)
    pad_left = data_1d[:(window_length-1)//2].flip(0)
    pad_right = data_1d[-((window_length-1)//2):].flip(0)
    data_padded = torch.cat([pad_left, data_1d, pad_right], dim=0)
    # 1D convolution and crop
    out = nnF.conv1d(
        data_padded.view(1, 1, -1),
        window.view(1, 1, -1)
    )
    out = out[0, 0]  # remove batch/channel
    return out

def smooth(data, window_length, window_type="hanning"):
    """
    Multi-dimensional torch tensor smoothing.
    data: shape [L, ...], torch tensor
    Returns: tensor of same shape, smoothed independently along leading axis.
    """
    orig_shape = data.shape
    # Flatten all remaining dims for easier iteration
    data_flat = data.reshape(orig_shape[0], -1)  # [L, D]
    smoothed = []
    for d in range(data_flat.shape[1]):
        # Smooth each dimension independently
        smoothed_1d = smooth_1d(data_flat[:, d], window_length, window_type)
        smoothed.append(smoothed_1d)
    smoothed = torch.stack(smoothed, dim=1)  # [L, D]
    # Restore to original shape
    smoothed = smoothed.reshape(orig_shape)
    return smoothed

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model_encoder = config["model_encoder"]
        self.model_decoder = config["model_decoder"]
        self.device = config["device"]
        self.seq_window_length = config["seq_window_length"]
        self.seq_window_overlap = config["seq_window_overlap"]
        self.orig_sequences = config["orig_sequences"]
        self.orig_seq1_index = config["orig_seq1_index"]
        self.orig_seq2_index = config["orig_seq2_index"]
                
        self.orig_seq1 = self.orig_sequences[self.orig_seq1_index]
        self.orig_seq2 = self.orig_sequences[self.orig_seq2_index]
        
        self.orig_seq1_changed = False
        self.orig_seq2_changed = False
        
        self.seq_window_offset = self.seq_window_length - self.seq_window_overlap
        
        self.seq1_length = self.orig_seq1.shape[0]
        self.seq2_length = self.orig_seq2.shape[0]
        
        self.joint_count = self.orig_seq1.shape[1]
        self.joint_dim = self.orig_seq1.shape[2]
        self.pose_dim = self.joint_count * self.joint_dim
        self.joint_offsets = self.skeleton ["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton ["parents"]
        self.joint_children = self.skeleton ["children"]
        
        self._create_edge_list()
        
        self.orig_seq1_frame_index = self.seq_window_offset
        self.orig_seq2_frame_index = self.seq_window_offset
        
        self.orig_seq1_frame_incr = self.seq_window_offset
        self.orig_seq2_frame_incr = self.seq_window_offset
        self.orig_seq1_frame_range = [0, self.seq1_length - self.seq_window_length]
        self.orig_seq2_frame_range = [0, self.seq2_length - self.seq_window_length]
        
        self.encoding_mix = torch.zeros((1, self.model_encoder.latent_dim)).to(self.device)
        self.encoding_offset = torch.zeros((1, self.model_encoder.latent_dim)).to(self.device)
        
        #self.gen_seq = torch.from_numpy(self.orig_seq[:self.seq_window_length, ...]).to(self.device)
        self.gen_seq = torch.Tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.seq_window_length, self.joint_count, 1).to(self.device)

        self.gen_seq_window = None
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
        self.seq_update_index = 0
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setSeq1Index(self, index):
        
        self.orig_seq1_index = min(index, len(self.orig_sequences)) 
        self.orig_seq1_changed = True
        
    def setSeq2Index(self, index):
        
        self.orig_seq2_index = min(index, len(self.orig_sequences)) 
        self.orig_seq2_changed = True
    
    def changeSeq1(self):
        
        self.orig_seq1 = self.orig_sequences[self.orig_seq1_index]
        self.seq1_length = self.orig_seq1.shape[0]
        self.orig_seq1_frame_index = self.seq_window_offset
        self.orig_seq1_frame_range = [0, self.seq1_length - self.seq_window_length]
        
        self.orig_seq1_changed = False
        
    def changeSeq2(self):
        
        self.orig_seq2 = self.orig_sequences[self.orig_seq2_index]
        self.seq2_length = self.orig_seq2.shape[0]
        self.orig_seq2_frame_index = self.seq_window_offset
        self.orig_seq2_frame_range = [0, self.seq2_length - self.seq_window_length]
        
        self.orig_seq2_changed = False
                
    def setSeq1FrameIndex(self, index):
        self.orig_seq1_frame_index = min(index, self.seq1_length - self.seq_window_length)

    def setSeq2FrameIndex(self, index):
        self.orig_seq2_frame_index = min(index, self.seq2_length - self.seq_window_length)
           
    def setSeq1FrameRange(self, startFrame, endFrame):
        self.orig_seq1_frame_range[0] = min(startFrame, self.seq1_length - self.seq_window_length)
        self.orig_seq1_frame_range[1] = min(endFrame, self.seq1_length - self.seq_window_length)
        
    def setSeq2FrameRange(self, startFrame, endFrame):
        self.orig_seq2_frame_range[0] = min(startFrame, self.seq2_length - self.seq_window_length)
        self.orig_seq2_frame_range[1] = min(endFrame, self.seq2_length - self.seq_window_length)
        
    def setSeq1FrameIncrement(self, incr):
        self.orig_seq1_frame_incr = incr
        
    def setSeq2FrameIncrement(self, incr):
        self.orig_seq2_frame_incr = incr
    
    def setEncodingMix(self, mix):
        self.encoding_mix = torch.tensor(mix, dtype=torch.float32).to(self.device)
        
    def setEncodingOffset(self, offset):
        self.encoding_offset = torch.tensor(offset, dtype=torch.float32).to(self.device)
                
    def update(self):
        
        if self.orig_seq1_changed == True:
            self.changeSeq1()
            
        if self.orig_seq2_changed == True:
            self.changeSeq2()

        #print("self.seq_update_index ", self.seq_update_index)
        
        # generate next skel pose
        pred_pose = self.gen_seq[self.seq_update_index, ...]

        pred_pose = pred_pose.reshape((-1, 4))
        pred_pose = nn.functional.normalize(pred_pose, p=2, dim=1)
        pred_pose = pred_pose.reshape((1, self.joint_count, self.joint_dim))
        
        zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
        zero_trajectory = zero_trajectory.to(self.device)
        
        self.synth_pose_wpos, self.synth_pose_wrot = self._forward_kinematics(torch.unsqueeze(pred_pose,dim=0), zero_trajectory)
        
        self.synth_pose_wpos = self.synth_pose_wpos.detach().cpu().numpy()
        self.synth_pose_wpos = self.synth_pose_wpos.reshape((self.joint_count, 3))
        
        self.synth_pose_wrot = self.synth_pose_wrot.detach().cpu().numpy()
        self.synth_pose_wrot = self.synth_pose_wrot.reshape((self.joint_count, 4))
        
        self.seq_update_index += 1
        
        if self.seq_update_index >= self.seq_window_offset:
            
            #print("_gen")
            self._gen()
            self._blend()

            self.seq_update_index = 0
        
    def _reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z 

    def _gen(self):
         
        # encode orig seq window 1 and orig seq window 2
        
        #print("orig seq1 excerpt from ", self.orig_seq_frame_index1, " to ", (self.orig_seq_frame_index1 + self.seq_window_length) )
        
        orig_seq1_window = self.orig_seq1[self.orig_seq1_frame_index:self.orig_seq1_frame_index + self.seq_window_length, ...]
        orig_seq1_window = torch.from_numpy(orig_seq1_window).reshape((1, self.seq_window_length, self.pose_dim)).to(self.device)
        
        #print("orig seq2 excerpt from ", self.orig_seq_frame_index2, " to ", (self.orig_seq_frame_index2 + self.seq_window_length) )

        orig_seq2_window = self.orig_seq2[self.orig_seq2_frame_index:self.orig_seq2_frame_index + self.seq_window_length, ...]
        orig_seq2_window = torch.from_numpy(orig_seq2_window).reshape((1, self.seq_window_length, self.pose_dim)).to(self.device)
        
        with torch.no_grad():
            encoder_output1 = self.model_encoder(orig_seq1_window)
            encoder_output1_mu = encoder_output1[0]
            encoder_output1_std = encoder_output1[1]
            mu1 = torch.tanh(encoder_output1_mu)
            std1 = torch.abs(torch.tanh(encoder_output1_std)) + 0.00001
            encoding1 = self._reparameterize(mu1, std1)
            
            encoder_output2 = self.model_encoder(orig_seq2_window)
            encoder_output2_mu = encoder_output2[0]
            encoder_output2_std = encoder_output2[1]
            mu2 = torch.tanh(encoder_output2_mu)
            std2 = torch.abs(torch.tanh(encoder_output2_std)) + 0.00001
            encoding2 = self._reparameterize(mu2, std2)

        # mix encoding 1 and encoding 2
        encoding = encoding1 * (1.0 - self.encoding_mix) + encoding2 * self.encoding_mix
        
        # add encoding offset
        encoding += self.encoding_offset 
        
        # decode encoding
        with torch.no_grad():
            self.gen_seq_window = self.model_decoder(encoding)
            
        self.gen_seq_window = self.gen_seq_window.reshape(self.seq_window_length, self.joint_count, self.joint_dim)
        
        self.gen_seq_window = nnF.normalize(self.gen_seq_window , dim=2)
        self.gen_seq_window = qfix(self.gen_seq_window)
        
        # increment orig frame index 1 and orig frame index 2
        self.orig_seq1_frame_index += self.orig_seq1_frame_incr
        if self.orig_seq1_frame_index < self.orig_seq1_frame_range[0]:
            self.orig_seq1_frame_index = self.orig_seq1_frame_range[0]  
        elif self.orig_seq1_frame_index >= self.orig_seq1_frame_range[1]:
            self.orig_seq1_frame_index = self.orig_seq1_frame_range[0]  
            
        self.orig_seq2_frame_index += self.orig_seq2_frame_incr
        if self.orig_seq2_frame_index < self.orig_seq2_frame_range[0]:
            self.orig_seq2_frame_index = self.orig_seq2_frame_range[0]  
        elif self.orig_seq2_frame_index >= self.orig_seq2_frame_range[1]:
            self.orig_seq2_frame_index = self.orig_seq2_frame_range[0]  

    def _blend(self):
        """
        Efficient quaternion blending using PyTorch tensors for all critical operations.
        Minimizes CPU/GPU transfers and keeps data on GPU where possible.
        """

        # --- 1. Roll previous sequence window for new blend region ---
        self.gen_seq = torch.roll(self.gen_seq, -self.seq_window_offset, 0)
    
        # --- 2. Enforce quaternion continuity (antipodal fix) on both inputs ---
        self.gen_seq = ensure_quaternion_continuity(self.gen_seq)
        self.gen_seq_window = ensure_quaternion_continuity(self.gen_seq_window)
    
        # --- 3. Extract overlap region to blend ---
        overlap = self.seq_window_overlap
        seqA = self.gen_seq[:overlap]             # [overlap, joints, 4]
        seqB = self.gen_seq_window[:overlap]      # [overlap, joints, 4]
    
        # --- 4. Prepare blend slope weights ---
        blend_slope = torch.linspace(0.0, 1.0, overlap, device=self.device).unsqueeze(1).repeat(1, self.joint_count)
    
        # --- 5. Slerp blend with PyTorch tensors ---
        blend_seq = slerp_batch(seqA, seqB, blend_slope)   # [overlap, joints, 4]
        
        # --- 6. Temporal smoothing ---
        #blend_seq_smoothed = smooth(blend_seq, 5)  # [overlap, joints, 4]
        #blend_seq = nnF.normalize(blend_seq_smoothed, p=2, dim=2)  # Extra normalization after smoothing
    
        # --- 7. Final continuity check and reintegration ---
        blend_seq = ensure_quaternion_continuity(blend_seq)
        self.gen_seq[:overlap] = blend_seq                   # Blend region
        self.gen_seq[overlap:] = self.gen_seq_window[overlap:]      # Rest of window
    
        # --- 8. Final global continuity check ---
        self.gen_seq = ensure_quaternion_continuity(self.gen_seq)

    # def _forward_kinematics(self, rotations, root_positions, prev_root_yaw=None):
    #     """
    #     Perform forward kinematics using the given trajectory and local rotations.
    #     Implements root joint realignment to enforce global yaw continuity.
    #     Arguments (where N = batch size, L = sequence length, J = number of joints):
    #     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
    #     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    #     -- prev_root_yaw: Optional, previous window's final global root yaw (float or tensor)
    #     """
    
    #     assert len(rotations.shape) == 4 and rotations.shape[-1] == 4
    #     N, L, J = rotations.shape[:3]
    #     toffsets = torch.tensor(self.joint_offsets).to(self.device)
    #     expanded_offsets = toffsets.expand(N, L, self.joint_offsets.shape[0], self.joint_offsets.shape[1])
    #     positions_world = []
    #     rotations_world = []
    
    #     # ---- Root joint realignment step ----
    #     # Get current yaw angles
    #     root_quat = rotations[:, :, 0]   # shape (N, L, 4)
    #     root_yaw = quaternion_to_yaw(root_quat)  # shape (N, L)
    
    #     if prev_root_yaw is not None:
    #         # Compute minimal yaw delta for first frame
    #         # Correct the entire sequence by this delta to align with previous window
    #         delta_yaw = prev_root_yaw - root_yaw[:, 0]  # shape (N,)
    #         # Broadcast correction across sequence
    #         correction = delta_yaw.unsqueeze(-1)
    #         root_yaw_aligned = root_yaw + correction
    #         # Replace root_quat in rotations with realigned version
    #         aligned_root_quat = yaw_to_quaternion(root_yaw_aligned)
    #         rotations = rotations.clone()
    #         rotations[:, :, 0] = nnF.normalize(aligned_root_quat, p=2, dim=-1)  # normalized for safety
    
    #     # Forward kinematics as before
    #     for jI in range(self.joint_offsets.shape[0]):
    #         if self.joint_parents[jI] == -1:
    #             positions_world.append(root_positions)
    #             rotations_world.append(rotations[:, :, 0])
    #         else:
    #             positions_world.append(
    #                 qrot(rotations_world[self.joint_parents[jI]], expanded_offsets[:, :, jI])
    #                 + positions_world[self.joint_parents[jI]])
    #             if len(self.joint_children[jI]) > 0:
    #                 rotations_world.append(qmul(rotations_world[self.joint_parents[jI]], rotations[:, :, jI]))
    #             else:
    #                 rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]).to(self.device))
    
    #     # Return world positions and rotations
    #     return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)

    
    def _forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4
        
        toffsets = torch.tensor(self.joint_offsets).to(self.device)
        
        positions_world = []
        rotations_world = []

        expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], self.joint_offsets.shape[0], self.joint_offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for jI in range(self.joint_offsets.shape[0]):
            if self.joint_parents[jI] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self.joint_parents[jI]], expanded_offsets[:, :, jI]) \
                                       + positions_world[self.joint_parents[jI]])
                if len(self.joint_children[jI]) > 0:
                    rotations_world.append(qmul(rotations_world[self.joint_parents[jI]], rotations[:, :, jI]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]).to(self.device))
                    
        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)
        