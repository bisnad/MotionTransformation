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

        self.joint_children = self.skeleton ["children"]
        
        self._calc_pos_normalisation()
        
        self._create_edge_list()
        
        self.orig_seq1_frame_index = self.seq_window_offset
        self.orig_seq2_frame_index = self.seq_window_offset
        
        self.orig_seq1_frame_incr = self.seq_window_offset
        self.orig_seq2_frame_incr = self.seq_window_offset
        self.orig_seq1_frame_range = [0, self.seq1_length - self.seq_window_length]
        self.orig_seq2_frame_range = [0, self.seq2_length - self.seq_window_length]
        
        self.encoding_mix = torch.zeros((1, self.model_encoder.latent_dim)).to(self.device)
        self.encoding_offset = torch.zeros((1, self.model_encoder.latent_dim)).to(self.device)
        
        self.gen_seq = torch.zeros(self.joint_dim).repeat(self.seq_window_length, self.joint_count, 1).to(self.device)

        self.gen_seq_window = None
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
        self.seq_update_index = 0
        
    def _calc_pos_normalisation(self):
         
         # calculate pose normalisation values
         # TODO: this value should have been stored during training and just retrieved here
         
         orig_sequence_all = np.concatenate(self.orig_sequences, axis=0)
 
         pose_mean = np.mean(orig_sequence_all, axis=0).flatten()
         pose_std = np.std(orig_sequence_all, axis=0).flatten()
         
         self.pose_mean = torch.tensor(pose_mean).reshape(1, 1, -1).to(self.device)
         self.pose_std = torch.tensor(pose_std).reshape(1, 1, -1).to(self.device)
        
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

        pred_pose = pred_pose.reshape((1, self.joint_count, self.joint_dim))

        self.synth_pose_wpos = pred_pose.detach().cpu().numpy()
        
        self.seq_update_index += 1
        
        if self.seq_update_index >= self.seq_window_offset:
            
            #print("_gen")
            self._gen()
            self._blend()

            self.seq_update_index = 0
        

    def _gen(self):
         
        # encode orig seq window 1 and orig seq window 2
        
        #print("orig seq1 excerpt from ", self.orig_seq_frame_index1, " to ", (self.orig_seq_frame_index1 + self.seq_window_length) )
        
        orig_seq1_window = self.orig_seq1[self.orig_seq1_frame_index:self.orig_seq1_frame_index + self.seq_window_length, ...]
        orig_seq1_window = torch.from_numpy(orig_seq1_window).to(self.device)
        
        orig_seq1_window = orig_seq1_window.reshape(1, -1, self.pose_dim)
        orig_seq1_window_norm = (orig_seq1_window - self.pose_mean ) / self.pose_std
        orig_seq1_window_norm = torch.nan_to_num(orig_seq1_window_norm)

        #print("orig seq2 excerpt from ", self.orig_seq_frame_index2, " to ", (self.orig_seq_frame_index2 + self.seq_window_length) )

        orig_seq2_window = self.orig_seq2[self.orig_seq2_frame_index:self.orig_seq2_frame_index + self.seq_window_length, ...]
        orig_seq2_window = torch.from_numpy(orig_seq2_window).to(self.device)
        
        orig_seq2_window = orig_seq2_window.reshape(1, -1, self.pose_dim)
        orig_seq2_window_norm = (orig_seq2_window - self.pose_mean ) / self.pose_std
        orig_seq2_window_norm = torch.nan_to_num(orig_seq2_window_norm)
        
        with torch.no_grad():
            encoding1 = self.model_encoder(orig_seq1_window_norm)
            encoding2 = self.model_encoder(orig_seq2_window_norm)

        # mix encoding 1 and encoding 2
        encoding = encoding1 * (1.0 - self.encoding_mix) + encoding2 * self.encoding_mix
        
        # add encoding offset
        encoding += self.encoding_offset 
        
        # decode encoding
        with torch.no_grad():
            gen_seq_window_norm = self.model_decoder(encoding)
            
        self.gen_seq_window = gen_seq_window_norm * self.pose_std + self.pose_mean
        
        self.gen_seq_window = self.gen_seq_window.reshape(self.seq_window_length, self.joint_count, self.joint_dim)
        
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
        
        # roll seq window
        self.gen_seq = torch.roll(self.gen_seq, -self.seq_window_offset, 0)    
        
        # blend overlap region between gen_seq and gen_seq_window
        #blend_slope = torch.linspace(0.0, ((self.seq_window_overlap - 1) / self.seq_window_overlap), self.seq_window_overlap).unsqueeze(1).repeat(1, self.joint_count).to(self.device)
        
        blend_slope = torch.linspace(0.0, 1.0, self.seq_window_overlap).reshape(-1, 1, 1).repeat(1, self.joint_count, 1).to(self.device)

        blend_seq = self.gen_seq[:self.seq_window_overlap] * (1.0 - blend_slope) + self.gen_seq_window[:self.seq_window_overlap] * blend_slope

        self.gen_seq[:self.seq_window_overlap]

        self.gen_seq[:self.seq_window_overlap] = blend_seq
        self.gen_seq[self.seq_window_overlap:] = torch.clone(self.gen_seq_window[self.seq_window_overlap:])
