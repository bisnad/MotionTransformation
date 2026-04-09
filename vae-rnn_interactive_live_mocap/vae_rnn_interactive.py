
import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg

import os, sys, time, subprocess
import numpy as np
import math
import pickle

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, qfix
from common.quaternion_np import slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap and Training Settings
"""

"""
# Example: ZED Mocap Recording
mocap_file_path = "E:/Data/mocap/Daniel/Zed/fbx/"
mocap_files = ["daniel_fooling_around.fbx", "daniel_fooling_around2.fbx", "daniel_zed_various.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 30

encoder_weights_file = "../vae-rnn/results_Zed_daniel_foolingaround_various/weights/encoder_weights_epoch_600"
decoder_weights_file = "../vae-rnn/results_Zed_daniel_foolingaround_various/weights/decoder_weights_epoch_600"
"""

# Example: ZED Mocap Recording
mocap_file_path = "../../../Data/Mocap/Zed/Daniel/Solos/fbx_30hz"
mocap_files = ["daniel_zed_solo1.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 30

encoder_weights_file = "../../../Data/Models/MotionTransformation/vae-rnn/results_Zed_Daniel_Solo1/weights/encoder_weights_epoch_600"
decoder_weights_file = "../../../Data/Models/MotionTransformation/vae-rnn/results_Zed_Daniel_Solo1/weights/decoder_weights_epoch_600"


"""
Model Settings
"""

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [ 512 ]


"""
OSC Settings
"""

osc_send_ip = "127.0.0.1"
osc_send_port = 9004

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007





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


all_pose_sequences = []

for mocap_data in all_mocap_data:
    
    pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences.append(pose_sequence)

joint_count = all_pose_sequences[0].shape[1]
joint_dim = all_pose_sequences[0].shape[2]
pose_dim = joint_count * joint_dim

"""
Load Model
"""

motion_model.config["seq_length"] = sequence_length
motion_model.config["data_dim"] = pose_dim
motion_model.config["latent_dim"] = latent_dim
motion_model.config["rnn_layer_count"] = ae_rnn_layer_count
motion_model.config["rnn_layer_size"] = ae_rnn_layer_size
motion_model.config["dense_layer_sizes"] = ae_dense_layer_sizes
motion_model.config["device"] = device
motion_model.config["weights_path"] = [encoder_weights_file, decoder_weights_file]

encoder, decoder = motion_model.createModels(motion_model.config) 

"""
Setup Motion Synthesis
"""

sequence_overlap = sequence_length // 4 * 3

motion_synthesis.config["skeleton"] = all_mocap_data[0]["skeleton"]
motion_synthesis.config["model_encoder"] = encoder
motion_synthesis.config["model_decoder"] = decoder
motion_synthesis.config["device"] = device
motion_synthesis.config["seq_window_length"] = sequence_length
motion_synthesis.config["seq_window_overlap"] = sequence_overlap
motion_synthesis.config["orig_sequences"] = all_pose_sequences
motion_synthesis.config["orig_seq_index"] = 0

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

"""
OSC Sender
"""

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
"""

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["latent_dim"] = latent_dim
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
