
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
import json
import pickle
from time import sleep

from common import utils
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

"""
mocap_config_file = "configs/Halpe26_config.json" 
mocap_file_path = "mocap/"
mocap_files = ["Mocap_class_0_time_1723812067.0081663.pkl", "Mocap_class_0_time_1723812067.0081663.pkl"]
mocap_valid_frame_ranges = [ [ [ 0, 9390 ] ] ]
mocap_sensor_ids = ["/mocap/0/joint/pos2d_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Hip"
mocap_fps = 30
mocap_joint_dim = 2
"""

mocap_config_file = "configs/Human36M_config.json" 
mocap_file_path = "mocap/"
mocap_files = ["Mocap_class_0_time_1724065746.5842216.pkl", "Mocap_class_0_time_1724065746.5842216.pkl"]
mocap_valid_frame_ranges = [ [ [ 0, 9390 ] ] ]
mocap_sensor_ids = ["/mocap/0/joint/pos3d_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Bottom_Torso"
mocap_fps = 30
mocap_joint_dim = 3

mocap_seq_window_length = 64
mocap_seq_window_overlap = 48


"""
Load Mocap Data
"""

with open(mocap_config_file) as f:
    mocap_config = json.load(f)


def config_to_skeletondata(mocap_config):
    
    skeleton_data = {}
    skeleton_data["joints"] = mocap_config["jointNames"]
    skeleton_data["root"] = skeleton_data["joints"][0]
    skeleton_data["parents"] = mocap_config["jointParents"]
    skeleton_data["children"] = mocap_config["jointChildren"]
    
    return skeleton_data

def recording_to_motiondata(mocap_recording, skeleton_data, mocap_sensor_ids):
    
    joint_count = len(skeleton_data["joints"])
    
    # gather sensor values
    motion_data = {}
    
    sensor_ids = mocap_recording["sensor_ids"]
    sensor_values = mocap_recording["sensor_values"]
    
    for sensor_id in mocap_sensor_ids:

        #print("sensor_id ", sensor_id)
        motion_data[sensor_id]  = [ sensor_values [vI] for vI in range(len(sensor_values)) if sensor_ids[vI].endswith(sensor_id) ]
        motion_data[sensor_id] = np.array(motion_data[sensor_id], dtype=np.float32)
        motion_data[sensor_id] = np.reshape(motion_data[sensor_id], (motion_data[sensor_id].shape[0], joint_count, -1))

    return motion_data

skeleton_data = config_to_skeletondata(mocap_config)

all_motion_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    with open(mocap_file_path + "/" + mocap_file, "rb") as f:
        mocap_recording = pickle.load(f)
        
        motion_data = recording_to_motiondata(mocap_recording, skeleton_data, mocap_sensor_ids)
        
        all_motion_data.append(motion_data)
        
# retrieve mocap properties

joint_count = len(skeleton_data["joints"])
joint_dim = mocap_joint_dim
pose_dim = joint_count * joint_dim

# set root position to zero
mocap_root_joint_index = skeleton_data["joints"].index(mocap_root_joint_name)

for motion_data in all_motion_data:

    if joint_dim == 3:
        joint_pos = motion_data["/mocap/0/joint/pos3d_world"]
    else:
        joint_pos = motion_data["/mocap/0/joint/pos2d_world"]
        
    root_pos = joint_pos[:, mocap_root_joint_index:mocap_root_joint_index+1, :]
    
    joint_pos_root_zero = joint_pos - root_pos
    
    motion_data["/mocap/0/joint/pos_root_zero"] = joint_pos_root_zero

all_pose_sequences = []
for motion_data in all_motion_data:
    pose_sequence = motion_data["/mocap/0/joint/pos_root_zero"]
    all_pose_sequences.append(pose_sequence)

"""
Load Model
"""

motion_model.config["seq_length"] = mocap_seq_window_length
motion_model.config["data_dim"] = pose_dim
motion_model.config["latent_dim"] = 32
motion_model.config["rnn_layer_count"] = 2
motion_model.config["rnn_layer_size"] = 512
motion_model.config["dense_layer_sizes"] = [512]
motion_model.config["device"] = device
#motion_model.config["weights_path"] = ["../vae-rnn/results_MMPose2D_HannahMartin/weights/encoder_weights_epoch_600", "../vae-rnn/results_MMPose2D_HannahMartin/weights/decoder_weights_epoch_600"]
motion_model.config["weights_path"] = ["../vae-rnn/results_MMPose3D_HannahMartin/weights/encoder_weights_epoch_600", "../vae-rnn/results_MMPose3D_HannahMartin/weights/decoder_weights_epoch_600"]


encoder, decoder = motion_model.createModels(motion_model.config) 

"""
Setup Motion Synthesis
"""

motion_synthesis.config["skeleton"] = skeleton_data
motion_synthesis.config["model_encoder"] = encoder
motion_synthesis.config["model_decoder"] = decoder
motion_synthesis.config["device"] = device
motion_synthesis.config["seq_window_length"] = mocap_seq_window_length
motion_synthesis.config["seq_window_overlap"] = mocap_seq_window_overlap
motion_synthesis.config["orig_sequences"] = all_pose_sequences
motion_synthesis.config["orig_seq1_index"] = 0
motion_synthesis.config["orig_seq2_index"] = 1

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004

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
motion_control.config["latent_dim"] = 32
motion_control.config["ip"] = "0.0.0.0"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)


"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
