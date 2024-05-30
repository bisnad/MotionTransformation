
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
Mocap Settings
"""

"""
mocap_file_path = "../../../../../../Data/mocap/stocos/solos/"
mocap_files = ["Muriel_Take1.bvh",
               "Muriel_Take2.bvh",
               "Muriel_Take3.bvh",
               "Muriel_Take4.bvh",
               "Muriel_Take5.bvh",
               "Muriel_Take6.bvh"]

mocap_valid_frame_ranges = [ [ 0, 16709 ],
                            [ 0, 11540 ],
                            [ 0, 12373 ],
                            [ 0, 5006 ],
                            [ 0, 27628 ],
                            [ 0, 12380 ]]
"""

mocap_file_path = "../../../../../../Data/mocap/stocos/solos/"
mocap_files = ["Muriel_Take2.bvh", "Muriel_Take4.bvh"]
mocap_valid_frame_ranges = [ [ 0, 11540 ], [ 0, 5006 ] ]

mocap_seq_window_length = 64
mocap_seq_window_overlap = 48
mocap_fps = 50

"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    bvh_data = bvh_tools.load(mocap_file_path + "/" + mocap_file)
    mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
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

motion_model.config = {
    "seq_length": mocap_seq_window_length,
    "data_dim": pose_dim,
    "latent_dim": 32,
    "rnn_layer_count": 2,
    "rnn_layer_size": 512,
    "dense_layer_sizes": [512],
    "device": device,
    "weights_path": ["../aae-rnn/results_xsens_64/weights/encoder_weights_epoch_600", "../aae-rnn/results_xsens_64/weights/decoder_weights_epoch_600"]
    }

encoder, decoder = motion_model.createModels(motion_model.config) 

"""
Setup Motion Synthesis
"""

motion_synthesis.config = {
    "skeleton": all_mocap_data[0]["skeleton"],
    "model_encoder": encoder,
    "model_decoder": decoder,
    "device": device,
    "seq_window_length": mocap_seq_window_length,
    "seq_window_overlap": mocap_seq_window_overlap,
    "orig_sequences": all_pose_sequences,
    "orig_seq1_index": 0,
    "orig_seq2_index": 1
    }

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
motion_control.config["ip"] = "127.0.0.1"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)


"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
