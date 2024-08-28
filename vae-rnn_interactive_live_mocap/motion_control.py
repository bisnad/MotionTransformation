import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "latent_dim": 32,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.latent_dim = config["latent_dim"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/joint/rot_local", self.setLiveSeq)
        self.dispatcher.map("/mocap/0/joint/rot_local", self.setLiveSeq)
        self.dispatcher.map("/mocap/seqindex", self.setSeqIndex)
        
        self.dispatcher.map("/mocap/seqframeindex", self.setSeqFrameIndex)
        self.dispatcher.map("/mocap/seqframerange", self.setSeqFrameRange)
        self.dispatcher.map("/mocap/seqframeincr", self.setSeqFrameIncrement)    
        
        self.dispatcher.map("/synth/encodingmix", self.setEncodingMix)
        self.dispatcher.map("/synth/encodingoffset", self.setEncodingOffset)    
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setLiveSeq(self, address, *args):
        
        osc_address = address
        osc_values = args
        
        rot_local = np.asarray(osc_values, dtype=np.float32)
        
        self.synthesis.setLiveSeq(rot_local)
        
    def setSeqIndex(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeqIndex(index)
        
    def setSeqFrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeqFrameIndex(index)
        
    def setSeqFrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeqFrameRange(startFrame, endFrame)
        
    def setSeqFrameIncrement(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setSeqFrameIncrement(incr)

        
    def setEncodingMix(self, address, *args):
        
        mix = []
        
        for d in range(self.latent_dim):
        
            mix.append(args[d])
        
        self.synthesis.setEncodingMix(mix)
        
    def setEncodingOffset(self, address, *args):
        
        offset = []
        
        for d in range(self.latent_dim):
        
            offset.append(args[d])
        
        self.synthesis.setEncodingOffset(offset)

