

config = {
    "mocap_data": None,
    "batch_size: ": 32,
    "excerpt_offset": 32,
    "synthesis": None
          }

class MotionMap():

    def __init__(self, config):
        
        self.mocap_data = config["mocap_data"]
        self.batch_size = config["batch_size"]
        self.excerpt_offset = config["excerpt_offset"]
        self.synthesis = config["synthesis"]