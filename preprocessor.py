import numpy as np

class Preprocessor:
    
    def __init__(self, n_frames, n_channels, width, height, scaled_width, scaled_height):
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.width = width
        self.height = height
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
    
    def np_array(self, image_bytes):
        
        int_array = np.frombuffer(image_bytes, dtype=np.int32)
        int_array = int_array.reshape((self.n_frames, self.n_channels, self.scaled_width, self.scaled_height))
        return int_array
        
        
    def resize_frames(self, frames):
        resized = np.ones((self.n_frames, self.n_channels, 84, 84))
        for f in range(self.n_frames):
            for ch in range(self.n_channels):
                resized[f][ch] = self.resize_frame(frames[f][ch])

        return torch.FloatTensor(resized)

    def resize_frame(self, frame):
        """
        Takes in a 2D list (240, 256) and resizes it to 84x84
        """
        # take of the the top 20 rows of pixels to get rid of the score, timer etc
      

        # resize the frame
        image_resized = resize(frame[20:240], (84, 84), anti_aliasing=True)
        return image_resized
    
    def process(self, image_bytes):
        
        int_array = self.np_array(image_bytes)
        return self.resize_frames(int_array)
        