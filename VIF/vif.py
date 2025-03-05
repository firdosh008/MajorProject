import pickle
import os
import numpy as np
import cv2
import math

from VIF.HornSchunck import HornSchunck


class VIF:
    """
    Visual Information Fidelity (VIF) crash detection implementation
    """
    
    def __init__(self):
        """Initialize VIF model and parameters"""
        self.subSampling = 3
        self.rows = 100
        self.cols = 134
        self.hs = HornSchunck()
        
        # Load trained model
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model-svm1.sav")
        self.clf = pickle.load(open(model_path, 'rb'))
        
        # Counters for tracking
        self.no_crash = 0
        self.crash = 0

    def createBlockHist(self, flow, N, M):
        """
        Create histogram features from blocks of optical flow
        
        Args:
            flow: Optical flow magnitude
            N, M: Number of blocks in height and width
            
        Returns:
            Feature vector from histograms
        """
        height, width = flow.shape
        B_height = int(math.floor((height - 11) / N))
        B_width = int(math.floor((width - 11) / M))

        frame_hist = []

        for y in np.arange(6, height - B_height - 5, B_height):
            for x in np.arange(6, width - B_width - 5, B_width):
                block_hist = self.createHist(flow[y:y + B_height - 1, x:x + B_width - 1])
                frame_hist.append(block_hist)

        return np.array(frame_hist).flatten()

    def createHist(self, mini_flow):
        """
        Create normalized histogram from flow values
        
        Args:
            mini_flow: Block of optical flow values
            
        Returns:
            Normalized histogram
        """
        H = np.histogram(mini_flow, np.arange(0, 1, 0.05))
        H = H[0]/float(np.sum(H[0]))
        return H

    def process(self, frames):
        """
        Process frames to extract VIF features
        
        Args:
            frames: List of video frames
            
        Returns:
            Feature vector for crash detection
        """
        # Initialize flow accumulator
        flow = np.zeros([self.rows, self.cols])
        index = 0
        N = 4  # Number of blocks in height
        M = 4  # Number of blocks in width
        shape = (self.cols, self.rows)

        # Process frames with subsampling
        for i in range(0, len(frames) - self.subSampling - 5, self.subSampling * 2):
            index += 1
            
            # Get three frames with spacing
            prevFrame = frames[i + self.subSampling]
            currFrame = frames[i + self.subSampling * 2]
            nextFrame = frames[i + self.subSampling * 3]

            # Resize frames to standard size
            prevFrame = cv2.resize(prevFrame, shape)
            currFrame = cv2.resize(currFrame, shape)
            nextFrame = cv2.resize(nextFrame, shape)

            # Calculate optical flow between consecutive frames
            u1, v1, m1 = self.hs.process(prevFrame, currFrame)
            u2, v2, m2 = self.hs.process(currFrame, nextFrame)

            # Detect significant changes in flow
            delta = abs(m1 - m2)
            flow = flow + (delta > np.mean(delta))

        # Normalize accumulated flow
        flow = flow.astype(float)
        if index > 0:
            flow = flow/index

        # Create feature vector from flow histograms
        feature_vec = self.createBlockHist(flow, N, M)

        return feature_vec