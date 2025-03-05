from time import time
import cv2
from scipy.ndimage.filters import convolve as filter2, gaussian_filter
import numpy as np

# Window matrices for computing derivatives
windowAvg = np.array([[1/12, 1/6, 1/12],
                     [1/6,    0, 1/6],
                     [1/12, 1/6, 1/12]], float)

windowX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

windowY = np.array([[-1, -1],
                    [ 1, 1]]) * .25  # kernel for computing d/dy

windowT = np.ones((2, 2)) * .25


class HornSchunck:
    def process(self, frame1, frame2, alpha=0.001, NumOfIter=8):
        """
        Compute optical flow using Horn-Schunck method
        
        Parameters:
        frame1: frame at t=0
        frame2: frame at t=1
        alpha: regularization constant
        NumOfIter: number of iteration
        
        Returns:
        H, V: Horizontal and vertical components of optical flow
        M: Magnitude of flow vectors
        """
        # Convert to float32 if needed
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)

        # Initialize flow vectors
        H = np.zeros([frame1.shape[0], frame1.shape[1]])
        V = np.zeros([frame1.shape[0], frame1.shape[1]])

        # Estimate derivatives
        [fx, fy, ft] = self.derivatives(frame1, frame2)

        # Iterative refinement to reduce error
        for i in range(NumOfIter):
            # Average the flow vectors
            hAvg = cv2.filter2D(H, -1, windowAvg)
            vAvg = cv2.filter2D(V, -1, windowAvg)
            
            # Common part of update step
            top = fx*hAvg + fy*vAvg + ft
            down = alpha**2 + fx**2 + fy**2
            der = top/down

            # Iterative step
            H = hAvg - fx * der
            V = vAvg - fy * der

        # Calculate magnitude
        M = pow(pow(H, 2) + pow(V, 2), 0.5)

        return H, V, M

    def derivatives(self, frame1, frame2):
        """Calculate spatial and temporal derivatives"""
        fx = filter2(frame1, windowX) + filter2(frame2, windowX)
        fy = filter2(frame1, windowY) + filter2(frame2, windowY)
        ft = filter2(frame1, windowT) + filter2(frame2, -windowT)
        return fx, fy, ft

    def draw_vectors_hs(self, im1, im2, step=10):
        """Draw optical flow vectors on image for visualization"""
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        U, V, M = self.process(im1_gray, im2_gray)

        rows, cols = im2_gray.shape
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                x = int(U[i, j]*2)
                y = int(V[i, j]*2)
                cv2.arrowedLine(im2, (j, i), (j + x, i + y), (255, 0, 0))
        return im2


if __name__ == "__main__":
    # Example usage
    cap = cv2.VideoCapture("2.mkv")
    i = 0
    while(i < 200):  # Skip first 200 frames
        ret, frame = cap.read()
        i += 1
        
    ret, old = cap.read()
    hs = HornSchunck()
    
    while(True):
        ret, new = cap.read()
        ret, new = cap.read()
        if not ret:
            break

        ret = hs.draw_vectors_hs(old, new)
        old = new
        cv2.imshow("frame", ret)
        cv2.waitKey(1)

    cv2.destroyAllWindows()