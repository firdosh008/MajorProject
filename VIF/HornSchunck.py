from time import sleep, time
import cv2
from scipy.ndimage.filters import convolve as filter2, gaussian_filter
import numpy as np

#
windowAvg =np.array([[1 / 12, 1 / 6, 1 / 12],
                     [1/6,    0, 1/6],
                     [1/12, 1/6, 1/12]], float)

windowX = np.array([[-1, 1],
                    [-1, 1]]) * .25 #kernel for computing d/dx

# windowX = np.array([[-1, 0,1],
#                     [-1, 0,1],
#                     [-1,0,1]]) * .25 #kernel for computing d/dx
#

windowY = np.array([[-1, -1],
                    [ 1, 1]]) * .25 #kernel for computing d/dy
#
# windowY = np.array([[-1, -1,-1],
#                     [0, 0,0],
#                     [1,1,1]]) * .25 #kernel for computing d/dx

windowT = np.ones((2, 2)) * .25




class HornSchunck:


    def process(self,frame1, frame2, alpha=0.001, NumOfIter=8):
        """
        frame1: frame at t=0
        frame2: frame at t=1
        alpha: regularization constant
        NumOfIter: number of iteration
        """
        #if the frame is integers then we need to convert it to floats
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)

        # making the shape of horizontal and vertical change
        # Set initial value for the flow vectors
        H = np.zeros([frame1.shape[0], frame1.shape[1]])
        V = np.zeros([frame1.shape[0], frame1.shape[1]])

        # Estimate derivatives
        [fx, fy, ft] = self.derivatives(frame1, frame2)


        # Iteration to reduce error
        for i in range(NumOfIter):
            # avrageing the flow vectors
            hAvg = cv2.filter2D(H, -1, windowAvg)
            vAvg = cv2.filter2D(V, -1, windowAvg)
            # common part of update step
            top = fx*hAvg + fy*vAvg + ft
            down = alpha**2 + fx**2 + fy**2
            der = top/down

            # iterative step
            H = hAvg - fx * der
            V = vAvg - fy * der

        M = pow(pow(H, 2) + pow(V, 2), 0.5)


        return H,V, M


    def derivatives(self,frame1, frame2):
        t = time()
        fx = filter2(frame1, windowX) + filter2(frame2, windowX)
        fy = filter2(frame1, windowY) + filter2(frame2, windowY)
       # ft = im2 - im1
        ft = filter2(frame1, windowT) + filter2(frame2, -windowT)
        # print(time() - t)
        return fx,fy,ft

    def draw_vectors_hs(self,im1, im2, step = 10):
        print("drawing vectors")
        t = time()
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # im1_gray = cv2.GaussianBlur(im1_gray,(5,5),0)
        # im2_gray = cv2.GaussianBlur(im2_gray,(5,5),0)


        U, V, M = self.process(im1_gray, im2_gray)

        print(time() - t)
        rows, cols = im2_gray.shape
        # print(rows, cols, range(0, rows, step))
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                x = int(U[i, j]*2)
                y = int(V[i, j] *2)
                cv2.arrowedLine(im2, (j, i), (j + x, i + y), (255, 0, 0))
        return im2



if __name__ == "__main__":
    cap = cv2.VideoCapture("2.mkv")
    i =0
    while(i < 200):
        ret, frame = cap.read()  # get first frame
        i+=1
    ret,old = cap.read()
    hs = HornSchunck()
    while(True):
        ret, new = cap.read()
        ret, new = cap.read()
        if not ret:
            break


        # kernel = np.ones((3, 3), np.float32) / 9
        # new  = cv2.filter2D(new, -1, kernel)
        ret = hs.draw_vectors_hs(old,new)
        old = new
        cv2.imshow("frame", ret)
        cv2.waitKey(1)

    cv2.destroyAllWindows()






























def ccw(A,B,C):
	return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# buscamos si hay vectores de flujo optico que se intersectan, a fuerza bruta
def check_intersection(lines, frame):
    print("vericando intersecciones entre " + str(len(lines)))
    print(lines)
    for i in range(0, len(lines)):
        for j in range(0, len(lines)):
            #L1 = line([lines[i][0], lines[i][1]], [lines[i][2], lines[i][3]])
            #L2 = line([lines[j][0], lines[j][1]], [lines[j][2], lines[j][3]])
            #print([lines[i][0], lines[i][1]], [lines[i][2], lines[i][3]], [lines[j][0], lines[j][1]], [lines[j][2], lines[j][3]])

            a1 = (lines[i][0], lines[i][1])
            a2 = (lines[i][2], lines[i][3])
            b1 = (lines[j][0], lines[j][1])
            b2 = (lines[j][2], lines[j][3])

            if i != j:
                # R = intersection(L1, L2)
                R = intersect(a1, a2, b1, b2)
                rows, cols, ch = frame.shape

                if R:
                    print("Intersection detected", a1, a2, b1,b2)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (cols - 1, rows - 1), (0, 0, 255), -1)
                    opacity = 0.4
                    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
