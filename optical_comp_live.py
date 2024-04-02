import cv2 as cv
import numpy as np

def optical_comp(frame2,prvs):
    subsample = 3
    frame2 = cv.GaussianBlur(frame2, (5, 5), 0)
    (h, w) = frame2.shape[:2]
    frame2 = cv.resize(frame2, (int(w / subsample), int(h / subsample)))
    next2 = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    horz = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
    vert = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')
    hor = cv.resize(horz, (w, h))
    ver = cv.resize(vert, (w, h))
#    cv.imshow("frame",hor)
#    cv.waitKey(2)
    #cap.release()
    #cv.destroyAllWindows()
    #print("ver", ver)
    return hor,ver

# print("here you go")
img = cv.imread("crop1.jpg")
#img = np.array(img)
print(img.shape)
hor, ver = optical_comp(img)

print("ver",ver)