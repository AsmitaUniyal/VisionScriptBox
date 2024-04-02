import cv2
import numpy as np


# Define the codec and create VideoWriter object
def vidCapture(counter):
    cap= cv2.VideoCapture(0)
    cv2.namedWindow("test")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_name = "vid_{}.avi".format(counter)
    out = cv2.VideoWriter(vid_name,fourcc, 20.0, (640,480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame,0)
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def main():
    img_counter = 0
    print(img_counter)
    vidCapture(img_counter)
    k = cv2.waitKey(1)
    while(k%256 != 27):
        inp= input("Do you want to continue recording?: Press y :: ")
        if(inp=='y'):
            img_counter = img_counter+1
            vidCapture(img_counter)       
        else:
            break 

if __name__ == "__main__":
    main()