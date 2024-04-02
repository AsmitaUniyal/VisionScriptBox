import numpy as np
import cv2
import tkinter

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

def cascade_fun(img,height,width,counter):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if np.shape(faces)[0] > 0:
        faces = np.reshape(faces[np.argmax(faces[:,3])],(1,4))
    print("faces",len(faces))
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        print("x cordinates: ",x,x+w)
        z= int((y+y+h)/2)
        print("Y will be: ",z )
        # cv2.line(img, (x,z),(x+w,z),(0,255,0),2)

        #drawing second rectangle
        w2= int((16/6)*w)
        h2= int(h+(h*2.8)+(0.1*h))
        if w2>width:
            w2=width
        elif h2>height:
            h2=height
        #out = cv2.VideoWriter("vid_2.avi",fourcc, 20.0, (h2,w2))
        
        mid1=int((x+x+w)/2)
#        cv2.line(img, (mid1,mid1),(int(mid1-(w2/2)),358),(0,255,0),2)
        p= int(mid1-(w2/2))
        q= int (y- 0.2*h)
        #cv2.rectangle(img,(p,q),(p+w2,q+h2),(255,0,0),2)
#        dst = np.zeros_like(img)
#        dst[p:p+w2,q:q+h2] = img[p:p+w2,q:q+h2]
#        cv2.imshow('dst',dst)
       # r = cv2.selectROI(img)
        #imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        img_name = "img_{}.jpg".format(counter)
        face_image = img[q:q+h2, p:p+w2] 
        cv2.imshow('dst',face_image)
        folder = "image_folder"
        image_name= folder+ img_name
        cv2.imwrite(image_name,face_image)   
        #out.write(face_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break
#        xFaceCenter=((x+w)-rect[0])/2
#        yFaceCenter=(rect[3]-rect[1])/2
#        print(x,y,w,h) # cordinates of the box
#        print("Width1: ",w)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
#        for (ex,ey,ew,eh) in eyes:
#            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img
 
    

 #for taking images from the camera 
def main(): 
    cap = cv2.VideoCapture(0)
    counter=0
    while(cap.isOpened()):
        try:
            return_value, image = cap.read()    
            #img = cv2.imread('opencv.jpg')
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            width = int(cap.get(3))   # float
            height = int(cap.get(4))
            print("Frame width,height :",height,width)
            #cv2.imshow('img',gray)
            counter= counter+1
            img= cascade_fun(image,height,width,counter)
            print("It's out")           
            #cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()                
                cv2.destroyAllWindows()
                
    #            #for taking downloaded images   
    #    img = cv2.imread('frame_0.jpg')
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    #    cv2.imshow('img',gray)
    #    height, width = gray.shape[:2]
    ##    lv_x = winfo_rootx()
    ##    lv_y = winfo_rooty()
    #    print("Frame width,height :",height,width)
    #    imag= cascade_fun(gray,height,width)
    #    
    #    cv2.imshow('img',imag)
    #    cv2.waitKey(0)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        cv2.destroyAllWindows()
        except (cv2.error,TypeError,IndexError):
            #cv2.imshow("img", img)0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                #out.release()
                #cv2.destroyAllWindows()
                cv2.destroyAllWindows()     
            
       
if __name__ == "__main__":
    main()







