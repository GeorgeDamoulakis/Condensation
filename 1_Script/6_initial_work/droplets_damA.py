import cv2
import numpy as np
from matplotlib import pyplot as plt 

#C:/Python37/python droplets_dam.py

def nothing(x):
    pass

def myBalls (img, minDist = 20 ,param1=50,param2=30,minRadius=0,maxRadius=0):
    img2 = img.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,minDist,param1,param2,minRadius,maxRadius)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img2,(i[0],i[1]),2,(0,0,255),3)
    return img2

def CC (img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return labeled_img, nlabels
 
 
img = cv2.imread('C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST5\\Test2split1.jpg')
cv2.namedWindow('image')

cv2.createTrackbar('T0','image',0,5,nothing)
cv2.createTrackbar('T1','image',50,55,nothing)
cv2.createTrackbar('T2','image',50,55,nothing)
cv2.createTrackbar('T3','image',20,22,nothing)
cv2.createTrackbar('T4','image',18,20,nothing)
cv2.createTrackbar('T5','image',20,23,nothing)

#img_circles = myBalls (img, minDist = 20 ,param1=50,param2=30,minRadius=0,maxRadius=0)
while(1):

    blurIter = cv2.getTrackbarPos('T0','image')
    minDist = cv2.getTrackbarPos('T1','image')
    param1 = cv2.getTrackbarPos('T2','image')
    param2 = cv2.getTrackbarPos('T3','image')
    minRadius = cv2.getTrackbarPos('T4','image')
    maxRadius = cv2.getTrackbarPos('T5','image')
    
    img2 = img.copy()
    for i in range (0,blurIter):
        img2 = cv2.GaussianBlur(img,(7,7),0)    
    img_circles = myBalls (img2, minDist, param1, param2, minRadius, maxRadius)
    
    small1 = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    small2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 
    small3 = cv2.resize(img_circles, (0,0), fx=0.5, fy=0.5) 
    
    result = np.hstack((small1, small2, small3))
    
    cv2.imshow('image',result)
      
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

img = cv2.imread('C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST5\\Test2split1.jpg',0)
cv2.namedWindow('image')

cv2.createTrackbar('T1','image',5,45,nothing)
cv2.createTrackbar('T2','image',2,9,nothing)
cv2.createTrackbar('T3','image',0,255,nothing)
cv2.createTrackbar('T4','image',0,255,nothing)
cv2.createTrackbar('T5','image',0,4,nothing)

while(1):
    current1 = cv2.getTrackbarPos('T1','image')
    if current1 % 2 == 0:
        current1 +=1
    current2 = cv2.getTrackbarPos('T2','image')
    current3 = cv2.getTrackbarPos('T3','image')
    current4 = cv2.getTrackbarPos('T4','image')
    current5 = cv2.getTrackbarPos('T5','image')
  
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,current1,current2)
    canny = cv2.Canny(thresh,current3,current4)
    sobelx = cv2.Sobel(thresh,cv2.CV_64F,1,0,ksize=5)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)
    abs_sobely = np.absolute(sobely)
    edgesSobel = abs_sobelx + abs_sobely
    kernel = np.ones((9,9),np.uint8)
    erosionSobel = cv2.erode(edgesSobel,kernel,iterations = current5)
    dilationSobel = cv2.dilate(erosionSobel,kernel,iterations = current5)
    edgesSobel = edgesSobel.astype(np.uint8)
    erosionSobel = erosionSobel.astype(np.uint8)
    dilationSobel = dilationSobel.astype(np.uint8)
    components1, labels1 = CC(canny)
    components2, labels2 = CC(dilationSobel)
    print(labels1, labels2)
    
    small1 = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    small2 = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5) 
    small3 = cv2.resize(canny, (0,0), fx=0.5, fy=0.5) 
    small4 = cv2.resize(edgesSobel, (0,0), fx=0.5, fy=0.5) 
    small5 = cv2.resize(erosionSobel, (0,0), fx=0.5, fy=0.5) 
    small6 = cv2.resize(dilationSobel, (0,0), fx=0.5, fy=0.5) 
    small7 = cv2.resize(components1, (0,0), fx=0.5, fy=0.5) 
    small8 = cv2.resize(components2, (0,0), fx=0.5, fy=0.5) 
    
    small1 = cv2.cvtColor(small1, cv2.COLOR_GRAY2BGR)
    small2 = cv2.cvtColor(small2, cv2.COLOR_GRAY2BGR)
    small3 = cv2.cvtColor(small3, cv2.COLOR_GRAY2BGR)
    small4 = cv2.cvtColor(small4, cv2.COLOR_GRAY2BGR)
    small5 = cv2.cvtColor(small5, cv2.COLOR_GRAY2BGR)
    small6 = cv2.cvtColor(small6, cv2.COLOR_GRAY2BGR)
    
    result = np.vstack((np.hstack((small1,small2,small3, small4)),np.hstack((small5,small6,small7, small8))))
    cv2.imshow('image',result)  
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

