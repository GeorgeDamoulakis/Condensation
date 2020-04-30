import cv2
import numpy as np
from matplotlib import pyplot as plt 

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
    return labeled_img, nlabels, labels, stats, centroids

img = cv2.imread('/Users/georgedamoulakis/PycharmProjects/a1/split.jpg',0)
cv2.namedWindow('image')


ret, thresh= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8) #find gaussian kenrnel
blur = cv2.blur(thresh, (2,2))
erosion = cv2.erode(blur,kernel,iterations = 2) #--- erosion is input for dilation
#dilation = cv2.dilate(erosion,kernel,iterations = 0)

components, nlabels, labels, stats, centroids = CC(erosion)
print(f' There are ', nlabels, '  different objects.')
print(f' with the following labels: ' ,labels)
    
small1 = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
small2 = cv2.resize(thresh, (0,0), fx=0.5, fy=0.5) 
small3 = cv2.resize(components, (0,0), fx=0.5, fy=0.5) 
    
small1 = cv2.cvtColor(small1, cv2.COLOR_GRAY2BGR)
small2 = cv2.cvtColor(small2, cv2.COLOR_GRAY2BGR)
    
result = np.hstack((small1,small2,small3))
cv2.imshow('image',result)  

#gray = img.copy()
#output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.4, 100)
#circles = np.round(circles[0, :]).astype("int")
#for (x, y, r) in circles:
  #  cv2.circle(output, (x, y), r, (0,255,0,4))
 #   cv2.rectangle(output, (x - 5, y-5), (x + 5, y + 5), (0, 128, 255))
#cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


#print(f'centroids', centroids)

cv2.waitKey(0) 
cv2.destroyAllWindows()



