import numpy as np
import cv2

from PIL import Image

# read the video
video = cv2.VideoCapture('C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST12\\124.cine')

#show basic staff about the video
def BasicVideoStats():
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = video.get(cv2.cv2.CAP_PROP_POS_MSEC)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f'The total number of frames is:', frames, 'frames', ', the total duration of the video is:', duration,
      'msec and the FPS of the video is:', fps)


# show video --- to stop showing you have to press the letter q
def PlayVideo():
    while (True):
        ret, frame = video.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

#Capture images per 40 frame
def VideoSplitter():
    video_images = 'C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST12\\Test124split'
    frameFrequency=10
# iterate all frames
    total_frame = 0
    id = 0
    while True:
        ret, frame = video.read()
        if ret is False:
            break
        total_frame += 1
        if total_frame % frameFrequency == 0:
            id += 1
            image_name = video_images + str(id) + '.jpg'
            cv2.imwrite(image_name, frame)
            print(image_name)
    video.release()

#shows and performs tasks on the first frame
def FirstImage():
    image = cv2.imread('C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\VideoImages80.jpg')
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CropImage():
    img = Image.open("C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST1\\VideoImages1.jpg")
    width, height = img.size
    print(f'width, height =', img.size)
    area = (400, 400, 600, 600)
    cropped_img = img.crop(area)
    cropped_img.show()
    width, height = cropped_img.size
    print(f'cropped_img width, height =', cropped_img.size)

def FixedColors():
    image = cv2.imread("C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST5\\Test2split15.jpg",1)
    s=(1280,800)
    new_image = np.zeros(s)
    a = 1.0 # Simple contrast control
    b = 50    # Simple brightness control
    new_image = cv2.convertScaleAbs(image, alpha= a , beta= b)
    # Initialize values
    #print(' Basic Linear Transforms ')
    #print('-------------------------')
    #try:
    #    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #    beta = int(input('* Enter the beta value [0-100]: '))
    #      except ValueError:
    #    print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    #for y in range(image.shape[0]):
    #    for x in range(image.shape[1]):
    #        for c in range(image.shape[2]):
    #            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    cv2.imshow('Original Image', image)
    cv2.imshow('New Image', new_image)
    # Wait until user press some key
    cv2.waitKey()



#Detecting Circles in Images using OpenCV
def circles():
    image = cv2.imread('C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\VideoImages3.jpg')
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0,255,0,4))
        cv2.rectangle(output, (x - 5, y-5), (x + 5, y + 5), (0, 128, 255))
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#def BinaryImg():
    # Read image
   # src = cv2.imread("C:\\Users\\Dino\\PycharmProjects\\IMG_PROC\\TEST11\\Test11split1.jpg", cv2.IMREAD_GRAYSCALE)
    # Set threshold and maxValue
    #thresh = 100
    #maxValue = 250
    # Basic threshold example
    #th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
    #cv2.imshow('New Image', dst)
    #print(dst)
    #pixel = 0
    #for i in dst[1,:]

    # Wait until user press some key
    #cv2.waitKey()

def PXtoUM():
    PX = int(input('* Enter how many pixels: '))
    UM1 = 100 * PX
    UM = UM1 / 43
    print(f'the ', PX, ' is ', round(UM), 'micrometers')

