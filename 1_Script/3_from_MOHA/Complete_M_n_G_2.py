import cv2
import numpy as np
import math
import time

################### start the clock ###################
start_time = time.time()

################### authentic image ###################
im_in_auth = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Condensation/2_Images/for_MOHA/Initial/0.jpg');
h, w = im_in_auth.shape[:2]
white = np.zeros([h + 300, w + 300, 3], dtype=np.uint8)
white.fill(255)
for i in range(1, h, 1):
    for j in range(1, w, 1):
        white[i + 150, j + 150] = im_in_auth[i, j]
im_in1 = white
Original = white.copy()

################### GAMMA IMAGE ###################
im_in = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Condensation/2_Images/for_MOHA/afterGamma/0.jpg');
h, w = im_in.shape[:2]
white = np.zeros([h + 300, w + 300, 3], dtype=np.uint8)
white.fill(255)
for i in range(1, h, 1):
    for j in range(1, w, 1):
        white[i + 150, j + 150] = im_in[i, j]
im_in = white
im_in = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)


def CountingCC(im_in):
    ################### thresh ##################
    th, im_th = cv2.threshold(im_in, 50, 150, cv2.THRESH_BINARY_INV);

    ################### define CC function ##################
    def CC(img):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0
        return labeled_img, nlabels, labels, stats, centroids

    ################### fixing the image ##################
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(im_th, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    components, nlabels, labels, stats, centroids = CC(dilation)

    ################### Droplet or Not##################
    d = 0
    droplet_counter = 0
    a = np.hsplit(stats, 5)
    b = np.hsplit(centroids, 2)
    horizontal = a[2]
    vertical = a[3]
    area = a[4]
    x_centr = b[0]
    y_centr = b[1]
    NEW_dimensions = np.zeros((nlabels, 6))
    Not_Droplet = np.empty(nlabels, dtype=object)
    for i in range(nlabels):
        d = ((horizontal[i] + vertical[i]) / 2)
        p = d * 3.14  # the perimeter
        circularity = 4 * (3.14) * ((area[i]) / (p ** 2))
        if circularity < 0.90:
            Not_Droplet[i] = "NOT a droplet"
        else:
            Not_Droplet[i] = "ok"
            droplet_counter = droplet_counter + 1

    ################### Dimensions ###################
    for row in range(nlabels):
        for column in range(8):
            if column == 0:
                NEW_dimensions[row, column] = (row + 1)
            elif column == 1:
                NEW_dimensions[row, column] = x_centr[row]
            elif column == 2:
                NEW_dimensions[row, column] = y_centr[row]
            elif column == 3:
                if horizontal[row] < 100:
                    NEW_dimensions[row, column] = horizontal[row] + 20
                else:
                    NEW_dimensions[row, column] = horizontal[row] + 40
            elif column == 4:
                if vertical[row] < 100:
                    NEW_dimensions[row, column] = vertical[row] + 20
                else:
                    NEW_dimensions[row, column] = vertical[row] + 40
            elif column == 5:
                NEW_dimensions[row, column] = ((NEW_dimensions[row][3]) + (NEW_dimensions[row][4])) * 3.14 * 0.25 * (
                        (NEW_dimensions[row][3]) + (NEW_dimensions[row][4]))

    ################### Draw Circles - Boxes - Numbers ###################
    i = 0
    XCENTER = []
    YCENTER = []
    r = []
    image = components
    out = image.copy()
    for row in range(1, nlabels, 1):
        for column in range(5):
            if Not_Droplet[row] == "ok":
                XCENTER.append((int(x_centr[row])))
                YCENTER.append((int(y_centr[row])))
                X = XCENTER[i]
                Y = YCENTER[i]
                cv2.rectangle(out, (int(X) - 3, int(Y) - 3), (int(X) + 3, int(Y) + 3), (0, 0, 0))
                r.append((math.sqrt(NEW_dimensions[row][5] * 0.31830988618) * 0.5))
                P = r[i]
                cv2.circle(out, (int(X), int(Y)), int(P), (255, 255, 0, 4))
                cv2.putText(out, ('%d' % (row)), (int(X), int(Y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                            2)
                i = i + 1

    cv2.putText(out, ('%d droplets' % droplet_counter), (5, 30), cv2.FONT_ITALIC, 1.2, (220, 220, 220), 2)

    return r, XCENTER, YCENTER, out, droplet_counter

r, x_centr, y_centr, output, droplet_counter = CountingCC(im_in)
U = int(len(r) / 5)

def AccurateDropletSize(U):
    R = np.zeros(U)
    X = np.zeros(U)
    Y = np.zeros(U)
    C = np.zeros(U)
    B = np.zeros(U)
    New_Cx = np.zeros(U)
    New_Cy = np.zeros(U)
    Radii = np.zeros(U)
    for i in range(0, U):
        R[i] = r[(i * 5) + 1]
        X[i] = x_centr[(i * 5) + 1]
        Y[i] = y_centr[(i * 5) + 1]
        C[i] = droplet_counter
        B[i] = 0

    ################### Hough Circle Detection ###################
    for t in range(0, U):
        # the actual CircleNO is i+1
        CircleNO = t
        if int(R[CircleNO]) < 10:
            RR = int(R[CircleNO]) + 15
        elif int(R[CircleNO]) < 70 and int(R[CircleNO]) > 10:
            RR = int(R[CircleNO]) + 20
        else:
            RR = int(R[CircleNO]) + 40
        x = int(X[CircleNO])
        y = int(Y[CircleNO])
        crop_img = im_in1[y - RR:y + RR, x - RR:x + RR]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        crop_img = cv2.GaussianBlur(crop_img, (5, 5), 0)
        cimg = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(crop_img, cv2.HOUGH_GRADIENT, 1.1, 1000, param1=20, param2=20, minRadius=1,
                                   maxRadius=600)
        circles = np.uint16(np.around(circles))
        j = 0
        RMAX = 0
        FR = np.zeros(len(circles[0]))

        for i in circles[0, :]:
            if len(circles[0]) >= 1:
                FR[j] = int(i[2])
                if FR[j] > RMAX:
                    RMAX = FR[j]
                    Radii[t] = RMAX
                    New_Cx[t] = x - RR + i[0]
                    New_Cy[t] = y - RR + i[1]
                    j = j + 1

    for i in range(0, len(Radii)):
        # draw the outer circle
        NCY = New_Cy[i]
        NCX = New_Cx[i]
        RDI = Radii[i]
        cv2.circle(im_in1, (int(NCX), int(NCY)), int(RDI), (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(im_in1, (int(NCX), int(NCY)), 2, (0, 0, 255), 2)

    ################### calculate background area  ###################
    TotalArea_Frame = int(im_in.shape[0] * im_in.shape[1])
    TotalArea_Droplets = 0
    for i in range(0, U):
        TotalArea_Droplets = (3.14 * Radii[i] * Radii[i]) + TotalArea_Droplets
    TotalArea_Background = TotalArea_Frame - TotalArea_Droplets
    for i in range(0, U):
        B[i] = TotalArea_Background

    ################### image show ###################
    # cv2.imshow("output", output)
    # im_in1 = cv2.resize(im_in1, (0, 0), fx=0.8, fy=0.8)
    # Original = cv2.resize(Original, (0, 0), fx=0.8, fy=0.8)
    # cv2.imshow("FinalCircles", im_in1)
    # cv2.imshow("Original", Original)
    # cv2.imshow("output", output)
    # cv2.imwrite("FinalCircles16050.png", im_in1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # show the images
    # cv2.imshow("Initial", im_in)
    # cv2.imshow("Final", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ################### save np arrays to csv ###################
    X = np.column_stack((New_Cx, New_Cy, Radii, C, B))
    np.savetxt("CentroidsAndRadii1111111.csv", X, delimiter=",")
    return X


################### Run and Print ###################
print(AccurateDropletSize(U))
################### timer ###################
print("--- %s seconds ---" % (time.time() - start_time))
