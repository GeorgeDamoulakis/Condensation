import cv2
import time
import numpy as np
#import math
#import matplotlib.pyplot as plt
#import pandas as p

start_time = time.time()

def InitialFix(img):
    def unsharp_mask(image, kernel_size=(9, 9), sigma=2.0, amount=2.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    image_name = '0'
    sharpened_image = unsharp_mask(img)
    gamma = adjust_gamma(img, gamma=0.6)
    gammaCrop = gamma[134:634, 134:634]
    return gammaCrop


ImageInput = cv2.imread('/Users/georgedamoulakis/Desktop/imagesFromM16V30C10/image_3000.tif', cv2.IMREAD_GRAYSCALE);
A = InitialFix(ImageInput)
cv2.imshow("Initial", ImageInput)
cv2.imshow("Final", A)
cv2.waitKey(0)
cv2.destroyAllWindows()



print("--- %s seconds ---" % (time.time() - start_time))