import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import time
import os
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt


############################
# sos: issue with problematic size of Matrix 2 --> if it is a row element blows up
#############################


#start the timer
start_time = time.time()

#read the csv files
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles
dirName = '/Users/georgedamoulakis/PycharmProjects/Droplets/3_csv/csv_practise_10_images';
listOfFiles = getListOfFiles(dirName)
listOfFiles.sort()
L = ( len(listOfFiles) - 1)

for i in range(L):
    print(listOfFiles[i])
    #M1 = genfromtxt(listOfFiles[i], delimiter=',', encoding= 'unicode_escape')
    #M2 = genfromtxt(listOfFiles[i+1], delimiter=',', encoding= 'unicode_escape')
    #print(M1)
    #print(M2)

print("--- %s seconds ---" % (time.time() - start_time))
