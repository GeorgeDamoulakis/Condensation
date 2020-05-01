import numpy as np
import time
import os
from numpy import genfromtxt
import pandas as pd

################## start the timer #######################
start_time = time.time()

################## read the CSV files #######################
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
dirName = '/Users/georgedamoulakis/PycharmProjects/Condensation/3_csvORtxt/csv_practise_10_images';
listOfFiles = getListOfFiles(dirName)
listOfFiles.sort()
L = (len(listOfFiles) - 1)
max=0

################## code starts here #######################
container = np.empty(((L+1),1), dtype=object)
for i in range(L+1):
    # edw ipirxe problima esbisa tin prwti grammi me ta logia giati to mperdeue
    M = genfromtxt(listOfFiles[i], delimiter=',', encoding= 'unicode_escape')
    container[i] =M.shape[0]

max = np.max(container)
#print(container, max)
dummy = np.array(( [0,0,0,0,0]  ), float)

for i in range(L+1):
    M = genfromtxt(listOfFiles[i], delimiter=',', encoding= 'unicode_escape')
    while M.shape[0] < max:
        M = np.vstack((M, dummy))

    my_df_1 = pd.DataFrame(M)
    my_df_1.columns =  ['0', '1', '2', '3', '4']
    my_df_1.to_csv(f'Fixed-sized matrix from {i} frame.csv', index=False)  # save as csv




print("--- %s seconds ---" % (time.time() - start_time))