import numpy as np
import time
import os
from numpy import genfromtxt

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

################## code starts here #######################
for i in range(L):
    # edw ipirxe problima esbisa tin prwti grammi me ta logia giati to mperdeue
    M1 = genfromtxt(listOfFiles[i], delimiter=',', encoding= 'unicode_escape')
    M2 = genfromtxt(listOfFiles[i+1], delimiter=',', encoding= 'unicode_escape')


    def G_Rate_function(M1, M2):
        M1_2col = np.empty((M1.shape[0], 2), dtype=float)
        for i in range(M1.shape[0]):
            M1_2col[i, 0] = M1[i, 0]
            M1_2col[i, 1] = M1[i, 1]

        M2_2col = np.empty((M2.shape[0], 2), dtype=float)
        for i in range(M2.shape[0]):
            M2_2col[i, 0] = M2[i, 0]
            M2_2col[i, 1] = M2[i, 1]

        # define the distance function:
        def distance(x1, y1, x2, y2):
            x_diff = 0
            y_diff = 0
            point_distance = 0
            x_diff = (x1 - x2) ** 2
            y_diff = (y1 - y2) ** 2
            point_distance = (x_diff + y_diff) ** 0.5
            return point_distance

        distances_stored = np.empty((M1_2col.shape[0], M2_2col.shape[0]), dtype=object)
        for i in range(M1_2col.shape[0]):
            d = 0
            for j in range((M2_2col.shape[0])):
                d = distance(M1_2col[i][0], M1_2col[i][1], M2_2col[j][0], M2_2col[j][1])
                distances_stored[i][j] = round(d, 2)
        # print(distances_stored)
        mini = np.empty((M1_2col.shape[0], M2_2col.shape[0]), dtype=object)
        for i in range(M1_2col.shape[0]):
            for j in range((M2_2col.shape[0])):
                if distances_stored[i, j] == np.amin(distances_stored[i]):
                    mini[i, j] = distances_stored[i, j]
                else:
                    mini[i, j] = -1
        # print(mini)
        G_Rate_matrix = np.empty((M1.shape[0], 1), dtype=object)
        for i in range(M1_2col.shape[0]):
            for j in range((M2_2col.shape[0])):
                if mini[i, j] == -1:
                    pass
                else:
                    G_Rate_matrix[i] = abs(
                        (M2[j][2] - M1[i][2]) / (M1[j][2])
                    )
        #print(G_Rate_matrix)
        G_Rate_M_no0 = (G_Rate_matrix == 0).sum(1)
        G_Rate_matrix_clean = G_Rate_matrix[G_Rate_M_no0 == 0, :]
        if G_Rate_matrix_clean.size > 0:
            G_Rate = round((np.mean(G_Rate_matrix_clean)), 3)
        else:
            G_Rate = 0


       # print(f'the total rate is: ', G_Rate)

        MatrixA = np.empty((3,), dtype=object)
        MatrixA[0] = G_Rate
        MatrixA[1] = M1[0][3]
        MatrixA[2] = M1[0][4]
        print(f'The matrix A is: ')
        print(MatrixA)



        return G_Rate


    a = G_Rate_function(M1, M2)

################## kill the timer #######################
print("--- %s seconds ---" % (time.time() - start_time))