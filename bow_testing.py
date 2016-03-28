from sklearn import svm 
import numpy as np
import argparse as ap
import glob as glob
import cv2
import os
from scipy.cluster import vq
import cPickle as pickle
from sklearn import svm

sift = cv2.SIFT()
parser = ap.ArgumentParser()
parser.add_argument('-v','--verbose', help = 'Pass the full path of where the testing data resides')
parser.add_argument('testdirectory',help = 'Directory to testing images',action = 'store')  
args = parser.parse_args()

os.chdir(args.testdirectory)
image_name = raw_input('Enter the file name: ' )
image_name = image_name+'.jpg'
I = cv2.imread(image_name)
kp,descriptor = sift.detectAndCompute(I,None)
cv2.imshow("Query image",I)
cv2.waitKey(0)

    
with open("codebook.txt","rb") as f:
     codebook = pickle.load(f)

words,distortion = vq.vq(codebook,descriptor)
image_histogram,bin_edges = np.histogram(words,bins = range(words.shape[0]+1))


with open("bag_of_words.txt","rb") as train:
     training_data = pickle.load(train)

rows,columns, = np.shape(training_data)
y = training_data[:,columns-1]
X = training_data[:,0:columns-1]
C = 5
svc = svm.SVC(kernel = 'rbf',gamma = 0.5,C = C).fit(X,y)
print svc.predict(image_histogram)


