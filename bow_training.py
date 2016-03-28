#Bag of Visual Words for object Recognition
import cv2
import numpy as np
import cPickle as pickle
import glob 
import os
import argparse as ap
from scipy.cluster import vq as vq
from sklearn import svm

class Training:
      def __init__(self):
          self.num_words = []
          self.num_clusters = 5000
      	  self.voc = []
	  self.idf = []
      
      def cluster(self,descriptor_list):
	  descriptor_list = descriptor_list[0]
          self.voc,distortion = vq.kmeans(descriptor_list,self.num_clusters)
          return self.voc # Codebook containing centroids

      def create_vocabulary(self,num_images):
          self.num_words = self.voc.shape[0]
          # Create the visual word for each image in the training class
          imwords = np.zeros((num_images,self.num_words))
	  # Next step form the histogram of the visual word for all the training images
	  for j in xrange(num_images):
              imwords[j] = self.calculate_histogram(desc[j])
	  
	  number_occurences = np.sum((imwords>0),axis = 0)
          self.idf = np.log((num_images)/(number_occurences+1))
	  return imwords

      def calculate_histogram(self,feature):
          words,distortion = vq.vq(self.voc,feature)
	  how,bin_edges = np.histogram(words,bins = range(words.shape[0]+1))
	    
	  return how

if __name__ == "__main__":
   parser = ap.ArgumentParser()
   parser.add_argument('-v','--verbose', help = 'Pass the full path of where the data resides')
   parser.add_argument('traindirectory',help = 'Directory to training images',action = 'store')
   args = parser.parse_args()

   os.chdir(args.traindirectory)
   cwd = os.getcwd()
   directory_list = os.listdir(cwd)
   sift = cv2.SIFT()
   desc = []
   image_ctr = 0
   label = []
   ctr = 0
   for directory in directory_list:
       current_directory = cwd+'/'+directory
       os.chdir(current_directory)
       file_list = glob.glob('*.jpg')
       for image in file_list:
	   label.append(ctr)
	   image_ctr+=1
           I = cv2.imread(image)
           kp,descriptor = sift.detectAndCompute(I,None)
           desc.append(descriptor)
       ctr+=1

#Create labels and append it to the training histogram
   label = np.asarray(label)
   label = np.reshape(label,(len(label),1))
   BOW = Training()
   v = BOW.cluster(desc)
   training_hist = BOW.create_vocabulary(image_ctr)
   training_data = np.hstack((training_hist,label))
   c = open("codebook.txt","wb")
   pickle.dump(v,c)
   c.close()   

   f = open("bag_of_words.txt","wb")
   pickle.dump(training_data,f)
   f.close()
