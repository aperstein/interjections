# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:16:29 2019

@author: yanivg
"""



import os
import os.path
import numpy as np
#import glob
import librosa
import matplotlib.pyplot as plt
#import time
import tensorflow as tf
import xml.etree.ElementTree as ET
#from matplotlib.pyplot import specgram
# %matplotlib inline
plt.style.use('ggplot')


#sound_names = ["ahah","mmm","nah","oy","negative_ex"]
#working_dir = 'D:/Afeka/FinalProject/Interjections/DataSets/Recordings/FixedLen/'
#working_dir = 'D:/Afeka/FinalProject/DataSets/Recordings/DataAugRecords/'
#working_dir = 'D:/Afeka/FinalProject/DataSets/Recordings/TestFolder/Tmp/'



class GENERATE_EXTRACED_FEATURE():
    
    
    def __init__(self, config):
        
        
        self._number_of_audio_per_speaker = 0       # Determines how many original audio files to take from each spoker
        self._start_count_from = 0                  # Determines from witch audio from speaker folder start counting the '_number_of_audio_per_speaker'
        self._speakers_list = []
        self._orig_directory = None
        self._target_directory = None

                                                    

        self._labels = []                           # Holds the classes labels
        self._config = config
        self._orig_wd = os.getcwd()
        #print("self.config=", self._config)
        
        self.read_directory_from_config()
        ##print("__init__ - self.orig_directory={} self.target_directory={} self.background_directory={}".format(self._orig_directory, self._target_directory, self._background_directory))
        
        
        print("__init__   self._orig_directory=", self._orig_directory)
        print("__init__   self._target_directory=", self._target_directory)
        print("__init__   self._number_of_audio_per_speaker=", self._number_of_audio_per_speaker)
        print("__init__   self._start_count_from=", self._start_count_from)
        print("__init__   self._speakers_list=", self._speakers_list)
        

        self.get_labels()

        # Create the target directory/temp directories in target directory to holds files between operations.
        # For example if file need to generate permutation ['0', '2', '3'], 
        # then each file after operation 2 and before operation 3 is saved in this temp folder.
        # At the end of all the oparation in the permutation, the files are transfers to the target directory.
        #dirsName = os.path.join(self._target_directory, "TempFolderFiles")
        if not os.path.exists(self._target_directory):
            os.makedirs(self._target_directory)
            
            
        self.save_folds()
        
        

    #############################################################################
    
    def read_directory_from_config(self):
    
    
        #print('================ in read_directory_from_config ==============')
        
        ##print('self._config=', self._config)
        
        xml_tree = ET.parse(self._config)
        root = xml_tree.getroot()
    
        # Get relevant Directories
        for directory in root.findall('directories'):            
            self._orig_directory = directory.get('orig_directory')            
            self._target_directory = directory.get('target_directory')            
            
        # Get relevant data
        for directory in root.findall('data'): 
            tmp_speakers =  directory.get('speakers')  
            self._speakers_list = tmp_speakers.split(',')       
            self._number_of_audio_per_speaker = directory.get('number_of_audio_per_speaker')            
            self._start_count_from = directory.get('start_count_from')            
            
            
            
        
    #############################################################################
    
    def get_labels(self):
        
        ##print("root dir = ", self._orig_directory)
        try:
            _, self._labels, _ = next(os.walk(self._orig_directory))
            
            print('class_labels = ' + str(self._labels))
        except Exception as e:
            print('Failed to get labels. ', str(e))
    
    #############################################################################


    # use this to process the audio files into numpy arrays
    def save_folds(self):
     
       #fold_name = 'fold' + str(k)
        #print ("\nSaving " + fold_name)
        #features, labels = self.parse_audio_files(data_dir, classes)
        try:
            features, y_labels = self.parse_audio_files()
            y_labels = self.one_hot_matrix(y_labels)
                
            print ("Features of X = ", features.shape)
            print ("Labels of Y = ", y_labels.shape)
            
            feature_file_path = os.path.join(self._target_directory, 'X.npy')
            labels_file_path = os.path.join(self._target_directory, 'Y.npy')
            np.save(feature_file_path, features)
            print ("Saved " + feature_file_path)
            np.save(labels_file_path, y_labels)
            print ("Saved " + labels_file_path)
        except OSError as e:
            print('Error processing in save_folds :  Error: %s' % e)

    #############################################################################

    def extract_feature(self, file_name):
        
        try:
            X, sample_rate = librosa.load(file_name)
            #print ("Features :",len(X), "sampled at ", sample_rate, "hz")
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        except OSError as e:
            print('Error processing in save_folds:  Error: %s' % e)
        return mfccs,chroma,mel,contrast,tonnetz

#=====================================================================================

    #def parse_audio_files(self, parent_dir,sub_dirs):
    def parse_audio_files(self):
        
        print('================ in parse_audio_files ==============')
        
        # print('sub_dirs=' + str(sub_dirs))
        features, labels = np.empty((0,193)), np.empty(0)
        

        for label, class_folder in enumerate(self._labels):
            
            n_class = class_folder.split('_')[1]
            print('FOR: n_class=' + str(n_class) + ', label=' + str(label) + ', class folder=' + str(class_folder))
            
            orig_directory_path = os.path.join(self._orig_directory, class_folder)
            #print('orig_directory_path=' + orig_directory_path)
    
            
            _, speakers, _ = next(os.walk(orig_directory_path))
            
            #print('class speakers = ', speakers)
            
            for _, speaker_folder in enumerate(speakers):
                
                # check here if speaker_folder in _speakers_list 
                # if no continue
                if speaker_folder not in self._speakers_list:
                    print(speaker_folder + " not in _speakers_list")
                    continue
            
                files_counter = 0
                #print(' speaker_folder folder=' + str(speaker_folder))
         
                orig_speaker_path =  os.path.join(orig_directory_path, speaker_folder) 
                
                #Iterate through all the wav files in the class folder, 
                #including subfolders like in negative_ex folder
                #for filename in glob.glob('**/*.wav', recursive=True):
                for filename in os.listdir(orig_speaker_path):
                    
                    #print("files_counter={} self.number_of_audio_per_speaker={} ".format(files_counter, self._number_of_audio_per_speaker))
                    if(files_counter < int(self._start_count_from)):
                        print("files_counter={}  - continue".format(files_counter))
                        files_counter+=1
                        continue
                    print(" Counter = " + str(files_counter) + " : take files from " + str(int(self._start_count_from)) + " to " + str((int(self._start_count_from) + int(self._number_of_audio_per_speaker))))
                    if(files_counter >= (int(self._start_count_from) + int(self._number_of_audio_per_speaker))):
                        print("files_counter={}  - break".format(files_counter))
                        break
                    
                    #print('filename path=' + filename)
                    try:
                        inputfile_path = os.path.join(orig_speaker_path, filename)
                        print('inputfile_path=' + inputfile_path)
                        
                        mfccs, chroma, mel, contrast, tonnetz = self.extract_feature(inputfile_path)
                        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                        features = np.vstack([features,ext_features])
                        #TODO - set a number for each class
                        labels = np.append(labels, n_class)
                        #labels = np.append(labels, 1)
                        #print('fn =')
                 
                        files_counter+=1
                    except OSError as e:
                        print('parse_audio_files: Error processing ' + inputfile_path + ' Error: %s' % e)
                        files_counter+=1
        
      
        return np.array(features), np.array(labels, dtype = np.int)

##=====================================================================================
#
#    def one_hot_encode(self, labels):
#        
#        print("In Function one_hot_encode")
#        
#        try:
#            n_labels = len(labels)
#            n_unique_labels = len(np.unique(labels))
#            one_hot_encode = np.zeros((n_labels,n_unique_labels))
#            one_hot_encode[np.arange(n_labels), labels-1] = 1
#        except OSError as e:
#            print('Error processing in save_folds:  Error: %s' % e)
#        return one_hot_encode

#=====================================================================================
    
    def one_hot_matrix(self, labels):
        """
        Creates a matrix where the i-th row corresponds to the ith class number 
        and the jth column corresponds to the jth training example. 
        So if example j had a label i. Then entry (i,j) will be 1. 
                         
        Arguments:
        labels -- vector containing the labels 
        n_classes -- number of classes, the depth of the one hot dimension
        
        Returns: 
        one_hot -- one hot matrix
        """
        
        try:
            n_classes = len(np.unique(labels))
            # Create a tf.constant equal to n_classes (depth), name it 'C'. (approx. 1 line)
            C = tf.constant(n_classes, name = "C")
            # Use tf.one_hot, be careful with the axis (approx. 1 line)
            one_hot_matrix = tf.one_hot(labels, C, axis=-1)
            # Create the session (approx. 1 line)
            sess = tf.Session()
            # Run the session (approx. 1 line)
            one_hot = sess.run(one_hot_matrix)
            # Close the session (approx. 1 line). See method 1 above.
            sess.close()
        except OSError as e:
            print('Error processing in save_folds:  Error: %s' % e)
        
        return one_hot

#=====================================================================================
    
#def assure_path_exists(path):
#    mydir = os.path.join(os.getcwd(), path)
#    if not os.path.exists(mydir):
#        os.makedirs(mydir)
#
##=====================================================================================
#        
#def load_npy(file_path):
#    
#    t1=time.time()
#    #array_reloaded = np.load('C:\FinalProject\Test\mmm_37.59_38.41.wav_st.npy')
#    array_reloaded = np.load(file_path)
#    
#    t2=time.time()
#    print(array_reloaded)
#
#    print(str(t2-t1) + " - Time took to load: {t2-t1} seconds.")
#    print('****************************************')
#    
#    print('\nShape: ',array_reloaded.shape)
#    print('\nShape[1]: ',array_reloaded.shape[1])

        
#=====================================================================================
        
def main():
#    os.chdir(working_dir)
#    labels = get_labels(working_dir)
#    save_folds(working_dir,labels)
    
    GENERATE_EXTRACED_FEATURE("D:\Afeka\FinalProject\DataSets\ScriptsCode\DataExtractFeature\ExtractedFeatureMng.xml")
    
    #return labels

#=====================================================================================
    
    
main()

#load_npy('D:/Afeka/FinalProject/DataSets/Recordings/DataAugRecordsTemp/X.npy')
#load_npy('D:/Afeka/FinalProject/DataSets/Recordings/DataAugRecordsTemp/Y.npy')
