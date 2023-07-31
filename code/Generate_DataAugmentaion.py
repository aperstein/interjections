
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import itertools
import os
import os.path

#import wave
import subprocess
import sox
import shutil as sh
import distutils.dir_util as dir_util

import wave
from pydub import AudioSegment

class data_augmentation_generator():
    
    
    def __init__(self, config):
        
        self._create_whitenoise = 'false'
        self._number_of_audio_per_speaker = 0       # Determines how many original audio files to take from each spoker
        self._wav_target_len = 0                    # The target wav length for wavs after augmentation
        self._wav_min_len = 0                       # The min length allowed for each orig wav file. If length is shorter the file will removed.
        self._silence_ratio_limit_percent = 0       # For adding silence - it determines the borders of silence in the beginning and ending of the wav file.
                                                    # If this value is 15 for example, then before adding the silence to the wav file, the length of silence
                                                    # to add is calculated. Then 15 percent of this length is added to the beginning of the wav, and the rest
                                                    # is added to the tail of the wav file. This value is promoted in 5 before each wav file, with modulo of
                                                    # 100 - _silence_ratio_limit_percent (The maximum value)
        self._silence_ratio_limit_percent_tmp = 0 
        
        self._orig_directory = None
        self._target_directory = None
        self._background_directory = None
        self._df_operations = []
        self._num_of_operations = 0                 # Counter of operations
        self._permutation_list = []                 # This list hold in each row the permutation of each audio 
                                                    # according to the operations from 'params_dict_list'
                                                    # For example: permutation [0,2,3] means that the audio file will
                                                    # do the next data augmentation:
                                                    # step1: element 0 - means this is the original audio
                                                    # step2: element 2 - means that the audio will do data augmentaion (operation from xml) with id 2
                                                    # step3: element 3 - means that the audio from the end of step 2 will do data augmentaion (operation from xml) with id 3
                                                    # $$$$$$$$ Each step produce some output audio according to the number of params value of the operation  $$$$$$$$
                                                    

        self._labels = []                           # Holds the classes labels
        self._config = config
        self._orig_wd = os.getcwd()
        #print("self.config=", self._config)
        
        self.read_directory_from_config()
        ##print("__init__ - self.orig_directory={} self.target_directory={} self.background_directory={}".format(self._orig_directory, self._target_directory, self._background_directory))
        
        self.read_data_augmentation_params_from_config()
        
        #print("__init__   self._create_whitenoise=", self._create_whitenoise)
        #print("__init__   self._num_of_operations=", self._num_of_operations)
        #print("__init__   self._number_of_audio_per_speaker=", self._number_of_audio_per_speaker)
        #print("type of self._number_of_audio_per_speaker=", type(self._number_of_audio_per_speaker))
        
           
        #print("self.wav_target_len=", self._wav_target_len)
        #print("type of self.wav_target_len=", type(self._wav_target_len))
        #print("self.wav_min_len=", self._wav_min_len)
        #print("type of self.wav_min_len=", type(self._wav_min_len))
        #print("self.silence_ratio_limit_percent=", self._silence_ratio_limit_percent)
        #print("type of self.silence_ratio_limit_percent=", type(self._silence_ratio_limit_percent))
  
        

#        
#       #print("_df_operations.loc[1]=" ,self._df_operations.loc['1'])
#       #print("_df_operations.loc[2]=" ,self._df_operations.loc['2'])
#       #print("_df_operations.loc[3]=" ,self._df_operations.loc['3'])
#        
#       #print('operation = ',self._df_operations.loc['3']["operation"])
#       #print('params = ',self._df_operations.loc['3']["params"])
#       #print('factor list = ',self._df_operations.loc['3']["params"]["factor"])
#       #print('factor list len= ', len(self._df_operations.loc['3']["params"]["factor"]))
#       #print('factor list one item= ', self._df_operations.loc['3']["params"]["factor"][0])

        self.get_labels()
        
        self.build_permutation_list()
        
        
        #print('permutations_list')
   
        #for per in self._permutation_list: 
            #print(per)
         
        # Create the target directory/temp directories in target directory to holds files between operations.
        # For example if file need to generate permutation ['0', '2', '3'], 
        # then each file after operation 2 and before operation 3 is saved in this temp folder.
        # At the end of all the oparation in the permutation, the files are transfers to the target directory.
        dirsName = os.path.join(self._target_directory, "TempFolderFiles")
        if not os.path.exists(dirsName):
            os.makedirs(dirsName)
        
        self.run_data_augmentation_all()
        
        
        # if os.path.exists(dirsName):
        #    os.remove(dirsName)
        
        
        # Change the working directory to the orig wd to allowed deleting the temp folders manualy.
        os.chdir(self._orig_wd)
        #self.token = self.read_directory_from_config(config)
        #self.base = "https://api.telegram.org/bot{}/".format(self.token)

#    def get_updates(self, offset=None):
#        url = self.base + "getUpdates?timeout=100"
#        if offset:
#            url = url + "&offset={}".format(offset + 1)
#        r = requests.get(url)
#        return json.loads(r.content)
#
#    def send_message(self, msg, chat_id):
#        url = self.base + "sendMessage?chat_id={}&text={}".format(chat_id, msg)
#        if msg is not None:
#            requests.get(url)

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
            self._background_directory = directory.get('background_directory')
            
        
    #############################################################################
    
 
    def read_data_augmentation_params_from_config(self):
    
    
        #print('================ in read_data_augmentation_params_from_config ==============')
        
        #print('self.config=', self._config)
        operation_id_list = []                 # List of operations id's
        operation_name_list = []               # List of operations names
        params_dict_list = []                  # Dictionary of operations and params values
        
        xml_tree = ET.parse(self._config)
        root = xml_tree.getroot()
        
        
        # find the white_noise parameter
        for directory in root.findall('dataAugmentations'):
            
            # Get the indication about generate white noise augmentation
            self._create_whitenoise = directory.get('create_whitenoise')
            # Get the number of original audio files to take from each spoker
            self._number_of_audio_per_speaker = directory.get('number_of_audio_per_speaker')
            
            # Get the target wav length for wavs after augmentation
            self._wav_target_len = directory.get('wav_target_len')
            # Get  min length allowed for each orig wav file
            self._wav_min_len = directory.get('wav_min_len')
            # Get the ratio to add silence to the original wav file.
            self._silence_ratio_limit_percent = directory.get('silence_ratio_limit_percent')
            self._silence_ratio_limit_percent_tmp = self._silence_ratio_limit_percent
            

        
        # Loop through operation elements
        for elem in root.iter(tag='operation'):
            #print ('elem.tag=' , elem.tag, 'elem.attrib=' , elem.attrib)
            #print ('elem.attrib[id]=' , elem.attrib['id'])
            #print ('elem.attrib[name]=' , elem.attrib['name'])
            
            # Init the parameters dictionary of the current operation
            tmp_dict = {}
            
            # Insert the id of the operation into id's list
            operation_id_list.append(elem.attrib['id'])
            # Insert the name of the operation into name's list
            operation_name_list.append(elem.attrib['name'])
            
            # Loop through  param elements of the current operation
            for subelem in elem.iter(tag='param'):
                #print (subelem.tag, subelem.attrib, subelem.text)
                #print ('subelem.tag=' , subelem.tag, 'subelem.attrib=' , subelem.attrib, 'subelem.text=' , subelem.text)
                #print ('subelem.attrib[desc]=' , subelem.attrib['desc'])
                
                # Get the param value
                str_params = str(subelem.text)
                # Create a list from the param values
                tmp_list = str_params.split(',')
        
                # Insert the parameter description (key) and its values (value) into the params dictionary
                tmp_dict[subelem.attrib['desc']] = tmp_list
            ##print('tmp_dict=' , tmp_dict)
            params_dict_list.append(tmp_dict)
            self._num_of_operations += 1
    
        # Create dictionary that contain the operation and their params as a key-value pair
        data_dict = {'operation':operation_name_list,
                'params':params_dict_list}
                 
        #print('self.operation_id_list = ', operation_id_list)
        
        # Crate a data frame that contain the data_dict with corresponding index for each [operation-params] 
        self._df_operations = pd.DataFrame(data_dict, index = operation_id_list)
 
#        # Print the output.
#        #print(df)
#        
#        #print("df.loc[1]=" ,df.loc['1'])
#        #print("df.loc[2]=" ,df.loc['2'])
#        #print("df.loc[3]=" ,df.loc['3'])
#        
#        #print('operation = ',df.loc['3']["operation"])
#        #print('params = ',df.loc['3']["params"])
#        #print('factor list = ',df.loc['3']["params"]["factor"])
#        #print('factor list len= ', len(df.loc['3']["params"]["factor"]))
#        #print('factor list one item= ', df.loc['3']["params"]["factor"][0])
    #############################################################################
    
    def build_permutation_list(self):
    
        num_of_op_arr = np.arange(self._num_of_operations+1)
        
        #l_per = list(itertools.permutations([0, 1, 2, 3]))
        permutation_list = list(itertools.permutations(num_of_op_arr))
    
        
        for per in permutation_list: 
            
            # if the permutation contain only '0' index then continue. Zero is the original audio file.
            
            cur_per_list = list(per)
            cur_per_list[0] = 0
            
            # Temp list to contain only the relevant operation id of current permutation
            tmp_list = []
            for i in range(len(cur_per_list)):
                # Append the item from current permutaion only if its value corresponds to its index
                if(cur_per_list[i]==i):
                    tmp_list.append(str(i))
            
            if(tmp_list not in self._permutation_list and len(tmp_list) > 1):
                self._permutation_list.append(tmp_list)  
                
        self._permutation_list.sort(key=len)
        
    #############################################################################
                
    def get_labels(self):
        
        ##print("root dir = ", self._orig_directory)
        try:
            _, self._labels, _ = next(os.walk(self._orig_directory))
            
            #print('class_labels = ' + str(self._labels))
        except Exception as e:
            print('Failed to get labels. ', str(e))
    
    #############################################################################
    
    def run_data_augmentation_all(self):
        
        #print('================ in run_data_augmentation_all ==============')
        
        
        try:
            
            # First copy 'self._number_of_audio_per_speaker' files from source to target
            self.copy_files_from_orig()
           
            # Loop through all permutation
            for per in self._permutation_list: 
                #print('per (self._permutation_list) = ', per)
                
                isFirst = True
                
                # Create a folder for each permutation
                foldername = 'per_'
                for num in per:
                    foldername += str(num)
                #print(foldername)
                
                
                   # Create the folder of the audio files for the current permutation (If not exist)
                per_target_folder = os.path.join(self._target_directory, "TempFolderFiles", foldername)
                if not os.path.exists(per_target_folder):
                    os.makedirs(per_target_folder)
             
                    
                # Checking if part of this current permutaion is already created.
                # If yes, then copy the existing permutation into the current permutaion folder (per_target_folder)
                # and start with the missing operation_id.
                # For example - if the current permutaion is 0123, the check if permutaion 012 exist 
                # (if yes - copy folder 012 into folder 0123 and continue with operation_id=3). If 012 doesn't exist
                # then check if permutation 01 exist (if yes copy 01 into 0123 and continue with operation_id=2)
                # If part of the current permutaion is exist, the update the valu of per (3 instead of 0123, and also isfirst=False)
                per, is_partially_exist = self.check_partially_permutation(per)
                #print('***********************  per = ', per, ' ,   is_partially_exist = ', is_partially_exist, ' *********************** ' )
                #per = return_pered
                if is_partially_exist == True:
                    isFirst = False
                    
             
                for operation_id in per:
                    # operation_id='0' means this is the original audio.
                    if operation_id=='0':
                        continue
                    
                    if isFirst:
                        #print('******  In IS_FIRST = ', isFirst)
                        per_orig_folder = self._target_directory
                        isFirst = False
                    else:
                        per_orig_folder = per_target_folder
                        
                    #print('******  per_orig_folder = ', per_orig_folder)
                         
                    # run operation with the opearation to run, the current original folder containing the audio files to change
                    # and the target folder where to put the audio files after the operation is done.
                    self.run_operation(operation_id,per_orig_folder, per_target_folder)   
                    
                           
            if (self._create_whitenoise.lower() == 'true'):
                self.generate_whitenoise_all()
            
            # Copy and unified all the folders with different data augmentation into the target folder
            self.copy_per_dirs_to_target()
            
            # Fix the duration of all files in the target folder - not including files with background noise.
            self.fix_target_files_duration()
                
    
        except OSError as e:
            print('Error in run_data_augmentation_all.  Error: %s' % e)
            os.chdir(self._orig_wd)
        
        # Change the working directory to the orig wd to allowed deleting the temp folders manualy.
        # os.chdir(self._orig_wd)
            
            
            
#    #############################################################################
    
    def generate_whitenoise_all(self):
        
        #print('================ in generate_whitenoise_all ==============')
        
        for label, class_folder in enumerate(self._labels):
            
            #n_class = class_folder.split('_')[1]
            ##print('n_class=' + str(n_class) + ', label=' + str(label) + ', class folder=' + str(class_folder))
            
            orig_directory_path = os.path.join(self._orig_directory, class_folder)
            #print('orig_directory_path=' + orig_directory_path)
            
                  
            target_directory_path = os.path.join(self._target_directory, class_folder)
            if not os.path.exists(target_directory_path):
                os.makedirs(target_directory_path)
            
            _, speakers, _ = next(os.walk(orig_directory_path))
            
            #print('class speakers = ', speakers)
            
            for _, speaker_folder in enumerate(speakers):
            
                files_counter = 0
                #print(' speaker_folder folder=' + str(speaker_folder))
         
                orig_speaker_path =  os.path.join(orig_directory_path, speaker_folder) 
                
                target_speaker_path =  os.path.join(target_directory_path, speaker_folder) 
                 
                if not os.path.exists(target_speaker_path):
                    os.makedirs(target_speaker_path)
                
                os.chdir(target_speaker_path)
            
                
                #Iterate through all the wav files in the class folder, 
                #including subfolders like in negative_ex folder
                #for filename in glob.glob('**/*.wav', recursive=True):
                for filename in os.listdir(orig_speaker_path):
                    
                    #print("files_counter={} self.number_of_audio_per_speaker={} ".format(files_counter, self._number_of_audio_per_speaker))
    
                    if(files_counter >= int(self._number_of_audio_per_speaker)):
                        break
                    
                    #print('filename path=' + filename)
                    try:
                        inputfile_path = os.path.join(orig_speaker_path, filename)
                        #print('inputfile_path=' + inputfile_path)
                        
                        # Checking if the audio file is valid and corrections. If the file is too short or too long - don't copy it.
                        audio_duration = self.get_wav_duration(inputfile_path)
                        #print('audio_duration:' + str(audio_duration))
                        if(audio_duration <=  float(self._wav_min_len) or audio_duration > float(self._wav_target_len)):
                            continue
                        
                        # Creating a fix duration of the input file before adding the background noise
                        # for the reason that the audio part from the origin record will be spreed out 
                        # differently acroos the output file with different 
                        fix_duration_inputfile_path = self.set_output_filename(filename, 'tmp')
                        #print('fix_duration_inputfile_path=' + fix_duration_inputfile_path)
                        sh.copyfile(inputfile_path, fix_duration_inputfile_path)
                        self.fix_wav_length(fix_duration_inputfile_path,7)
                        
                        name_addition = '[wm]'

                        whitenoise_outputfile = self.set_output_filename(filename, name_addition)
                        
                        outputfile_path = os.path.join(target_speaker_path, whitenoise_outputfile)
                        #print('************************')
                        self.add_whitenoise(str(fix_duration_inputfile_path),str(outputfile_path))
                        os.remove(fix_duration_inputfile_path)
                        files_counter+=1
                    except OSError as e:
                        print('Error processing ' + inputfile_path + ' Error: %s' % e)
                        files_counter+=1
    #return np.array(features), np.array(labels, dtype = np.int)
        
    #############################################################################  
    
    
            
    def copy_files_from_orig(self):
        
        #print('================ in copy_files_from_orig ==============')
        
        for label, class_folder in enumerate(self._labels):
            
            #n_class = class_folder.split('_')[1]
            ##print('n_class=' + str(n_class) + ', label=' + str(label) + ', class folder=' + str(class_folder))
            
            orig_directory_path = os.path.join(self._orig_directory, class_folder)
            #print('orig_directory_path=' + orig_directory_path)
            
                  
            target_directory_path = os.path.join(self._target_directory, class_folder)
            if not os.path.exists(target_directory_path):
                os.makedirs(target_directory_path)
            
            _, speakers, _ = next(os.walk(orig_directory_path))
            
            #print('class speakers = ', speakers)
            
            for _, speaker_folder in enumerate(speakers):
            
                files_counter = 0
                #print(' speaker_folder folder=' + str(speaker_folder))
         
                orig_speaker_path =  os.path.join(orig_directory_path, speaker_folder) 
                
                target_speaker_path =  os.path.join(target_directory_path, speaker_folder) 
                 
                if not os.path.exists(target_speaker_path):
                    os.makedirs(target_speaker_path)
                
                os.chdir(target_speaker_path)
            
                
                #Iterate through all the wav files in the class folder, 
                #including subfolders like in negative_ex folder
                #for filename in glob.glob('**/*.wav', recursive=True):
                for filename in os.listdir(orig_speaker_path):
                    
                    #print("files_counter={} self.number_of_audio_per_speaker={} ".format(files_counter, self._number_of_audio_per_speaker))
    
                    if(files_counter >= int(self._number_of_audio_per_speaker)):
                        break
                    
                    #print('filename path=' + filename)
                    try:
                        inputfile_path = os.path.join(orig_speaker_path, filename)
                        
                        # Checking if the audio file is valid and corrections. If the file is too short or too long - don't copy it.
                        audio_duration = self.get_wav_duration(inputfile_path)
                        #print('audio_duration:' + str(audio_duration))
                        if(audio_duration <=  float(self._wav_min_len) or audio_duration > float(self._wav_target_len)):
                            continue
                        ##print('inputfile_path=' + inputfile_path)
                        outputfile_path = os.path.join(target_speaker_path, filename)     
                        sh.copyfile(inputfile_path, outputfile_path)
                        
#                        if (self._create_whitenoise.lower() == 'true'):
#                             name_addition = '[wm]'
#                             # filename.split('_')[0] + '_' + '[wn]_' + filename.split('_')[1] + '_' + filename.split('_')[2]
#                             #  whitenoise_outputfile = filename.split('_')[0] + '_' + '[wn]_' + filename.split('_')[1] + '_' + filename.split('_')[2]
#                             whitenoise_outputfile = self.set_output_filename(filename, name_addition)
#                             self.add_whitenoise(str(outputfile_path),str(whitenoise_outputfile))
                        
                        files_counter+=1
                    except OSError as e:
                        print('Error processing ' + inputfile_path + ' Error: %s' % e)
                        files_counter+=1
                             
    ############################################################################# 
    
    
    def check_partially_permutation(self, cur_per):
        
        #print('================ in check_partially_permutation ==============')
        
        return_per = []
        partially_per_folder = ''
        is_partially_exist = False
        cur_per_foldername = ''
        
        # Get the folder name of current permutation
        for num in cur_per:
            cur_per_foldername += str(num)
        #print('cur_per_foldername (cur_per) = ', cur_per_foldername)
        
        try:
            
                part_per_foldername = ''
                temp_foldername = ''
                for num in cur_per:
                    temp_foldername += str(num)
                    #print('temp_foldername (cur_per) = ', temp_foldername)
                    
                    if temp_foldername == cur_per_foldername:
                        #print('continue')
                        continue
                
                    # Checking if partially per folder of cur_per exist
                    temp_per = os.path.join(self._target_directory, "TempFolderFiles", 'per_' + temp_foldername)
                    #print('<<< temp_per >>> = ', temp_per)
                    if os.path.exists(temp_per):
                        #print('((( temp_per exists )))')
                        partially_per_folder = temp_per
                        is_partially_exist = True
                        part_per_foldername = temp_foldername
                
                # If it's a true partially folder of the current permutaion folder, 
                # then copy the content of the partially folder into the cuurent permutation folder
                #print('is_partially_exist = ', is_partially_exist, '   len(part_per_foldername) = ', len(part_per_foldername), '   len(cur_per_foldername) = ', len(cur_per_foldername))
                if ((is_partially_exist == True) and (len(part_per_foldername) < len(cur_per_foldername))):
                    #print(' -----------  copy the tree of ', part_per_foldername, ' into ', cur_per_foldername, '--------')
                    target_per_folder = os.path.join(self._target_directory, "TempFolderFiles", 'per_' + cur_per_foldername)
                    self.copy_directory_tree(partially_per_folder, target_per_folder)
                    
                  
                    for index, num in enumerate(cur_per):
                        print ('index = ', index, ' : num in cur_per =', num, ' : len=' ,len(part_per_foldername))
                        if (int(index) < len(part_per_foldername) and num == part_per_foldername[index]):
                            continue
                        else: 
                            return_per.append(num)
                            
                # If partially folder not found, return the current permutation.
                else:
                    return_per = cur_per
                      
                      
        except OSError as e:
            print('Error in check_partially_permutation. Error: %s' % e)
        
        #print('return ' , return_per, ' : ', is_partially_exist)
        return return_per, is_partially_exist
        
        
    ############################################################################# 
    
    def copy_directory_tree(self, src, dest):
        
        #print('================ in copy_directory_tree ==============')
        
        #print('src =', src)
        #print('dest =', dest)
        
        
        try:
                
            for label, class_folder in enumerate(self._labels):
        
                
                orig_directory_path = os.path.join(src, class_folder)
                #print('orig_directory_path=' + orig_directory_path)
                
                target_directory_path = os.path.join(dest, class_folder)
                #print('target_directory_path=' + target_directory_path)
                
                if os.path.exists(target_directory_path):
                    #print('REMOVE DIRS')
                    os.removedirs(target_directory_path)
                sh.copytree(orig_directory_path, target_directory_path)
        # Directories are the same
        except sh.Error as e:
            print('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('Directory not copied. Error: %s' % e)
        
    
            
    
        
#        try:
#            sh.copytree(src, dest)
#        # Directories are the same
#        except sh.Error as e:
#            #print('Directory not copied. Error: %s' % e)
#        # Any error saying that the directory doesn't exist
#        except OSError as e:
#            #print('Directory not copied. Error: %s' % e)
        
    #############################################################################
    
        
    def run_operation(self, operation_id, per_orig_folder, per_target_folder):
        
        #print('================ in run_operation ==============')
        
               
        operation_name = self._df_operations.loc[str(operation_id)]["operation"]
        operation_params = self._df_operations.loc[str(operation_id)]["params"]
       
        #print('operation ',str(operation_id), ' = ', operation_name)
        #print('operation ',str(operation_id), ' = ', operation_params)
        
        for label, class_folder in enumerate(self._labels):
            
            #n_class = class_folder.split('_')[1]
            ##print('n_class=' + str(n_class) + ', label=' + str(label) + ', class folder=' + str(class_folder))
            
            orig_directory_path = os.path.join(per_orig_folder, class_folder)
            #print('orig_directory_path=' + orig_directory_path)
            
                  
            target_directory_path = os.path.join(per_target_folder, class_folder)
            if not os.path.exists(target_directory_path):
                os.makedirs(target_directory_path)
            
            _, speakers, _ = next(os.walk(orig_directory_path))
            
            ##print('class speakers = ', speakers)
            
            for _, speaker_folder in enumerate(speakers):
            
                ###files_counter = 0
                ##print(' speaker_folder = ' + str(speaker_folder))
         
                orig_speaker_path = os.path.join(orig_directory_path, speaker_folder) 
                
                target_speaker_path = os.path.join(target_directory_path, speaker_folder) 
                 
                if not os.path.exists(target_speaker_path):
                    os.makedirs(target_speaker_path)
                
                os.chdir(target_speaker_path)
            
                
                #Iterate through all the wav files in the class folder, 
                #including subfolders like in negative_ex folder
                #for filename in glob.glob('**/*.wav', recursive=True):
                for filename in os.listdir(orig_speaker_path):
                    
                    ####print("files_counter={}   ******   self.number_of_audio_per_speaker={} ".format(files_counter, self._number_of_audio_per_speaker))
    
                    ###if(files_counter >= int(self._number_of_audio_per_speaker)):
                       ### break
                    
                    ##print('filename path=' + filename)
                    try:
                        inputfile_path = os.path.join(orig_speaker_path, filename)
                        ##print('inputfile_path=' + inputfile_path)
                        
                        # BACKGROUND Operation
                        if(operation_name=="background"):
                            
                            #Iterate through all the wav files in the background noise folder.
                            #Each filename from orig folder mixed with all the background noise wavs.
                            with os.scandir(self._background_directory) as bg_entries:
                                for bg_entry in bg_entries:
                                    #
                                    bg_filename = bg_entry.name            
                                    bg_file_path = os.path.join(self._background_directory, bg_filename)
                                    ##print('back ground filename path=' + bg_file_path)
                                    
#                                    OrigWavVolume_list = operation_params["OrigWavVolume"]
#                                    #print('OrigWavVolume_list = ',OrigWavVolume_list)
#                                    BckGroundWavVolume_list = operation_params["BckGroundWavVolume"]
#                                    #print('BckGroundWavVolume_list = ',BckGroundWavVolume_list)
                                    
                                    df_volume = pd.DataFrame(operation_params)
                                    ##print('df_volume = ',df_volume)
                                    
                            
                                    #Each mixed wav file (orig wav & background noise wav) is created with different volumes 
                                    # accordind the ones defines in df_volume dataframe.
                                    for index, row in df_volume.head().iterrows():
                                        # access data using column names
                                        ##print(index, '  :  orig_volume = ', row['OrigWavVolume'], ' $  bg_volume = ', row['BckGroundWavVolume'])
                                        
                                        
                                        orig_volume  = row['OrigWavVolume']
                                        bg_volume  = row['BckGroundWavVolume']
                                        
                                        name_addition = '[bg_' + bg_filename.split('.')[0] + '_ov' + str(orig_volume) + '_bv' + str(bg_volume) + ']'
                                        outputfile = self.set_output_filename(filename, name_addition)
                                        ##print('outputfile=' + outputfile)
                                        outputfile_path = os.path.join(target_speaker_path, outputfile)
                                        
                                        # Creating a fix duration of the input file before adding the background noise
                                        # for the reason that the audio part from the origin record will be spreed out 
                                        # differently acroos the output file with different 
                                        fix_duration_inputfile_path = self.set_output_filename(filename, 'tmp')
                                        #print('fix_duration_inputfile_path=' + fix_duration_inputfile_path)
                                        sh.copyfile(inputfile_path, fix_duration_inputfile_path)
                                        self.fix_wav_length(fix_duration_inputfile_path,7)
                                        
                                        #outputfile = filename.split('_')[0] + '_' + '[bg_' + bg_filename.split('.')[0] + '_ov' + str(orig_volume) + '_bv' + str(bg_volume) + ']_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                                        ##print('outputfile=' + outputfile)
                                        #outputfile_path = os.path.join(target_directory_path, outputfile)
                                        self.add_Background_noise(fix_duration_inputfile_path, bg_file_path, outputfile_path, str(orig_volume), str(bg_volume))
                                        os.remove(fix_duration_inputfile_path)
                            
                        
                        # TEMPO Operation
                        if(operation_name=="tempo"):
                        
                            factor_list = operation_params["factor"]
                            ##print('factor list = ',factor_list)
                            for factor in factor_list:
                                #outputfile = filename.split('_')[0] + '_' + '[tm_' + str(factor) + ']_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                                name_addition = '[tm_' + str(factor) + ']'
                                outputfile = self.set_output_filename(filename, name_addition)
                                ##print('outputfile=' + outputfile)
                                outputfile_path = os.path.join(target_speaker_path, outputfile)
                                self.change_tempo(float(factor), str(inputfile_path), str(outputfile_path))
                   
                            ####print('************************')
                            ###files_counter = files_counter+1
                            #os.remove(os.path.join(target_speaker_path, filename))
                        
                        # PITCH Operation
                        if(operation_name=="pitch"):
                
                            semitone_list = operation_params["semitone"]
                            ##print('semitone list = ',semitone_list)
                            for semitone in semitone_list:
                                name_addition = '[pt_' + str(semitone) + ']'
                                #outputfile = filename.split('_')[0] + '_' + '[pt_' + str(semitone) + ']_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                                outputfile = self.set_output_filename(filename, name_addition)
                                ##print('outputfile=' + outputfile)
                                outputfile_path = os.path.join(target_speaker_path, outputfile)
                                self.change_pitch(float(semitone), inputfile_path,outputfile_path)
           
                            ####print('************************')
                            ###files_counter = files_counter+1
                            
                        os.remove(os.path.join(target_speaker_path, filename))
                            
                    
                    except OSError as e:
                        print('Error processing ' + inputfile_path + ' Error: %s' % e)
                        
        
    #############################################################################  
        
    def add_whitenoise(self, inputwav_path, outputwav_path):
        
        #print('================ in add_whitenoise ==============')
        
        #print('sox ' + str(inputwav_path) + ' -p synth whitenoise vol 0.02 | sox -m ' + str(inputwav_path) + ' - ' + str(outputwav_path))

    
        cmdAddWhiteNoise = str('sox ' + str(inputwav_path) + ' -p synth whitenoise vol 0.02 | sox -m ' + str(inputwav_path) + ' - ' + str(outputwav_path))

        try:
            # returns the exit code
            ret_val = subprocess.call(cmdAddWhiteNoise, shell=True)                 
    
        except OSError as e:
            print("Error adding wn " + inputwav_path + ' Error: %s' % e)
           
        #print('returned value:', ret_val)
        return ret_val
    
    ############################################################################# 
    
    
    
    def add_Background_noise(self, inputwav_path, bgwav_path, outputwav_path, orig_volume, background_volume):
        
        try:
          
            cmdMixWavs = 'sox -m -v ' + orig_volume + ' ' + inputwav_path + ' -v ' + background_volume + ' ' + bgwav_path + ' ' + outputwav_path
        
            ##print('cmdMixWavs:', cmdMixWavs)
        
            # returned_value = subprocess.call(cmdMixWavs, shell=True)  # returns the exit code in unix
            subprocess.call(cmdMixWavs, shell=True)  # returns the exit code in unix
        
            ##print('returned value:', returned_value)
    
        except OSError as e:
            print("Error add_Background_noise " + inputwav_path + ' Error: %s' % e)           
          
       
    #############################################################################
    
    def change_tempo(self, factor,inputwav_path, outputwav_path):
    
       try:
          
           # create transformer
           tfm = sox.Transformer()
        
           tfm.tempo(float(factor),'s')   #tempo(factor, audio_type=None, quick=False)
           # create the output file.
           tfm.build(str(inputwav_path), str(outputwav_path))
       except OSError as e:
           print("Error change tempo " + inputwav_path + ' Error: %s' % e)
           
    ############################################################################# 
    
    
    def change_pitch(self, semitones,inputwav_path, outputwav_path):
    
   
        try:
            # create transformer
            tfm = sox.Transformer()
            # create the output file.
            tfm.pitch(semitones)
            tfm.build(inputwav_path, outputwav_path)
    
        except OSError as e:
           print("Error change pitch " + inputwav_path + ' Error: %s' % e)
           
    ############################################################################# 
    
    
    ############################################################################# 
    
    
    def set_output_filename(self, filename,name_addition):
    
   
        try:
            filename_list = filename.split('_')
            
            newfilename = ''
            
            for section in filename_list:
                if section != filename_list[len(filename_list)-1] and section != filename_list[len(filename_list)-2]:
                    newfilename += section + '_'
                    
                    
            newfilename += name_addition + '_' + filename_list[len(filename_list)-2] + '_' + filename_list[len(filename_list)-1]
            
            return newfilename
    
        except OSError as e:
           print("Error set_output_filename " + ' Error: %s' % e)
           
    ############################################################################# 
    
    def copy_per_dirs_to_target(self):
     
        _, permutation, _ = next(os.walk(os.path.join(self._target_directory, "TempFolderFiles")))
                
        #print('class permutation = ', permutation)
                
        for _, per_folder in enumerate(permutation):
            
            ###files_counter = 0
            #print(' per_folder = ' + str(per_folder))
     
            orig_per_path = os.path.join(self._target_directory, "TempFolderFiles", per_folder) 
            #print(' orig_per_path = ' + str(orig_per_path))
            
            try:
                dir_util.copy_tree(orig_per_path, self._target_directory,)
            except OSError as e:
                print('Directory not copied. Error: %s' % e)
      
     #############################################################################   
        
    def get_wav_duration(self, wav_filename):
        
        f = wave.open(wav_filename, 'r')
        ##print("222")
        frames = f.getnframes()
        ##print("333")
        rate = f.getframerate()
        duration = frames / float(rate)
        f.close()
        return duration
    
    #############################################################################
    
    def add_silence_to_wav(self, wav_filename, duration_ms, location):
    
        # create silence audio segment of duration_ms length
        silence_segSment = AudioSegment.silent(duration=duration_ms)  #duration in milliseconds
        
        #read wav file to an audio segment
        orig_wav = AudioSegment.from_wav(wav_filename)
        
        #Add above two audio segments
        if (location == 'start'):    
            final_wav = silence_segSment + orig_wav
        else:
            final_wav = orig_wav + silence_segSment
        
        #Either save modified audio
        #final_song.export(audio_out_file, format="wav")
        final_wav.export(wav_filename, format="wav")
    
        #Or Play modified audio
        #play(final_wav)
     
    #############################################################################
    
    #def fix_wav_length(self, wav_filename, duration, wav_target_len, silence_ratio): 
    def fix_wav_length(self, wav_filename, ratio_to_add): 
        
       try:
           duration_before = self.get_wav_duration(wav_filename)
           
           if(duration_before > float(self._wav_target_len)):
               return
               #fix_wav_length(name, duration_before, wav_target_len, silence_ratio_percent)
               
           ratio_percentage = int(self._silence_ratio_limit_percent)/100
           #print('ratio_percentage=' + str(ratio_percentage))
    
           silence_to_add = (float(self._wav_target_len) - duration_before) * 1000
           #print("silence_to_add=" + str(silence_to_add))
           #print("silence_to_add at start=" + str((silence_to_add)*ratio_percentage))
           #print("silence_to_add at end=" + str((silence_to_add)*(1-ratio_percentage)))
    
           self.add_silence_to_wav(wav_filename, (silence_to_add)*ratio_percentage, 'start')
           self.add_silence_to_wav(wav_filename, (silence_to_add)*(1-ratio_percentage), 'end')
    
    
           duration_after = self.get_wav_duration(wav_filename)
           #print('Duration after:' + str(duration_after))
           #silence_ratio_percent = (silence_ratio_percent + 1)%80
           self._silence_ratio_limit_percent = (int(self._silence_ratio_limit_percent) + ratio_to_add)%(100-int(self._silence_ratio_limit_percent_tmp))
           #print('self._silence_ratio_limit_percent:' + str(self._silence_ratio_limit_percent))
    
           if (int(self._silence_ratio_limit_percent) < int(self._silence_ratio_limit_percent_tmp)):
               self._silence_ratio_limit_percent = int(self._silence_ratio_limit_percent) + int(self._silence_ratio_limit_percent_tmp)
       except OSError as e:
           print('In function fix_wav_length. Error: %s' % e)
            
    #############################################################################        
    
            
    def fix_target_files_duration(self):
        
        #print('================ in fix_target_files_duration ==============')
        
        for label, class_folder in enumerate(self._labels):
            
            #n_class = class_folder.split('_')[1]
            ##print('n_class=' + str(n_class) + ', label=' + str(label) + ', class folder=' + str(class_folder))
       
            target_directory_path = os.path.join(self._target_directory, class_folder)
            if not os.path.exists(target_directory_path):
                os.makedirs(target_directory_path)
            
            _, speakers, _ = next(os.walk(target_directory_path))
            
            #print('class speakers = ', speakers)
            
            for _, speaker_folder in enumerate(speakers):
       
                target_speaker_path =  os.path.join(target_directory_path, speaker_folder) 
       
                #Iterate through all the wav files in the class folder, 
                #including subfolders like in negative_ex folder
                #for filename in glob.glob('**/*.wav', recursive=True):
                for filename in os.listdir(target_speaker_path):
                    
                    #if(filename.find('bg') >= 0)):
                    if('bg' in filename):
                        continue
          
                    try:
                        inputfile_path = os.path.join(target_speaker_path, filename)
                        self.fix_wav_length(inputfile_path,7)
              
                    except OSError as e:
                        print('Error processing ' + inputfile_path + ' Error: %s' % e)
                             
    ############################################################################# 
     

#############################################################################



def main():   
    
    data_gen = data_augmentation_generator("D:\Afeka\FinalProject\DataSets\ScriptsCode\DataAugmentMng.xml")
    
    
#    
#    _, permutation, _ = next(os.walk(os.path.join("D:/Afeka/FinalProject/DataSets/Recordings/TestFolder", "TempFolderFiles")))
#            
#    #print('class permutation = ', permutation)
#            
#    for _, per_folder in enumerate(permutation):
#        
#        ###files_counter = 0
#        #print(' per_folder = ' + str(per_folder))
# 
#        orig_per_path = os.path.join("D:/Afeka/FinalProject/DataSets/Recordings/TestFolder", "TempFolderFiles", per_folder) 
#        #print(' orig_per_path = ' + str(orig_per_path))
#        
#        try:
#            dir_util.copy_tree(orig_per_path, "D:/Afeka/FinalProject/DataSets/Recordings/TestFolder")
#        except OSError as e:
#            #print('Directory not copied. Error: %s' % e)
      
    
main()
        
