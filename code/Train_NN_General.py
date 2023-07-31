# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Mar 11 22:38:20 2019

@author: yanivg
"""

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import time
import os
import os.path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


xml_filepath = 'D:\Afeka\FinalProject\Models\DNN\ScriptsCode\DnnModel_ParamMng.xml'
#xml_filepath = 'ParamMng_Linux.xml'


def main_train():
      
    print('================  START ==============')
    
    Test_Number,Train_Extract_Features_Directory,Test_Extract_Features_Directory,Test_Result_Directory,training_epochs,n_classes,optimizerType,learning_rate,n_hidden_units_one,n_hidden_units_two,n_hidden_units_three,hidden_one_actF,hidden_two_actF,hidden_three_actF = load_parameters(xml_filepath)
    print(" MAIN Train_Extract_Features_Directory:" , Train_Extract_Features_Directory)
    print(" MAIN Test_Extract_Features_Directory:" , Test_Extract_Features_Directory)
    #os.chdir(ExtractFeatures_Directory)
    
    
    train_x = []
    train_y = []
    test_x = []
    test_y =[]
    
   
    train_x = load_npy(os.path.join(Train_Extract_Features_Directory, 'X.npy'))
    #features = load_npy('D:\Afeka\FinalProject\Models\DNN\ExtractFeatures\DataAugmentFeatures\X.npy')
    #labels = load_npy('D:\Afeka\FinalProject\Models\DNN\ExtractFeatures\DataAugmentFeatures\Y.npy')
    train_y = load_npy(os.path.join(Train_Extract_Features_Directory, 'Y.npy'))
    print("train_y[2:5]:" , train_y[2:5])
    #print("train_y[50000:50005]:" , train_y[50000:50005])
    
    
    test_x = load_npy(os.path.join(Test_Extract_Features_Directory, 'X.npy'))
    #features = load_npy('D:\Afeka\FinalProject\Models\DNN\ExtractFeatures\DataAugmentFeatures\X.npy')
    #labels = load_npy('D:\Afeka\FinalProject\Models\DNN\ExtractFeatures\DataAugmentFeatures\Y.npy')
    test_y = load_npy(os.path.join(Test_Extract_Features_Directory, 'Y.npy'))
    print("test_y[2:5]:" , test_y[2:5])
    #print("test_y[50000:50005]:" , test_y[50000:50005])
    
    print("hidden_one_actF:" , hidden_one_actF)
    print("hidden_two_actF:" , hidden_two_actF)
    print("hidden_three_actF:" , hidden_three_actF)
    
    print("n_hidden_units_one:" , n_hidden_units_one)
    
    # =============================================================================
   
    #n_dim = features.shape[1]   #193
    n_dim = train_x.shape[1]   #193
    sd = 1 / np.sqrt(n_dim)   #0.072
    
    tf.reset_default_graph()
    
    # =============================================================================
    
    print("Test_Result_Directory:" , Test_Result_Directory)
    if not os.path.exists(Test_Result_Directory):
        os.makedirs(Test_Result_Directory)
        
        
#    train_x = []
#    train_y = []
#    test_x = []
#    test_y =[]
#    
#    train_test_split = np.random.rand(len(features)) < 0.70
#    train_x = features[train_test_split]
#    train_y = labels[train_test_split]
#    test_x = features[~train_test_split]
#    test_y = labels[~train_test_split]
    
    print('train_x Shape: ',train_x.shape)
    print('train_y Shape: ',train_y.shape)
    print('test_x Shape: ',test_x.shape)
    print('test_y Shape: ',test_y.shape)
    
    
    print('train_y : ',train_y)
    print('test_y : ', test_y)
    
    test_x_file = os.path.join(Test_Result_Directory, 'test_x.npy')
    np.save(test_x_file, test_x)
    test_y_file = os.path.join(Test_Result_Directory, 'test_y.npy')
    np.save(test_y_file, test_y)
    
    # X is a matrix of shape [none, n_dim=193 features]
    X = tf.placeholder(tf.float32,[None,n_dim], name='X')
    # Y is a matrix of shape [none, n_classes=5]
    Y = tf.placeholder(tf.float32,[None,n_classes])
    
    #tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd)
    
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))   # W_1 = [193,280]
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))         # b_1 = [1,280]
    
    # z_1 = [none,280]    none = number of samples (Rows in X which is the rows at train_x\test_x)
    if hidden_one_actF=='tangent':
        z_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)
    elif hidden_one_actF=='relu':
        z_1 = tf.nn.relu(tf.matmul(X,W_1) + b_1)
    elif hidden_one_actF=='sigmoid':
        z_1 = tf.nn.sigmoid(tf.matmul(X,W_1) + b_1)
    else: 
        z_1 = tf.nn.sigmoid(tf.matmul(X,W_1) + b_1)

    
    
    print('\z_1 Shape: ',z_1.shape)
    
    
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))      # W_2 = [280,300]
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))                         # b_2 = [1,300]
      
    # z_2 = [none,300]     
    if hidden_two_actF=='tangent':
        z_2 = tf.nn.tanh(tf.matmul(z_1,W_2) + b_2)
    elif hidden_two_actF=='relu':
        z_2 = tf.nn.relu(tf.matmul(z_1,W_2) + b_2)
    elif hidden_two_actF=='sigmoid':
        z_2 = tf.nn.sigmoid(tf.matmul(z_1,W_2) + b_2)
    else: 
        z_2 = tf.nn.sigmoid(tf.matmul(z_1,W_2) + b_2)                                                 
    
    print('\z_2 Shape: ',z_2.shape)
    
    
    if hidden_three_actF!='null':
        W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
        b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
        #z_3 = tf.nn.sigmoid(tf.matmul(z_2,W_3) + b_3)
        
        if hidden_three_actF=='tangent':
            z_3 = tf.nn.tanh(tf.matmul(z_2,W_3) + b_3)
        elif hidden_three_actF=='relu':
            z_3 = tf.nn.relu(tf.matmul(z_2,W_3) + b_3)
        elif hidden_three_actF=='sigmoid':
            z_3 = tf.nn.sigmoid(tf.matmul(z_2,W_3) + b_3)
        else: 
            z_3 = tf.nn.sigmoid(tf.matmul(z_2,W_3) + b_3)
        
        print('\z_3 Shape: ',z_3.shape)
        
        
        W = tf.Variable(tf.random_normal([n_hidden_units_three,n_classes], mean = 0, stddev=sd), name='W_var')   # W = [300,5]
        b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))                        # b = [1,5]
        y_ = tf.nn.softmax(tf.matmul(z_3,W) + b, name='y_softmax_op')  
        print("3 layers - saving y_softmax_op")
                                                       # y_ = [none,5]
    else:
        W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd), name='W_var')   # W = [290,5]
        b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))                        # b = [1,5]
        y_ = tf.nn.softmax(tf.matmul(z_2,W) + b, name='y_softmax_op') 
        print("2 layers - saving y_softmax_op")
       
    
 
    
    #print("22222222")
    # reduce_sum = Computes the sum of elements across dimensions
    # reduce_mean = Computes the mean of elements across dimensions
    #  reduction_indices = 1  => reduce sum by rows
    cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) 
    #print("33333333")
    
    if optimizerType == 'Gradient Descent':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
        print("Gradient Descent")
    elif optimizerType == 'Adam Optimizer':
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_function)
        print("Adam Optimizer")
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
    print("44444444")
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # argmax = Returns the index with the largest value across axes
    # tf.argmax(y_,1) return for each row the index where the value is 1 (true)
    # tf.equal return true where the index of true is equal for y and y_ (y hot)
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
    
#    saver = tf.train.Saver()
    
    print("55555555")
    
    cost_history = np.empty(shape=[1],dtype=float)
    y_true, y_pred = None, None
    
    #print("66666666")
    
    
    
    saver = tf.train.Saver()
    #with tf.Session(graph=new_graph) as sess:
    with tf.Session() as sess:
        
        sess.run(init)
        #print("8888888888")
        for epoch in range(training_epochs):   
            #print("999999999")
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
            cost_history = np.append(cost_history,cost)
            if epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, cost))
        
        print('\cost_history : ',cost_history)
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
        print("y_pred:" , y_pred)
        print("y_pred[2]:" , y_pred[2])
        print("y_pred[233]:" , y_pred[30])
#        print("y_pred[17777]:" , y_pred[17777])
#        print("y_pred[9877]:" , y_pred[9877])
#        print("y_pred[11111]:" , y_pred[11111])
#        print("y_pred[4341]:" , y_pred[4341])
        
        print("y_[2,:]:" , y_[2,:])
        #print("y_[5000:5005]:" , y_[5000:5005])
        y_true = sess.run(tf.argmax(test_y,1))
    
        print("y_true[2]:" , y_true[2])
        print("y_true[233]:" , y_true[30])
#        print("y_true[17777]:" , y_true[17777])
#        print("y_true[9877]:" , y_true[9877])
#        print("y_true[11111]:" , y_true[11111])
#        print("y_true[4341]:" , y_true[4341])
        print("y_true:" , y_true)
        
        
        print("W_var = ", sess.run(W))
        #print("y_ = ", sess.run(y_))
        

        saver.save(sess, os.path.join(Test_Result_Directory,'my_test_model' + Test_Number))
        save_model(saver, sess, Test_Result_Directory, y_pred, y_true, cost_history, Test_Number)
        
        
    display_result("Test" + Test_Number, cost_history, training_epochs, accuracy, y_true, y_pred, Test_Result_Directory)
        
    print("99999999")
    


# =============================================================================
    

def load_parameters(xml_path):
    
    
    print('================ in load parameters ==============')
    
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()

    # find the first 'item' object
    for model in root.findall('Model'):
        #Test_Number = model.attrib['Test_Number']
        Test_Number = model.get('Test_Number')
        print("Test_Number=", Test_Number.strip())
        Train_Extract_Features_Directory = model.find('Directories').get('Train_Extract_Features_Directory')
        print("Train_Extract_Features_Directory=",Train_Extract_Features_Directory.strip())
        Test_Extract_Features_Directory = model.find('Directories').get('Test_Extract_Features_Directory')
        print("Test_Extract_Features_Directory=",Test_Extract_Features_Directory.strip())
        Test_Result_Directory = model.find('Directories').get('Test_Result_Directory')
        print("Test_Result_Directory=",Test_Result_Directory.strip())
        
        #  Get HyperParameters
        for hyperparameter in model.findall('HyperParameters'):
            #General Parameters
            training_epochs = int(hyperparameter.find('General').get('training_epochs').strip())
            print("training_epochs=",training_epochs)
            n_classes = int(hyperparameter.find('General').get('num_classes').strip())
            print("n_classes=",n_classes)
            optimizerType = hyperparameter.find('General').get('optimizerType').strip()
            print("optimizerType=",optimizerType)
            learning_rate = float(hyperparameter.find('General').get('learning_rate').strip())
            print("learning_rate=",learning_rate)
            
            #Number Hidden Units Parameters
            n_hidden_units_one = int(hyperparameter.find('NumberHiddenUnits').get('hidden_1').strip())
            print("n_hidden_units_one=",n_hidden_units_one)
            n_hidden_units_two = int(hyperparameter.find('NumberHiddenUnits').get('hidden_2').strip())
            print("n_hidden_units_two=",n_hidden_units_two)
            n_hidden_units_three = int(hyperparameter.find('NumberHiddenUnits').get('hidden_3').strip())
            print("n_hidden_units_three=",n_hidden_units_three)
            
            #Hidden Activation Function Parameters
            hidden_one_actF = hyperparameter.find('HiddenActivationFunction').get('hidden_1').strip()
            print("hidden_one_actF=",hidden_one_actF)
            hidden_two_actF = hyperparameter.find('HiddenActivationFunction').get('hidden_2').strip()
            print("hidden_two_actF=",hidden_two_actF)
            hidden_three_actF = hyperparameter.find('HiddenActivationFunction').get('hidden_3').strip()
            print("hidden_three_actF=",hidden_three_actF)
            
    Test_Result_Directory =  Test_Result_Directory + Test_Number
    print("######Test_Result_Directory=",Test_Result_Directory)
          
    return Test_Number,Train_Extract_Features_Directory,Test_Extract_Features_Directory,Test_Result_Directory,training_epochs,n_classes,optimizerType,learning_rate,n_hidden_units_one,n_hidden_units_two,n_hidden_units_three,hidden_one_actF,hidden_two_actF,hidden_three_actF
    

# =============================================================================


def save_model(saver, sess, test_directory, y_pred, y_true, cost_history, test_number):
    
    print("save_model")
    
    
    #saver.save(sess, os.path.join(test_directory,'my_test_model' + test_number))
    
    print("y_pred types:" , str(type(y_pred)))
    print('\ny_pred Shape: ',y_pred.shape)
    print("y_true types:" , str(type(y_true)))
    print('\y_true Shape: ',y_true.shape)
    print("cost_history types:" , str(type(cost_history)))
    print('\cost_history Shape: ',cost_history.shape)
    
    y_pred_file = os.path.join(test_directory, 'y_pred.npy')
    y_true_file = os.path.join(test_directory, 'y_true.npy')
    cost_history_file = os.path.join(test_directory, 'cost_history.npy')
    
    np.save(y_pred_file, y_pred)
    print ("Saved " + y_pred_file)
    np.save(y_true_file, y_true)
    print ("Saved " + y_true_file)
    np.save(cost_history_file, cost_history)
    print ("Saved " + cost_history_file)
    
    
    

# =============================================================================


def display_result(testName, cost_history, training_epochs, accuracy, y_true, y_pred, test_directory):
       
    print("display_result")
    
    
    s_font_size=25
    s_label_size=20

    plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.title(testName,fontsize=s_font_size)
    plt.ylabel("Cost",fontsize=s_font_size)
    plt.xlabel("Iterations",fontsize=s_font_size)
    #plt.axis([0,training_epochs,0,np.max(cost_history)], labelsize=BIGGER_SIZE)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    

    #plt.rc('font', size=s_label_size)          # controls default text sizes
    #plt.rc('axes', titlesize=s_label_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=s_label_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s_label_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=s_label_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=s_label_size)    # legend fontsize
    #plt.rc('figure', titlesize=s_label_size)  # fontsize of the figure title
    plt.show()
    
    
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print("accuracy: ", accuracy)
    print ("Precision: ", str(p))
    print ("recall: ", str(r))
    print ("F-Score: ", str(round(f,3)))
    print ("support: ", str(s))
    accurate = get_model_accurate(test_directory)
    print("accurate: ",accurate)
 
# =============================================================================


def load_npy(path):
    
    t1=time.time()
    array_reloaded = np.load(path)
    t2=time.time()
    
    print("Time took to load:" , {t2-t1}, "seconds.")
    print('****************************************')
    print('\nShape: ',array_reloaded.shape)
 
    return array_reloaded

# =============================================================================


def get_model_accurate(path):
    
    #for Windows
    #y_pred = load_npy(path + "\y_pred.npy")
    #y_true = load_npy(path + "\y_true.npy")
    
    #for Linux
    y_pred = load_npy(path + "/y_pred.npy")
    y_true = load_npy(path + "/y_true.npy")
    
    len_pred = len(y_pred)
    print("y_pred len:" , str(len_pred))
    len_true = len(y_true)
    print("y_true len:" , str(len_true))
    
    #print(y_pred[586])
    true_pred_counter = 0
    
    for x in range(len_pred):
        if(y_pred[x] == y_true[x]):
            true_pred_counter = true_pred_counter + 1
    
    return true_pred_counter/len_pred
        


#y_pred, y_true = main()  
main_train()




