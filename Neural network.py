import csv
import math
import numpy as np
from numpy import exp, array, random, dot
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
mnist_train='mnist_train.csv'      # path for training data
mnist_test='mnist_test.csv'        # path for test data

## Gloabal variables
global x,x_t,t,y,Sum,T_predicted,acc_training,target,target_t,acc_testing,predicted_digit

def get_weights(k,l):                                           # Generate a random values between -0.05 and 0.05
    w=[]                                                        # for 7850 weights    
    for i in range(k):
        a=[]
        for j in range(l):
            a.append(random.uniform( -0.05, 0.05 ))
        w.append(a)                                             # append the random generated value to list
    w=np.array(w)                                               # convert that list into numpy array 
    return w

def get_dataset(file):
    with file as f:
        csv_reader=csv.reader(f)                        # read the csv file of training and test data
        x=[]                                            # for extracting the input values and target value
        target=[]
##        for j,row in enumerate(csv_reader):           # condition for experiment 3
##            i=0
##            if(j==15000):
##                break
            X=[1]                                       # add bias for input layer
            
            for col in row:                             # if its first element then it is target value
                if(i==0):                               
                    target.append(int(col))             # append target value to list
                    i=1
                else:                                   # else it is input values
                    X.append(float(col)/255)            # append input values to list 
            x.append(X)
    x=np.array(x)                                       # convert the both list into numpy array                                    
    target=np.array(target)
    return x, target

def training_testing():                                 # training for 60000 data and testing for 10000 data
    momentum=0.9                                        # set momentum 
    eta= 0.1                                            # eata value for weight update(learning rate)   
    acc_training=[]                                     # list to store the accuracy after each epoch 
    acc_testing=[]          
    #hidden_layer=[20,50,100]                           # different hidden units for experiment 1
    h_units=100                                         # set the hidden units
    #momentum_list=[0,0.25,0.5]                         # different momentum valuse for experiment 2
    k=0
    #for momentum in momentum_list:                                       
    test_acc=[]                                             
    train_acc=[]
    print("caculateion for momentum: ",momentum)  
    w= get_weights(h_units,len(x[0]))                   # initailize the weights for neural network
    w1= get_weights(10,h_units+1)
    if(k==0):   
        M=acc_calculate(x,target,w,w1)                  # calculate the accuracy before training   
        acc=(M/60000)*100
        k=1
    print(acc)
    test_acc.append(acc)                            # append that accuracy to both traiing and test list to compare 
    train_acc.append(acc)
                
    for b in range(50):                             # set data for 50 epoch
        print("epoch number :",b)
        a=0
        delta_w=np.zeros((h_units,785))             # initailize the differential weights to zero
        delta_w1=np.zeros((10,h_units+1))

        for inp in x:                                                               # Input from dataset  
            t=np.array([0.1 for i in range(10)])                                          
            t[target[a]]=0.9                                                        # set target value  for perceptron
            Sum=summation(w,inp)                                                    # caculate the summation for input to hidden layer                             
            h_output=np.append([1],sigmoid(Sum),axis=0)                             # add bias to hidden layer
            Sum_1=summation(w1,h_output)                                            # calculate the summation for hidden to output layer
            output=sigmoid(Sum_1)
            if(np.array_equal(t,output)==False):                                    # if target is not equal to predicted outiut then
                delta_o,delta_h=cal_error(output,h_output,t,w1)                     # calculate the error unction
                delta_h=np.delete(delta_h,0)
                delta_w1=update_weights(h_output,w1,delta_o,delta_w1,eta,momentum)  # update the weights for hidden to output layer
                w1=w1+delta_w1                                                      # update the weights
                delta_w=update_weights(inp,w,delta_h,delta_w,eta,momentum)          # update the weights for input to hidden layer
                w=w+delta_w                                                         # update the weight
            a=a+1
        predicted_op=acc_calculate(x,target,w,w1)                                   # after completing the each epoch caculate the accuracy 
        acc_train=(predicted_op/60000)*100                                          # for training data            
        train_acc.append(acc_train) 
        predicted_op_test=acc_calculate(x_t,target_t,w,w1)                          # caculate the accuracy of testing data
        acc_test=(predicted_op_test/10000)*100
        test_acc.append(acc_test)
        
        
    caclulate_confusion_mat(x_t,target_t,w,w1)              # After 50 epoch caclulate the confusion matrix for particuler eata
    acc_testing.append(test_acc)                            # append that accuracy to both traiing and test list to compare 
    acc_training.append(train_acc)
 
    return acc_training,acc_testing

def summation(Weights,Input):                               # function for calculation of dot prodoct of weight and inputs
    return np.dot(Weights,np.transpose(Input))

def sigmoid(i):                                             # function for calculation of sigmoid activation function
    return 1 / (1 + exp(-i))    

def cal_error(output,h_output,t,w):                                                 # function for calulating the error
    delta_output=np.array([])
    delta_h=np.array([])
    delta_output=output*(1-output)*(t-output)                                       # error calculation for output layer
    delta_h=h_output*(1-h_output)*(np.dot(np.transpose(w),delta_output))            # error calculation for hidden layer
    
    return delta_output,delta_h

def update_weights(inputs,weight,delta,delta_w,eta,momentum):                       # function for weight udpdate
    delta_updated_w=[]
    for i in range(len(weight)): 
        delta_updated_w.append((eta*delta[i])*inputs+momentum*np.transpose(delta_w[i]))
    delta_updated_w=np.array(delta_updated_w)
    return delta_updated_w

def acc_calculate(x,target,w,w1):                               # function for caulating the accuracy
    a=0
    T_predicted=0
    for inp in x:                                               # accuracy calculation    
        Sum=summation(w,inp)                                    # checking each perceptron for prediction
        h_output=np.append([1],sigmoid(Sum),axis=0)
        Sum_1=summation(w1,h_output)
        output=sigmoid(Sum_1)
        index = np.argmax(output)                           # find a preddicted digit
        if(index==target[a]):                               # if dot product of weights and input of the perceptron is max  
            T_predicted= T_predicted+1                      # and target is match with it then increment the accuracy count
        a=a+1
                         
    return T_predicted

def caclulate_confusion_mat(x,target,w,w1):                     # function to display confusion matrix 
    predicted_digit=[]      
    for inp in x:                                               # accuracy calculation    
        Sum=summation(w,inp)                                    # checking each perceptron for prediction
        h_output=np.append([1],sigmoid(Sum),axis=0)
        Sum_1=summation(w1,h_output)
        output=sigmoid(Sum_1)
        index = np.argmax(output)                               # find a preddicted digit
        predicted_digit.append(index)
    r=confusion_matrix(target, predicted_digit)
    print(r)                                                    # display the confusion matrix for actual digit and predicted digit
    return 0    

file_1=open(mnist_train,'r')                                # open training file for reading
x,target=get_dataset(file_1)                                # get the target values and inputs
file_1.close()
file_2=open(mnist_test,'r')                                 # open the test data file for reading
x_t, target_t=get_dataset(file_2)                           # get the target and input values
file_2.close()
acc_training,acc_testing= training_testing()                # start training the perceptrons and testing it
print(acc_training)
print(acc_testing)                                          # print accuracy 
for i in range(3):                                          # display the graph of epoch vs accuracy for each eta
    plt.xlabel('epoch')                                     # set x lable of graph
    plt.ylabel('Accuracy(%)')                               # set y lable of graph
    plt.yticks(np.arange(0, 101, 5))
    plt.plot(acc_testing[i])                                # ploat the accuracy on graph
    plt.plot(acc_training[i])
    plt.legend(['Accuracy on the test data','Accuracy on the training data'], loc='lower right')   
    plt.show()                                              # show the graph

