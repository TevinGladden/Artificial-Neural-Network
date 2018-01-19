# Language:       python2
# Creator:        Tevin Gladden
# Project:        Artificial Neural Network
# <Summary>
# This ANN uses a data set of 20000 feature vectors, each with 17 features.
# Each instance is a representation of hand-written letters, where feature 0 is the classification(letter)
# 75 percent of the data set goes into a training set and the remaining 25 percent to a test set.
# The ANN is set to train until it's prediction accuracy reaches 93 percent

import numpy as np
import random

class ANN:
    def __init__(self):
        self.counter = 0
        #input layer inputs
        self.inputs = []
        self.test = []
        #layer outputs
        self.o1 = []
        self.o2 = []
        #activation function outputs
        self.a1 = []
        self.a2 = []
        #weights at each layer
        self.w1 = []
        self.w2 = []
        self.neurons = 151
        self.classification = []
        self.yout = 0
        self.learningRate = 0.1
        self.accuracy = 0.0
        
    def input_data(self):
        data = open("letters1.txt", mode = 'r')
        counter = 0
        inputArray = []
        classification = []
        for line in data:
            inputArray.append(line.strip("\n").split(","))
            classification.append(line[0])
            del(inputArray[counter][0])
            inputArray[counter] = map(lambda x: float(x)/15, inputArray[counter])
            inputArray[counter].append(1.0)           
            counter += 1
        self.classification = np.array(map(lambda x: ord(x)-65, classification))
        self.inputs = np.array(inputArray)
        self.test = self.inputs[14999:19999]
        
    def weights(self):
        for i in range(self.neurons+1):
            if i < 17:
                self.w1.append(map(lambda x: float(x+random.random()), np.zeros(self.neurons)))
            self.w2.append(map(lambda x: float(x+random.random()), np.zeros(26)))  
        self.w1 = np.array(self.w1)
        self.w2 = np.array(self.w2)
        
    def forward(self, array):
        #calculations for input and layer 1 weights.
        self.o1 = np.append(np.dot(array[self.counter], self.w1), 1.0)
        #activation function applied to output of previous calculation
        self.a1 = self.sigmoid(self.o1)
        self.a1[-1] = 1.0
        self.a1.reshape(1,self.neurons+1)   
        #calculations for layer2 weights and activated output of layer 1
        self.o2 = np.dot(self.a1.T, self.w2)
        self.a2, self.yout = self.softmax()        
 
    def update(self):
        array = np.zeros(26)
        array[self.classification[self.counter]] = 1
        
        d2 = np.dot(-1, np.subtract(array, self.a2).reshape(26,1).T)
        D2 = np.dot(self.a1.T.reshape(self.neurons+1,1), d2)
        
        d1 = ((np.dot(d2, self.w2.T))[0])*self.sigmoidPrime(self.o1)
        D1 = np.dot(self.inputs[self.counter].reshape(17,1), d1.T.reshape(1,self.neurons+1))      
      
        n = np.delete(D1, np.s_[self.neurons::], 1) # refrain from updating bias weights by removing their errors.
        self.o1 = np.append(self.o1, 1.0).T
        self.w1 -= self.learningRate*n
        self.w2 -= self.learningRate*D2
    
    #sigmoid activation function
    def sigmoid(self, X):       
        return np.array(map(lambda x: 1/(1+np.exp(-x)), X))
    
    #derivative of the sigmoid activation function for backpropogation
    def sigmoidPrime(self, X):
        return np.exp(-X)/((1+np.exp(-X))**2)
    
    #softmax activation function to determine the classification by selecting the maximum of the 26 neurons output layer.
    def softmax(self):
        array = np.exp(self.o2)/np.sum(np.exp(self.o2))
        letter = np.argmax(array)
        return array, letter    
    
    def classify(self):
        actual = 0
        epoch = 1        
        while self.accuracy < .98:  
            self.counter = 0
            self.classifyTest()
            correct = 0            
            while self.counter <= 14999:
                self.forward(self.inputs)
                actual = self.classification[self.counter]
                if self.yout == actual:
                    correct += 1
                else: 
                    self.update()
                self.counter += 1
            print "Epoch: " + str(epoch)            
            epoch += 1            
                    
    def classifyTest(self): #specify training set in forward function 
        actual = 0
        correct = 0
        self.counter = 0        
        while self.counter <= 4999:
            self.forward(self.test)
            actual = self.classification[self.counter+14999]
            if self.yout == actual:
                correct += 1
            self.counter += 1    
        self.accuracy = float(correct)/self.counter 
        print("Test Accuracy: "+str(self.accuracy)+"\n")      
           
    def run(self):
        self.input_data()
        self.weights()
        self.classify()
        
ANN().run()
