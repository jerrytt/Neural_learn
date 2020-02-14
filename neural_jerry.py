#!/usr/bin/python
import glob
import re
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

class neural_alpha():
  def __init__(self):
    self.descript='just a demo'
    self.batch_size=5
    self.epochs=5
    self.validation_split=0.1
    self.verbose=1

  def set_verbose(self,verbose):
    self.verbose=verbose

  def set_weight_nr(self,weight_nr):
    self.weight_nr = weight_nr

  def set_batch_size(self,batch_size):
    self.batch_size=batch_size

  def set_epochs(self,epochs):
    self.epochs=epochs
 
  def fill_data_x(self,x):
    self.x_data = x

  def fill_data_y(self,y):
    self.y_data = y

  def build_sigmoid(self):
    self.model = keras.Sequential([
      layers.Dense(20, activation='relu',input_shape=[self.weight_nr,]),
      layers.Dense(10, activation='relu'),
      layers.Dense(1, activation='sigmoid'),
    ])
    self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

  def build_softmax(self):
    self.model = keras.Sequential([
      layers.Dense(30, activation='relu',input_shape=[self.weight_nr,]),
      layers.Dense(40, activation='relu'),
      layers.Dense(3, activation='softmax'),
    ])
    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

  def training(self):
    self.model.fit(self.x_data, self.y_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1, verbose=self.verbose)

  def summary(self):
    self.model.summary()

  def eval(self):
    self.model.evaluate(self.x_data,self.y_data)

  def predict(self,x):
    print("predict: {}".format(self.model.predict(x)))

def load_data_in_dir(data_dir,reg_x):
  file_x = glob.glob('./'+data_dir+'/'+reg_x)
  x_data=np.zeros(shape=[len(file_x),4])
  y_data=np.zeros(shape=[len(file_x),1])
  print("x:shape {} ; y:shape {}".format(x_data.shape,y_data.shape))
  #print('{}'.format())

  file_index = 0
  unmatch = 0
  for log_file in file_x:
    print('processing file:{}'.format(log_file))
    file_index += 1
    with open(log_file,'r') as x_axis:
      verify_match = 0
      for line in x_axis:
        print('line {} in {}'.format(line,log_file))
        #array2 11 12 13 14
        match_r = re.search('array2 (\d+) (\d+) (\d+) (\d+) (\d+)',line)
        if(match_r):
          print('found the params {}'.format('no'))
          (p1,p2,p3,p4,p5) = (match_r.group(1),match_r.group(2),match_r.group(3),match_r.group(4),match_r.group(5))
          try:
            print('try to assign the value {} {} {} {} into array index:{} '.format(p1,p2,p3,p4,file_index-1))
            x_data[file_index-1-unmatch] = [ p1,p2,p3,p4 ]
          except Exception as e:
            print('this is no good {}'.format(e))
            break 
          ######handle y lable here here here########
          ###
          y_data[file_index-1-unmatch] = [ p5 ]
          ###
          ######handle y lable here here here########
          verify_match=1
          break
      if(verify_match != 1):
        unmatch += 1
  x_data=x_data[:len(x_data)-unmatch]
  y_data=y_data[:len(x_data)-unmatch]
  print(x_data,y_data)
  return x_data,y_data
        
      

if (__name__ == '__main__'):
  
  learn = neural_alpha()
  learn.set_epochs(100)
  learn.set_batch_size(2)

  #x_data=np.zeros(shape=[])
  #y_data=np.zeros(shape=[])
  #x_data = np.random.normal(0,100,(100,10))
  #y_data = np.random.randint(0,3,(100,1))
  x_data,y_data = load_data_in_dir('test','*log_*')
  learn.set_weight_nr(x_data.shape[1])
  learn.fill_data_x(x_data)
  learn.fill_data_y(y_data)

  #learn.build_sigmoid()
  learn.build_softmax()
  learn.set_verbose(0)
  learn.training()
  learn.summary()
  learn.eval()
  print('00000000000000000000000000000000')
  learn.predict(x_data[0:1])



