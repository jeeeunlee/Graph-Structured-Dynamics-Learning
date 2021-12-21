from time import localtime, strftime
import os
import time
import shutil

### util functions

## list shaping
def shuffle_list(list_in):
    print(type(list_in))
    list_out = []
    if(len(list_in)>0):
        list_out = [list_in[-1]]
        list_out.extend(list_in[0:len(list_in)-1])
    return list_out

def shuffle_list_reverse(list_in):
    list_out = []
    if(len(list_in)>0):        
        list_out.extend(list_in[1:len(list_in)])
        list_out.append(list_in[0])
    return list_out

def get_drop_down_list(list_in, act_idx):
  return [ list_in[i] for i in act_idx ]


# FILE I/O UTILS
def string_to_number(str):
  # if("." in str):
  #   try:
  #     res = float(str)
  #   except:
  #     res = str  
  # elif("-" in str):
  #   res = int(str)
  # elif(str.isdigit()):
  #   res = int(str)
  # else:
  #   res = str
  res = float(str)
  return(res)

def string_to_list(str):
  return [string_to_number(element) 
          for element in str.split()]

## LOG UTIL FUNCTIONS
def get_local_time():
  return strftime("%Y-%m-%d--%H:%M:%S", localtime())

def log_with_time(logmsg):
    logt = time.time()
    print("[{:.4f}] {}".format(logt, logmsg) )

def create_folder(directory):
  try:
      if not os.path.exists(directory):
          os.makedirs(directory)
  except OSError:
      print ('Error: Creating directory. ' +  directory)

def delete_and_create_folder(directory):
  try:
      if not os.path.exists(directory):
        os.makedirs(directory)
      else:
        print("path exists : delete all dataset")
        shutil.rmtree(directory)
        os.makedirs(directory)      
      print("directory made @ " + directory)
  except OSError:
      print ('Error: Creating directory. ' +  directory)

### util class
class Logger():
  def __init__(self, filename):
    self._filename = filename
    self._f =  open(self._filename, 'w')

  def record_time_loss(self, time, loss):
    with open(self._filename, 'a') as self._f:
      self._f.write(str(time))
      self._f.write(', ')
      self._f.write(str(loss.numpy()))
      self._f.write('\n')

  def record_string(self, str_input):
    with open(self._filename, 'a') as self._f:
      self._f.write(str_input)
      self._f.write('\n')

  def record_list(self, list_input):
    if(list_input is None):
      return
    with open(self._filename, 'a') as self._f:
      for list_element in list_input:
        self._f.write(str(list_element))
        self._f.write(", ")
      self._f.write('\n')

  def record_val(self, val_input):
    with open(self._filename, 'a') as self._f:
      self._f.write(str(val_input))
      self._f.write('\n')

  def get_file(self):
    self._f = open(self._filename, 'a')
    return self._f
  
  def close_file(self):
    self._f.close()

class Timer():
  def __init__(self):
    self._start_time = time.time() 

  def reset(self):
    self._start_time = time.time()

  def elapsed(self):
    self._elapsed_time = time.time() - self._start_time
    return self._elapsed_time

  def check(self, check_time):
    if(self.elapsed() > check_time):
      self.reset()
      return True
    else:
      return False

class IterCounter():
    def __init__(self):
        self._iter = 0
        self._check_iter = 0

    def reset(self):
        self._iter = 0
        self._check_iter = 0

    def get_count(self):
        return self._iter  
    
    def count(self):
        self._iter += 1
        self._check_iter += 1
        return self._iter    

    def check(self, check_iter):
        if( self._check_iter >  check_iter):
            self._check_iter = 0
            return True
        else:
            return False

    def print_iter(self, str_input=''):
        print("{} iter : {}".format(str_input, self._iter))

    def count_check_print(self, check_iter):
        self.count()
        if( self._check_iter >=  check_iter):
            self._check_iter = 0
            self.print_iter()
            return True
        else:
            return False