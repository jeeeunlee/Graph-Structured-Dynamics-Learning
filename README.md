# gn-inverse-dynamics
Inverse dynamics Learning of [Magneto](https://research.csiro.au/robotics/paper-magneto-a-versatile-multi-limbed-inspection-robot/)

The code is developed based on
- [graph_nets](https://github.com/deepmind/graph_nets) from deepmind
- [dartpy](https://github.com/dartsim/dart) for dynamics simulation environment 

## 1. Virtual environment settings
Due to the current issue in using darpy with conda, please use python3-venv.

### Python3-venv setup
Install python3-venv package that provides the venv module.
```
$ sudo apt install python3-venv
```
Within the directory run the following command to create your new virtual environment:
(just replace "path/to/workspace" to your workspace path)
```
$ cd path/to/workspace
$ mkdir venv
$ python3 -m venv venv1
```
Activate the created venv
```
$ source venv1/bin/activate
```

### 1.[Optional-just for simulator] Install dartpy 
Please make sure to install all the dependancies before installing dartpy. 

Details can be found [here](https://dartsim.github.io/install_dartpy_on_ubuntu.html)

### 2-1 Pip list for auto install
(CPU)
```
$ pip install -r path/to/workspace/requirements.txt
```

### 2-2 Pip list for manual install
- (CPU)
```
$ pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```
- (GPU)
```
$ pip install graph_nets "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

```
$ pip install PyGeometry
$ pip install scipy
$ pip install urdfpy
```
[optional]
```
$ pip install dartpy
```

## 2.Code Explanation
- a_dataset: raw dataset / tf dataset are saved here
- a_result: learned model, validation errors, logs are saved here
- ffnn_inverse_dynamics: all codes for feed forward neural network 
- typical_gnn_inverse_dynamics: all codes for typical graph neural network
- gn_inverse_dynamics: all codes for symetry considered graph neural network
- sn_inverse_dynamics: all codes for symetry considered nerual network

### 1. tfData generation 
Dataset generation for tensorflow use, in each model representation
(e.g. ffnn-vector tensor, tgnn,gnn-graph tuple tensor, sn-group of vectors tensor)
- script : 
  run a script <a_scripts/run_dataset>
  ```
  source a_scripts/run_dataset
  ```
- runfile:

  ffnn: ffnn_inverse_dynamics/hexamagneto/main_ffnn_make_dataset.py 
  
  tgnn: typical_gnn_inverse_dynamics/robot_graph_generator/hexamagneto/main_make_dataset_gn.py
  
  sn(proposed):sn_inverse_dynamics/hexamagneto/main_sn_make_dataset.py
  
  gnn(proposed):gn_inverse_dynamics/robot_graph_generator/hexamagneto/main_make_dataset_gn.py
  
### 2. train the model / calculate test error for other data 
- script: 

  magneto: a_scripts/run_test_(ffnn/gn/tgn/sn)
  
  hexa-magneto: a_scripts/run_test_(ffnn/gn/tgn/sn)_hexa
- runfile for train

  main_train_model_(ffnn/gn/tgn/sn).py 
  
- runfile for generalization error

  main_check_model_(ffnn/gn/tgn/sn).py
