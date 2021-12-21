# gn-inverse-dynamics
Inverse dynamics Learning of [Magneto](https://research.csiro.au/robotics/paper-magneto-a-versatile-multi-limbed-inspection-robot/)

The code is developed based on
- [graph_nets](https://github.com/deepmind/graph_nets) from deepmind
- [dartpy](https://github.com/dartsim/dart) for dynamics simulation environment 

## Install dartpy
Please make sure to install all the dependancies before installing dartpy. 

Details can be found [here](https://dartsim.github.io/install_dartpy_on_ubuntu.html)

## Virtual environment settings
Due to the current issue in using darpy with conda, please use python3-venv.

### Python3-venv setup
Install python3-venv package that provides the venv module.
```
$ sudo apt install python3-venv
```
Within the directory run the following command to create your new virtual environment:
```
$ cd gn-inverse-dynamics
$ mkdir venv
$ python3 -m venv gn-inverse-dynamics
```
Activate the created venv
```
$ source gn-inverse-dynamics/bin/activate
```

### Pip list for auto install
(CPU)
```
$ pip install -r path/to/requirements.txt
```

### Pip list for manual install
(CPU)
```
$ pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```
(GPU)
```
$ pip install graph_nets "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

```
$ pip install PyGeometry
$ pip install scipy
$ pip install urdfpy
$ pip install dartpy
```

## Code Explanation
- main_train_model.py : training the model
- main_make_dataset.py : generate graph tfdataset from robot configure file and trajectory files
- main_check_model.py : compute error of the trained model for validation/test dataset

- magneto_main.py : run the simulator
- magneto_world_node.py : node for simulator