# Pixel2Plan
This project aims to solve the game of Tangram using reinforcement learning. The game itself is implemented using the
PyBullet physics simulator. Our approach is based on the Deep Deterministic Policy Gradient (DDPG) with Hindsight
Experience Replay. The project is implemented for MPI and Cuda to allow for increased training performance.
## Getting Started
### Prerequisites
This project requires
[openmpi](https://www.open-mpi.org/software/ompi/v4.0/ "Download Openmpi")
and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

##### Install this project
Install virtual environment
```bash
$ conda env create -f pixel2plan.yml
```
Navigate to pixel2plan ../pixel2plan/ and activate conda environment
```bash
$ conda activate pixel2plan
```
or
```bash
$ source activate pixel2plan
```

## Project Structure
```sh
pixel2plan/
    main_mpi.py
    tester.py
    Readme.md
    setup.py
    demo.py
    job_script.sh
    environment/
        targets/
        urdf/
        __init__.py
        gametoken.py
        tangram.py
    models/
        __init__.py
        actor_cnn.py
        actor_critic.py
        conv_model.py
        critic_cnn.py
        replaybuffer.py
    runs/
        ...
        data/
            ...
    simulation/
        __init__.py
        dispatcher.py
        initMPI.py
        master.py
        masterMPI.py
        worker.py
        workerMPI.md
    utilities/
        __init__.py
        argumentparser.py
        ddpg.py
        settings.json
        utilities.py
```

## Usage
The project runs as follows. During training at least three cores are required. The first core is reserved for the 
master leading the whole training process. Is set to Cuda as default but can be changed to cpu in "main_mpi.py". The
master runs the algorithm updates and contains the current policies for actors and critic. The second core reserved for
the dispatcher to accept and distribute jobs from the master to the workers to gather training data. Every additional
core is occupied by a single worker each to gather training data. Workers run on CPU by default for highest training 
speed in our setup. 

### Training
Specify all parameters required in settings.json. Training can be started using
```bash
$ mpiexec -np <cores> python -u main_mpi.py
```
Please ensure that the "max_pending_sims" parameter is not lower than the specified amount of cores-2 as it otherwise 
blocks the usage of some cores. A run can be restarted after it succesfully terminated using
```bash
$ mpiexec -np <cores> python -u main_mpi.py --load=<run name>
```
Here <run name> is relative to the pixel2plan/runs/ folder. If the specified run does not exist, the run at the bottom
of the runs folder is used as default. <br />

### Debugging

PyCharm Debugging can be started using the `--debug` flag:
```bash
$ mpiexec -np <cores> python -u main_mpi.py  --debug
```

Specify the ports of the PyCharm Debug Servers inside `main_mpi.py`.

#### Setup 

To debug the project when run with multiple MPI threads there is a way using PyCharm Professional:

https://stackoverflow.com/questions/57519129/how-to-run-python-script-with-mpi4py-using-mpiexec-from-within-pycharm

Note that for PyCharm Professional 2022 that "Python Remote Debug" was renamed to "Python Debug Server"

### Demo
The demo is designed to show the performance of a specified policy. Before executing the demo please specify the correct
token in settings in correct order to match the training case of the policy. <br />

To run use: 
```bash
$ python -u demo.py --load=<run name>
```
Here <run name> is relative to the pixel2plan/runs/ folder. If the specified run does not exist, the run at the bottom
of the runs folder is used as default. <br />

To load a target image use:
```bash
$ pythin -u demo.py --load_target=<target_name>
```
Here <target_name> is relative to the pixel2plan/ folder. 
If the specified target does not exist, no target image is used. <br />

To see results use:
```bash
$ tensorboard --logdir=runs --samples_per_plugin=images=<n_tests>
```
Default for <n_tests> is set to 100 and can be adjusted in settings.json. Without samples_per_plugin Tensorboard can 
only display 10 images.


## Developers
- Michael Lutter
- Janosch Moos
- Kay Hansel

