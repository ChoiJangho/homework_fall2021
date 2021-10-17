Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).


## Installation guide for mujoco_py in ubuntu

error message:

```
fatal error: GL/osmesa.h: No such file or directory
        1 | #include <GL/osmesa.h>
          |          ^~~~~~~~~~~~~
    compilation terminated.
    error: command 'gcc' failed with exit status 1
```

solution:

```
sudo apt-get install libosmesa6-dev
```

error message:

```
error: [Errno 2] No such file or directory: 'patchelf': 'patchelf'
```

solution:

```
sudo apt-get install patchelf
```
