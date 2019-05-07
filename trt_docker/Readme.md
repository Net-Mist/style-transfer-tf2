# Docker for TensorRT

If you want to experiment with TensorRT in tensorflow 2, you will need to build you own version. Here the compilation of Tensorflow is make inside docker.
To compile tensorflow just run `make build`. It will generate the python wheel
 `tensorflow_pkg/tf_nightly_gpu-1.13.1-cp36-cp36m-linux_x86_64.whl` (Please note than even if it's writen 1.13.1 it is version 2) and a docker image `style_transfer/tf:0.1`
 
## Credit
The dockerfile are mainly not from me but from the official tensorflow repository. Please note the following difference between the version of the files in this projet and 
the official : 

### devel.Dockerfile
Same as https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu.Dockerfile but:
- ubuntu version change from 16.04 to 18.04
- Remove ARCH because useless here
- Only keep TF_CUDA_COMPUTE_CAPABILITIES=6.1,7.0
- Force python 3
- write the command for compiling tensorflow at the end of the file

### Dockerfile
Same as https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile but:
- ubuntu version change from 16.04 to 18.04
- Remove ARCH because useless
- Force python 3
