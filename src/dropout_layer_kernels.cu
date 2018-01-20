#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer l, network net)
{
    if (!net.train) return;
    int size = l.inputs*l.batch;
    cuda_random(l.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        l.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, l.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, l.rand_gpu, l.probability, l.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer l, network net)
{
    if(!net.delta_gpu) return;
    int size = l.inputs*l.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, l.rand_gpu, l.probability, l.scale);
    check_error(cudaPeekAtLastError());
}
