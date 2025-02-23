import os 

def set_gpu(gpu_input):
    gpus = gpu_input
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return [int(x) for x in gpus.split(",")]