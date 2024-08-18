## Accuracy ##
**Across 10 Epochs**: 10.327455919395465, 10.52896725440806, 10.226700251889168, 11.259445843828715, 15.340050377833753, 21.007556675062972, 13.501259445843829, 15.365239294710328, 13.954659949622167, 20.05037783375315
**Max**: 21.01%  

## Variables (speechClass.py) ##
batch_size = 256            # batch size 
target_error_rate = 0.01    # loss threshold 
eps = 0.08                  # epsilon   
step_size = eps/25          
train_step = 20             
seed = 8
sample_rate = 16000

# Used in Testing/debugging
SR = 16000
EXAMPLES = 3
## Variables (trainUnlearn.py) ##
num_classes = 13 
batch_size = 256
perturb_tensor_path = "experiments/perturbation.pt"
log_interval = 20
n_epoch = 10
poison_rate = 1.0
seed = 8   
sample_rate = 16000
