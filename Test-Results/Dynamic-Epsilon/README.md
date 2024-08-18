## Accuracy ##
**Across 10 Epochs**: 14.861460957178842, 14.836272040302267, 14.282115869017632, 12.317380352644836, 14.73551637279597, 7.607052896725441, 16.19647355163728, 12.241813602015114, 16.675062972292192, 16.297229219143578  
**Max**: 16.68%  

## Variables (speechClass.py) ##
batch_size = 256            # batch size   
target_error_rate = 0.01    # loss threshold   
eps_max_value = 0.13        # MAX epsilon   
step_size_factor = 25 
segment_size = 1000  
train_step = 20   
seed = 8  
sample_rate = 16000  



## Variables (trainUnlearn.py) ##
num_classes = 13   
batch_size = 256  
log_interval = 20  
n_epoch = 10  
poison_rate = 1.0  
seed = 8     
sample_rate = 16000  
