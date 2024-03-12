from config import get_config

from  process.train_1_1_each_sample_in_single_batch_TPU_single import train_and_val as train_1_1_each_sample_in_single_batch_TPU_single_train_and_val
from  process.train_1_1_each_sample_in_single_batch_TPU_multi import train_and_val as train_1_1_each_sample_in_single_batch_TPU_multi_train_and_val
from  process.train_1_1_each_sample_in_single_batch_TPU_spmd import train_and_val as train_1_1_each_sample_in_single_batch_TPU_spmd_train_and_val

import torch_xla.distributed.xla_multiprocessing as xmp    # for multi core processing

import tpu_related.set_env_variables_for_TPU as set_env_variables_for_TPU

import checkpoint

# Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again
# (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader
# instance creation logic here, as it doesn’t need to be re-executed in workers.

if __name__ == '__main__':    

    set_env_variables_for_TPU.set_env_variables_for_TPU_PJRT( )

    config = get_config()
    
    if( config.tpu_cores == 'spmd' ):
        set_env_variables_for_TPU.set_env_variables_for_TPU_SPMD( )
    
    set_env_variables_for_TPU.set_env_debug_variables_for_TPU_PJRT( config )
    
    experiment_no = config.first_experiment
    
    config.copy_config_file_to_output_folder( experiment_no )

    if( config.tpu_cores == 'spmd' ):
        chkpt_mgr = checkpoint.get_chkpt_mgr(config)
    
    N_images_in_batch = config.training_params[0][0]
    N = config.training_params[0][1]
    batch_size = config.training_params[0][2]
    
    if(config.operation == 'train'):
    
        if(config.input_type == '1_to_1'):
            
            if( N_images_in_batch >= 1 and N == batch_size ):
        
                if( config.tpu_cores == 'single' ):
                    
                    learning_rate = config.learning_rate
                    n_epochs = config.n_epochs
                    num_workers = config.num_workers
                    model_type = config.model_type
                    optimizer_type = config.optimizer_type
                    en_grad_checkpointing = config.en_grad_checkpointing
                    
                    print('Training starts for ' + 'train_1_1_each_sample_in_single_batch_TPU_single')
                    
                    training_results = train_1_1_each_sample_in_single_batch_TPU_single_train_and_val(   
                                                                                                        config,
                                                                                                        experiment_no,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        model_type,
                                                                                                        en_grad_checkpointing,
                                                                                                        N_images_in_batch,
                                                                                                        N,
                                                                                                        batch_size,
                                                                                                        optimizer_type, )
                elif( config.tpu_cores == 'multi' ):
                    
                    print('Training starts for ' + 'train_1_1_each_sample_in_single_batch_TPU_multi')
                    
                    FLAGS = {}
                    FLAGS['config']                     = config
                    FLAGS['experiment_no']              = experiment_no
                    FLAGS['learning_rate']              = config.learning_rate * 8  # Learning Rate is increased for 8 cores operation
                    FLAGS['n_epochs']                   = config.n_epochs
                    FLAGS['num_workers']                = config.num_workers
                    FLAGS['model_type']                 = config.model_type
                    FLAGS['optimizer_type']             = config.optimizer_type
                    FLAGS['en_grad_checkpointing']      = config.en_grad_checkpointing
                    FLAGS['N_images_in_batch']          = config.training_params[0][0]
                    FLAGS['N']                          = config.training_params[0][1]
                    FLAGS['batch_size']                 = config.training_params[0][2]
                    
                    xmp.spawn(train_1_1_each_sample_in_single_batch_TPU_multi_train_and_val, args=(FLAGS,) )
                    
                elif( config.tpu_cores == 'spmd' ):
                    
                    learning_rate = config.learning_rate
                    n_epochs = config.n_epochs
                    num_workers = config.num_workers
                    model_type = config.model_type
                    optimizer_type = config.optimizer_type
                    en_grad_checkpointing = config.en_grad_checkpointing
                    
                    print('Training starts for ' + 'train_1_1_each_sample_in_single_batch_TPU_spmd')
                    
                    training_results = train_1_1_each_sample_in_single_batch_TPU_spmd_train_and_val(   
                                                                                                        config,
                                                                                                        experiment_no,
                                                                                                        learning_rate,
                                                                                                        n_epochs,
                                                                                                        num_workers,
                                                                                                        model_type,
                                                                                                        en_grad_checkpointing,
                                                                                                        N_images_in_batch,
                                                                                                        N,
                                                                                                        batch_size,
                                                                                                        optimizer_type,
                                                                                                        chkpt_mgr, )
                    
                else:
                    raise ValueError(f"The provided arguments are not valid: {config.tpu_cores}")
                    
            else:
                raise ValueError(f"The provided arguments are not valid: {N_images_in_batch} {N} {batch_size}")
            
        else:
            
            raise ValueError(f"The provided arguments are not valid: {config.input_type}")
    else:
        raise ValueError(f"The provided arguments are not valid: {config.operation}")
