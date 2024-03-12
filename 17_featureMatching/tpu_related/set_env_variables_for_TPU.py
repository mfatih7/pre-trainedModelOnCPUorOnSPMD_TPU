import os

def set_env_variables_for_TPU_PJRT( ):

    os.environ['PJRT_DEVICE'] = 'TPU'
    
def set_env_variables_for_TPU_SPMD( ):
    
    import torch.distributed as dist
    import torch_xla.distributed.xla_backend
    from torch_xla import runtime as xr
    
    xr.use_spmd()
    
    dist.init_process_group('gloo', init_method='xla://')
        
def set_env_debug_variables_for_TPU_PJRT( config ):
    
    if(config.XLA_USE_BF16==1):
        
        os.environ['XLA_USE_BF16'] = '1'

    if(config.TPU_DEBUG==1):

        os.environ['PT_XLA_DEBUG'] = '1'
        # os.environ['XLA_IR_DEBUG'] = '1'
        # os.environ['XLA_HLO_DEBUG'] = '1'
        os.environ['XLA_SAVE_TENSORS_FMT'] = 'text'    
        
        folder_name = os.path.join(config.tpu_debug_path, 'xla_debug')
        file_name = 'save_' + 'file_deb' + '.ir'    
        
        if(os.path.exists(folder_name)==0):
            os.makedirs(folder_name)
        
        os.environ['XLA_SAVE_TENSORS_FILE'] = os.path.join(folder_name, file_name)       