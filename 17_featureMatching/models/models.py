
from models.models_exp20 import get_model as model_exp20_get_model
from models.models_exp23 import get_model as model_exp23_get_model

import torch
from torchsummary import summary
from thop import profile
from models.torchSummaryWrapper import get_torchSummaryWrapper
        
def get_model( config, model_type, N, model_width, en_checkpointing ):   
    
    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):        
        
        if( model_type == 'model_exp20' ):
            return model_exp20_get_model( config, N, model_width, en_checkpointing )
		elif( model_type == 'model_exp23' ):
            return model_exp23_get_model( config, N, model_width, en_checkpointing )
        
        else:
            raise ValueError(f"The provided argument is not valid: {model_type}")
    else:        
        raise ValueError(f"The provided argument is not valid: {N}")
def set_tl_block_eval_mode( config, model ):
    
    if( ( config.model_type == 'model_exp20' and ( (config.model_exp_no>=200 and config.model_exp_no<400) or (config.model_exp_no>=600 and config.model_exp_no<800) ) ) or
        ( config.model_type == 'model_exp22' and ( (config.model_exp_no>=200 and config.model_exp_no<400) ) ) ):
        model.tl_block.eval()    
    return model

def get_model_structure( config, device, model, N, model_width, en_grad_checkpointing):   
    
    if(en_grad_checkpointing==False):
        summary(model, (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    else:                    
        summary(get_torchSummaryWrapper( model ), (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
def get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing):   
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
    return params, flops
    