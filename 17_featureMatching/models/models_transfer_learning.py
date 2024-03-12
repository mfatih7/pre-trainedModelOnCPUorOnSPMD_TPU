import torch
import subprocess
import os

from loss_module import loss_functions_n_to_n

from models.models_LTFGC import LTFGC
from models.models_OANET import Order_Aware_Network
from models.models_OANET import Order_Aware_Network_Iterative
        
class LTFGC_Modified(LTFGC):
    def __init__(self, *args, **kwargs):
        super(LTFGC_Modified, self).__init__(*args, **kwargs)
    def forward(self, x):
            
        out = self.layer_1(x)
        
        out = self.ResNet_1(out)
        out = self.ResNet_2(out)
        out = self.ResNet_3(out)
        out = self.ResNet_4(out)
        out = self.ResNet_5(out)
        out = self.ResNet_6(out)
        out = self.ResNet_7(out)
        out = self.ResNet_8(out)
        out = self.ResNet_9(out)
        out = self.ResNet_10(out)
        out = self.ResNet_11(out)
        out = self.ResNet_12(out)
        
        return out
    
def get_model_LTFGC_Modified( N, model_width, ):   
    
    inner_dimension = 128
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return LTFGC_Modified( N, model_width, inner_dimension, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )
    
class OANET_Modified(Order_Aware_Network):
    def __init__(self, *args, **kwargs):
        super(OANET_Modified, self).__init__(*args, **kwargs)
    def forward(self, x):
        out = self.layer_1(x)
        
        out = self.ResNet_1(out)
        out = self.ResNet_2(out)
        out = self.ResNet_3(out)
        out = self.ResNet_4(out)
        out = self.ResNet_5(out)
        out_1st_ResNet_Group = self.ResNet_6(out)
        
        out_pool = self.OANET_pool(out_1st_ResNet_Group)
        
        out = self.Order_Aware_Filter_Block_1(out_pool)
        out = self.Order_Aware_Filter_Block_2(out)
        out = self.Order_Aware_Filter_Block_3(out)
        out = self.Order_Aware_Filter_Block_4(out)
        out = self.Order_Aware_Filter_Block_5(out)
        out = self.Order_Aware_Filter_Block_6(out)
        
        out_unpool = self.OANET_unpool(out_1st_ResNet_Group, out)
        
        out = torch.cat( [out_1st_ResNet_Group, out_unpool], dim=1)
        
        out = self.ResNet_7(out)
        out = self.ResNet_8(out)
        out = self.ResNet_9(out)
        out = self.ResNet_10(out)
        out = self.ResNet_11(out)
        out = self.ResNet_12(out)

        return out
    
def get_model_OANET_Modified( N, model_width, ):   
    
    inner_dimension = 128
    m = 500
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'    
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return OANET_Modified( N, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

class OANET_Iter_Modified(Order_Aware_Network_Iterative):
    def __init__(self, *args, **kwargs):
        super(OANET_Iter_Modified, self).__init__(*args, **kwargs)

    def forward(self, x):
        
### STAGE 1 ###
        
        out = self.layer_1_stage_1(x)
        
        out = self.ResNet_1_stage_1(out)
        out = self.ResNet_2_stage_1(out)
        out_1st_ResNet_Group = self.ResNet_3_stage_1(out)
        
        out_pool = self.OANET_pool_stage_1(out_1st_ResNet_Group)
        
        out = self.Order_Aware_Filter_Block_1_stage_1(out_pool)
        out = self.Order_Aware_Filter_Block_2_stage_1(out)
        out = self.Order_Aware_Filter_Block_3_stage_1(out)
        
        out_unpool = self.OANET_unpool_stage_1(out_1st_ResNet_Group, out)
        
        out = torch.cat( [out_1st_ResNet_Group, out_unpool], dim=1)
        
        out = self.ResNet_4_stage_1(out)
        out = self.ResNet_5_stage_1(out)
        out = self.ResNet_6_stage_1(out)
        
        out = self.layer_last_stage_1(out)
        
        # out_1 = torch.reshape(out, (-1, self.n) )

### STAGE 1 TO STAGE 2 ###
        
        e_hat = loss_functions_n_to_n.weighted_8points(x[:,:,:,:4], out)
        residual = loss_functions_n_to_n.batch_episym(x[:,0,:,:2], x[:,0,:,2:4], e_hat).unsqueeze(dim=1).unsqueeze(dim=-1).detach()
        
        logits = torch.relu( torch.tanh( out ) ).detach()
        
        x_stage_2 = torch.cat( ( x, residual, logits ), dim=-1)
        
### STAGE 2 ###
        
        out = self.layer_1_stage_2( x_stage_2 )
        
        out = self.ResNet_1_stage_2(out)
        out = self.ResNet_2_stage_2(out)
        out_1st_ResNet_Group = self.ResNet_3_stage_2(out)
        
        out_pool = self.OANET_pool_stage_2(out_1st_ResNet_Group)
        
        out = self.Order_Aware_Filter_Block_1_stage_2(out_pool)
        out = self.Order_Aware_Filter_Block_2_stage_2(out)
        out = self.Order_Aware_Filter_Block_3_stage_2(out)
        
        out_unpool = self.OANET_unpool_stage_2(out_1st_ResNet_Group, out)
        
        out = torch.cat( [out_1st_ResNet_Group, out_unpool], dim=1)
        
        out = self.ResNet_4_stage_2(out)
        out = self.ResNet_5_stage_2(out)
        out = self.ResNet_6_stage_2(out)

        return out

def get_model_OANET_Iter_Modified( N, model_width, ):   
    
    inner_dimension = 128
    m = 500
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'    
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return OANET_Iter_Modified( N, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

def load_and_freeze_model( config, tl_block, tl_exp_no, tl_best_exp_epoch, freeze_tl_block_parameters, tl_checkpoint_best_or_last, ):    

    checkpoint_file_local = os.path.join( '..', config.output_folder_name, f'{tl_exp_no:04d}', 'checkpoints', 'model.pth.tar' )

    import torch_xla.utils.serialization as xser
    checkpoint = xser.load( checkpoint_file_local )
    
    tl_block.load_state_dict(checkpoint['model_state_dict'])
    
    if( freeze_tl_block_parameters==1 ):
        # Freeze the parameters
        for param in tl_block.parameters():
            param.requires_grad = False

    return tl_block

def get_modified_tl_modelv( config, N, in_width, tl_model, tl_exp_no, tl_best_exp_epoch, load_model_params_from_checkpoint, freeze_tl_block_parameters, tl_checkpoint_best_or_last, ):
    
    if(tl_model == 'LTFGC'):
        tl_block =  get_model_LTFGC_Modified( N = N, model_width = in_width, )
    elif(tl_model == 'OANET'):
        tl_block =  get_model_OANET_Modified( N = N, model_width = in_width, )
    elif(tl_model == 'OANET_Iter'):
        tl_block =  get_model_OANET_Iter_Modified( N = N, model_width = in_width, )
        
    if load_model_params_from_checkpoint == 1 :
        if ( config.device == 'tpu' and config.tpu_cores == 'spmd' and ( config.model_type == 'model_exp20' or config.model_type == 'model_exp23' ) ) :
            tl_block = load_and_freeze_model( config, tl_block, tl_exp_no, tl_best_exp_epoch, freeze_tl_block_parameters, tl_checkpoint_best_or_last, )
            print( f'Loaded {tl_model} model of experiment {tl_exp_no:04d} for transfer learning' )
        else:
            raise NotImplementedError("load_model_params_from_checkpoint param is not implemented for these selections")
        
    return tl_block
    
