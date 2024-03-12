import torch
import torch.nn as nn

class model_exp_00(nn.Module):
    def __init__(self,   config,
                         N,
                         in_width,
                         tl_model,
                         tl_exp_no,                         
                         tl_best_exp_epoch,
                         tl_checkpoint_best_or_last, ):
        super(model_exp_00, self).__init__()

        self.N = N
        self.in_width = in_width

        load_model_params_from_checkpoint = 1
        freeze_tl_block_parameters = 1

        self.tl_block = get_modified_tl_modelv( config, self.N, self.in_width , tl_model, tl_exp_no, tl_best_exp_epoch,
                                                load_model_params_from_checkpoint, freeze_tl_block_parameters, tl_checkpoint_best_or_last)
        
    def forward(self, x):
        
        out = self.tl_block( x)    # 1,128,2048,1
        
        out = torch.transpose(out, 1, 3)  # 1,1,2048,128

        return out
        
def get_model( config, N, ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        tl_best_exp_epochs = [ 88, 88, 96, 82, 84, 89, 98, 55, 48, 79, 81, 80]  # if last place 'last instead of number, since nchunks=1 exp_id is equal to epoch no 
            
        tl_models = [ 'LTFGC', 'LTFGC', 'LTFGC', 'LTFGC', 'OANET', 'OANET', 'OANET', 'OANET', 'OANET_Iter', 'OANET_Iter', 'OANET_Iter', 'OANET_Iter',  ]
        tl_exp_nos = [ 950, 951, 953, 954, 960, 961, 963, 964, 970, 971, 973, 974,  ]            
            
        if( config.model_exp_no in [  0,   1,   3,   4,  10,  11,  13,  14,  20,  21,  23,  24, ] ):
            exp_no_list_index = [  0,   1,   3,   4,  10,  11,  13,  14,  20,  21,  23,  24, ].index(config.model_exp_no)
            
            tl_checkpoint_best_or_last = 'last'

        elif( config.model_exp_no in [  100,   101,   103,   104,  110,  111,  113,  114,  120,  121,  123,  124, ] ):
            exp_no_list_index = [  100,   101,   103,   104,  110,  111,  113,  114,  120,  121,  123,  124, ].index(config.model_exp_no)
            
            tl_checkpoint_best_or_last = 'best'
            
        elif( config.model_exp_no in [  200,   201,   203,   204,  210,  211,  213,  214,  220,  221,  223,  224, ] ):
            exp_no_list_index = [  200,   201,   203,   204,  210,  211,  213,  214,  220,  221,  223,  224, ].index(config.model_exp_no)
            
            tl_checkpoint_best_or_last = 'last'

        elif( config.model_exp_no in [  300,   301,   303,   304,  310,  311,  313,  314,  320,  321,  323,  324, ] ):
            exp_no_list_index = [  300,   301,   303,   304,  310,  311,  313,  314,  320,  321,  323,  324, ].index(config.model_exp_no)
            
            tl_checkpoint_best_or_last = 'best'        

        else:
            raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")

        if( (exp_no_list_index % 4) < 2 ):
            in_width = 4
        else:
            in_width = 6
            
        tl_model = tl_models[exp_no_list_index]
        tl_exp_no = tl_exp_nos[exp_no_list_index]
        tl_best_exp_epoch = tl_best_exp_epochs[exp_no_list_index]
                
        return model_exp_00( config,
                             N,
                             in_width,
                             tl_model,
                             tl_exp_no,
                             tl_best_exp_epoch,
                             tl_checkpoint_best_or_last, )
    else:
        raise ValueError(f"The provided argument is not valid: {N}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    
    from models.models_transfer_learning import get_modified_tl_modelv
    
    config = get_config()
    
    device = 'cpu'
    
    N = 512
    # N = 1024
    # N = 2048
    model_width = 4
    en_checkpointing = False
    
####################################################################################

    model_nos = [   0,   1,   3,   4, ]
    # model_nos = [  10,  11,  13,  14, ]
    # model_nos = [  20,  21,  23,  24, ]

    # model_nos = [  100,  101,  103,  104, ]
    # model_nos = [  110,  111,  113,  114, ]
    # model_nos = [  120,  121,  123,  124, ]
    
####################################################################################
    
    for model_no in model_nos:
        
        config.model_exp_no = model_no 
        
        random_input = torch.rand( 1, 1, N, model_width, )
        
        model = get_model( config, N, model_width, en_checkpointing, ).to(device)
        
        output = model( random_input ) 
        
        config.model_exp_no = model_no 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)
               
else:
    
    from models.models_transfer_learning import get_modified_tl_modelv
    