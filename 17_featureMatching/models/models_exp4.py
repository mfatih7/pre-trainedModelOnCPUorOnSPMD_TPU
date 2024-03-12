import torch
import torch.nn as nn
import math

class Non_Lin(nn.Module):
    def __init__(self, non_lin):
        super(Non_Lin, self).__init__()
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
        elif non_lin == 'tanh':
            self.non_lin = nn.Tanh()

    def forward(self, x):        
        out = self.non_lin( x )        
        return out

class Conv2d_N(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride ):
        super(Conv2d_N, self).__init__()
        
        self.cnn = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = False, )
        
        self.norm = nn.BatchNorm2d( out_channels, track_running_stats=False, )
            
    def forward(self, x):
        
        out = self.norm( self.cnn(x) )
            
        return out
    
class Width_Reduction(nn.Module):
    def __init__(self, in_width, out_channels, non_lin):
        super(Width_Reduction, self).__init__()
        
        self.width_reduction = Conv2d_N( in_channels = 2, out_channels = out_channels, kernel_size = (1, in_width), stride = (1,1), )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        
        out = self.non_lin( self.width_reduction(x) )
        
        return out
    
class Pointwise_Conv_Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, non_lin):
        super(Pointwise_Conv_Shortcut, self).__init__()
        
        self.pointwise_conv = Conv2d_N( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        shortcut = x
        x = self.pointwise_conv(x)
        x = self.non_lin( x )
        x = shortcut + x        
        return x
    
class Pool_1_to_1(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Pool_1_to_1, self).__init__()
        
        self.conv = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False, )     
        self.softmax = nn.Softmax(dim=2)    
        
    def forward(self, x):
        
        out = self.conv(x)
        Spool = self.softmax(out)        
        out = torch.matmul( x.squeeze(3), torch.transpose(Spool, 1, 2).squeeze(3) ).unsqueeze(3)        
        return out

class Block_Height_Reducing_Filtering(nn.Module):
    def __init__(self, in_channels, out_channels, height, pointwise_conv_count, second_pooling_enable, non_lin):
        super(Block_Height_Reducing_Filtering, self).__init__()
        
        self.height = height
        
        self.pointwise_conv_count = pointwise_conv_count
        
        self.second_pooling_enable = second_pooling_enable
        
        self.d = int(self.height/2)
        
        self.channel_expansion = Conv2d_N( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), )
        
        self.pool_1_to_1_A = Pool_1_to_1( in_channels = out_channels, out_channels = self.d, )
        
        if(self.pointwise_conv_count>0):
            pointwise_Conv_Shortcut_layers = []
            for lay in range( self.pointwise_conv_count ):
                pointwise_Conv_Shortcut_layers.append( Pointwise_Conv_Shortcut( in_channels = self.d, out_channels = self.d, non_lin = non_lin, ) )
            self.pointwise_Conv_Shortcut_layers_net = nn.Sequential(*pointwise_Conv_Shortcut_layers)
        
        if(self.second_pooling_enable):
            self.pool_1_to_1_B = Pool_1_to_1( in_channels = out_channels, out_channels = self.d, )
        
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        # Shortcut connection is on the block
        
        out = self.non_lin( self.channel_expansion( x ) )
        
        out = self.pool_1_to_1_A( out )
        
        if(self.pointwise_conv_count>0):
        
            out = torch.transpose( out, 1, 2 )        
        
            out = self.pointwise_Conv_Shortcut_layers_net( out )
        
            out = torch.transpose( out, 1, 2 )
        
        if(self.second_pooling_enable):
            out = self.pool_1_to_1_B( out )
             
        return out

class model_exp_00(nn.Module):
    def __init__(self,  N, 
                         in_width,
                         channel_counts_in,
                         channel_counts_out,
                         pointwise_conv_count,
                         second_pooling_enable,
                         non_lin, ):
        super(model_exp_00, self).__init__()        
        
        self.N = N
        self.in_width = in_width
        self.channel_counts_in = channel_counts_in
        self.channel_counts_out = channel_counts_out
        self.pointwise_conv_count = pointwise_conv_count
        self.second_pooling_enable = second_pooling_enable
        self.non_lin = non_lin
        
        self.n_blocks = int( len(channel_counts_in) )

        height = N
        
        layers = []
        
        for block_no in range(self.n_blocks):

            if(block_no==0):                
                layers.append( Width_Reduction( in_width = self.in_width, out_channels = self.channel_counts_out[block_no], non_lin = self.non_lin, ) )
            else:
                layers.append( Block_Height_Reducing_Filtering( in_channels = channel_counts_in[block_no],
                                                                out_channels = channel_counts_out[block_no],
                                                                height = height,
                                                                pointwise_conv_count = pointwise_conv_count,
                                                                second_pooling_enable = second_pooling_enable,
                                                                non_lin = non_lin, ) )
            if(block_no>0):
                height = int( height / 2 ) # Height is reduced at each blocks    
            
        self.net = nn.Sequential(*layers)
        
        self.initial_fully_connected_size = channel_counts_out[self.n_blocks-1]
        
        self.fc1 = nn.Linear(self.initial_fully_connected_size * 1 * 1, 128 )
        
        self.fc2 = nn.Linear( 128 * 1 * 1, 1)
        
        self.non_lin = Non_Lin( non_lin )
            
    def calculate_block_channel_counts(self):
        
        n0_matches = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n1_info_size = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n2_info_size_with_channels_raw = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n3_info_size_with_channels_processed = torch.zeros( (self.n_blocks), dtype=torch.float32 )
        n4_info_size_with_channels_processed_reduced = torch.zeros( (self.n_blocks), dtype=torch.float32 )

        for i in range( self.n_blocks ):
            
            n0_matches[i] = 2**i
            
            for j in range( n0_matches[i], 0, -1 ):
                n1_info_size[i] += j
                
            n2_info_size_with_channels_raw[i] = self.init_channel_count * n1_info_size[i]
            
            n3_info_size_with_channels_processed[i] = n2_info_size_with_channels_raw[i] * (self.ch_expans_base_param**(i*self.ch_expans_power_param))
            
            n4_info_size_with_channels_processed_reduced[i] = n3_info_size_with_channels_processed[i] * self.channel_reduction_ratio
        
        return n3_info_size_with_channels_processed.to(torch.int), n4_info_size_with_channels_processed_reduced.to(torch.int)
    
    def distribute_integer(self, pointwise_or_order_aware, N, n, a, power_param):
        """
        Distributes the integer 'a' among elements based on the given weights.
        Ensures that each element gets an integer value.

        :param a: Integer to be distributed.
        :param weights: List of float weights for each element.
        :return: List of integers representing the distributed amounts.
        """
        
        weights = []
        
        if( pointwise_or_order_aware == 'pointwise' ):
            for i in range(N):
                weights.append(i**power_param)
        elif( pointwise_or_order_aware == 'order_aware' ):
            for i in range(N):
                if(i==0):
                    weights.append(0)
                else:
                    weights.append(i**power_param)             

        # Normalize weights so their sum is 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Initial distribution based on weights
        distribution = [int(a * w) for w in normalized_weights]

        # Adjust for rounding errors
        distributed_sum = sum(distribution)
        difference = a - distributed_sum

        # Sorting indices based on the fractional part lost during rounding
        fractional_parts = sorted(range(n), key=lambda i: normalized_weights[i] * a - distribution[i], reverse=True)

        # Distributing the remaining amount based on the fractional parts
        for i in range(difference):
            distribution[fractional_parts[i]] += 1

        return weights, distribution
    
    def forward(self, x):
        
        x = self.net( x )
        
        x = x.view( -1, self.initial_fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( config, N, model_width, en_checkpointing, model_adjust_params = None ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        in_width = model_width
        
        non_lin = 'ReLU'
        # non_lin = 'LeakyReLU'  
        # non_lin = 'tanh'
        
        if(model_adjust_params != None):            
            raise NotImplementedError(f"The feature '{model_adjust_params}' is not implemented yet.")
        else:        
            
            if( config.model_exp_no >= 0 and config.model_exp_no < 4  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [   2,   32,   48,   64,   
                                       96,  128,  192,  256,
                                      384,  512,  768, 1024, ]
                
                channel_counts_out = [  32,   48,   64,   96,
                                       128,  192,  256,  384,
                                       512,  768, 1024, 1536, ]
                
                exp_no_list_index = config.model_exp_no - 0
                
            elif( config.model_exp_no >= 10 and config.model_exp_no < 14  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [   2,   48,    72,    96,
                                      144,  192,   288,   384,
                                      576,  768,  1152,  1536,  ]
                
                channel_counts_out = [   48,    72,    96,   144,
                                        192,   288,   384,   576,
                                        768,  1152,  1536,  2304,  ]
                
                exp_no_list_index = config.model_exp_no - 10
            
            elif( config.model_exp_no >= 20 and config.model_exp_no < 24  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [   2,    64,    96,   128,
                                      192,   256,   384,   512,
                                      768,  1024,  1536,  2048,  ]
                
                channel_counts_out = [    64,    96,    128,   192,
                                         256,   384,    512,   768,
                                        1024,  1536,   2048,  3072,   ]
                
                exp_no_list_index = config.model_exp_no - 20
                
            elif( config.model_exp_no >= 25 and config.model_exp_no < 29  ):
                
                pointwise_conv_counts = [ 0, 2, 0, 2, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [   2,    64,    96,   128,
                                      192,   256,   384,   512,
                                      768,  1024,  1536,  2048,  ]
                
                channel_counts_out = [    64,    96,    128,   192,
                                         256,   384,    512,   768,
                                        1024,  1536,   2048,  3072,   ]
                
                exp_no_list_index = config.model_exp_no - 25
                
            elif( config.model_exp_no >= 30 and config.model_exp_no < 34  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [    2,    128,    192,    256,
                                       384,    512,    640,    768,
                                       896,   1024,   1280,   1536, ]
                
                channel_counts_out = [  128,    192,   256,    384,
                                        512,    640,   768,    896,
                                       1024,   1280,  1536,   2048, ]
                
                exp_no_list_index = config.model_exp_no - 30
                
            elif( config.model_exp_no >= 35 and config.model_exp_no < 39  ):
                
                pointwise_conv_counts = [ 0, 2, 0, 2, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [    2,    128,    192,    256,
                                       384,    512,    640,    768,
                                       896,   1024,   1280,   1536, ]
                
                channel_counts_out = [  128,    192,   256,    384,
                                        512,    640,   768,    896,
                                       1024,   1280,  1536,   2048, ]
                
                exp_no_list_index = config.model_exp_no - 35
                
            elif( config.model_exp_no >= 40 and config.model_exp_no < 44  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [    2,    128,    256,   384,
                                       512,    640,    768,   896,
                                      1024,   1280,   1536,  1792, ]
                
                channel_counts_out = [  128,    256,   384,    512,
                                        640,    768,   896,   1024,
                                       1280,   1536,  1792,   2048, ]
                
                exp_no_list_index = config.model_exp_no - 40
                
            elif( config.model_exp_no >= 50 and config.model_exp_no < 54  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [      2,    128*1,    128*2,   128*3,
                                       128*4,    128*5,    128*6,   128*7,
                                       128*8,    128*9,   128*10,  128*11, ]
                
                channel_counts_out = [  128*1,    128*2,   128*3,    128*4,
                                        128*5,    128*6,   128*7,    128*8,
                                        128*9,   128*10,  128*11,   128*12, ]
                
                exp_no_list_index = config.model_exp_no - 50
                
            elif( config.model_exp_no >= 60 and config.model_exp_no < 64  ):
                
                pointwise_conv_counts = [ 0, 1, 0, 1, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [      2,    256*1,    256*2,   256*3,
                                       256*4,    256*5,    256*6,   256*7,
                                       256*8,    256*9,   256*10,  256*11, ]
                
                channel_counts_out = [  256*1,    256*2,   256*3,    256*4,
                                        256*5,    256*6,   256*7,    256*8,
                                        256*9,   256*10,  256*11,   256*12, ]
                
                exp_no_list_index = config.model_exp_no - 60
                
            elif( config.model_exp_no >= 70 and config.model_exp_no < 74  ):
                
                pointwise_conv_counts = [ 0, 2, 0, 2, ]
                
                second_pooling_enables = [ 0, 0, 1, 1, ]
                
                channel_counts_in = [      2,    256*1,    256*2,   256*3,
                                       256*4,    256*5,    256*6,   256*7,
                                       256*8,    256*9,   256*10,  256*11, ]
                
                channel_counts_out = [  256*1,    256*2,   256*3,    256*4,
                                        256*5,    256*6,   256*7,    256*8,
                                        256*9,   256*10,  256*11,   256*12, ]
                
                exp_no_list_index = config.model_exp_no - 70
            
            else:
                raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")    
            
            pointwise_conv_count = pointwise_conv_counts[exp_no_list_index]
            
            second_pooling_enable = second_pooling_enables[exp_no_list_index]
                
        return model_exp_00( N, 
                             in_width,
                             channel_counts_in,
                             channel_counts_out,
                             pointwise_conv_count,
                             second_pooling_enable,
                             non_lin, )
    else:
        raise ValueError(f"The provided argument is not valid: {N}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    config = get_config()
    
    device = 'cpu'
    
    # N = 512
    # N = 1024
    N = 2048
    model_width = 4
    en_checkpointing = False
    
####################################################################################
    
    # first_model_no = 0
    # last_model_no = 4
    
    # first_model_no = 10
    # last_model_no = 14
    
    # first_model_no = 20
    # last_model_no = 24
    
    # first_model_no = 25
    # last_model_no = 29
    
    # first_model_no = 30
    # last_model_no = 34
    
    first_model_no = 35
    last_model_no = 39
    
    # first_model_no = 40
    # last_model_no = 44
    
    # first_model_no = 50
    # last_model_no = 54
    
    # first_model_no = 60
    # last_model_no = 64
    
    # first_model_no = 70
    # last_model_no = 74
    
####################################################################################
    
    for i in range( first_model_no, last_model_no, 1 ):
        config.model_exp_no = i 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)
        
        for name, layer in model.named_modules():            
            if 'cnn' in name or 'conv' in name:
                print(f'{name}')

