import torch
import torch.nn as nn

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
        
        self.conv2d = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = False, )
        
        self.norm = nn.BatchNorm2d( out_channels, track_running_stats=False, )
            
    def forward(self, x):
        
        out = self.norm( self.conv2d(x) )
            
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
    
class Pool_1_to_1_First(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Pool_1_to_1_First, self).__init__()
        
        self.conv2d = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False, )     
        self.softmax = nn.Softmax(dim=2)    
        
    def forward(self, x):        
        out = self.conv2d(x)
        Spool = self.softmax(out)
        return Spool
    
class Pool_1_to_1_MatMul(nn.Module):
    def __init__( self ):
        super(Pool_1_to_1_MatMul, self).__init__()        
        pass        
    def forward(self, x, Spool):        
        out = torch.matmul( x.squeeze(3), torch.transpose(Spool, 1, 2).squeeze(3) ).unsqueeze(3)        
        return out

class Block_Height_Reducing_Filtering(nn.Module):
    def __init__(self, n_head, height_in, height_out, channel_count_in, channel_count_out, pointwise_conv_count, non_lin):
        super(Block_Height_Reducing_Filtering, self).__init__()
        
        self.n_head = n_head
        
        self.pointwise_conv_count = pointwise_conv_count
        
        self.d = height_out
        
        self.pool_1_to_1_layers = nn.ModuleList([Pool_1_to_1_First(in_channels=channel_count_in, out_channels=self.d) for _ in range(n_head)])
        
        if self.pointwise_conv_count == 1:
            self.pointwise_Conv_Shortcut_layers = nn.ModuleList([Pointwise_Conv_Shortcut(in_channels=self.d, out_channels=self.d, non_lin=non_lin) for _ in range(n_head)])
        
        self.pool_1_to_1_MatMul = Pool_1_to_1_MatMul()
        
        self.heads_merging = Conv2d_N(in_channels=channel_count_in*n_head, out_channels=channel_count_out, kernel_size=(1,1), stride=(1,1))
                
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        head_outputs = []
        
        # Process each head
        for idx in range(self.n_head):
            head_output = self.pool_1_to_1_layers[idx](x)
            
            head_output = self.pool_1_to_1_MatMul(x, head_output)
            
            # If pointwise_conv_count is 1, apply Pointwise_Conv_Shortcut layer individually
            if self.pointwise_conv_count == 1:
                
                head_output = torch.transpose( head_output, 1, 2 )       
                
                head_output = self.pointwise_Conv_Shortcut_layers[idx](head_output)
                
                head_output = torch.transpose( head_output, 1, 2 )       
            
            head_outputs.append(head_output)
        
        # Concatenate outputs from all heads
        concatenated_heads_output = torch.cat(head_outputs, dim=1)
        
        # Merge the heads
        merged_heads_output = self.heads_merging(concatenated_heads_output)
        
        # Apply non-linearity
        out = self.non_lin(merged_heads_output)
        
        return out

class model_exp_00(nn.Module):
    def __init__(self,   config,
                         N,
                         d,
                         in_width,
                         tl_model,
                         tl_exp_no,                         
                         tl_best_exp_epoch,
                         load_model_params_from_checkpoint,
                         freeze_tl_block_parameters,
                         tl_checkpoint_best_or_last,
                         n_head,
                         heights_in,
                         heights_out,
                         channel_counts_in,
                         channel_counts_out,
                         pointwise_conv_count,
                         non_lin, ):
        super(model_exp_00, self).__init__()        
        
        self.N = N
        self.d = d
        self.in_width = in_width
        self.tl_model = tl_model
        self.tl_exp_no = tl_exp_no
        self.pointwise_conv_count = pointwise_conv_count
        self.non_lin = non_lin
        
        self.heights_in = heights_in 
        self.heights_out = heights_out
        self.channel_counts_in = channel_counts_in
        self.channel_counts_out = channel_counts_out
        
        self.tl_block = get_modified_tl_modelv( config, self.N, self.in_width, self.tl_model, self.tl_exp_no,
                                                tl_best_exp_epoch, load_model_params_from_checkpoint, freeze_tl_block_parameters, tl_checkpoint_best_or_last)
        if(freeze_tl_block_parameters==1):
            self.tl_block.eval()
        
        # self.model_transpose = model_transpose()
        
        # self.model_repeat_0 = model_repeat_0( self.N )  
        
        # self.model_repeat_1 = model_repeat_1( self.N )       
        
        # self.model_cat = model_cat( )           
        
        self.n_blocks = int( len(self.channel_counts_in) )         
        
        layers = []
        
        for block_no in range(self.n_blocks):
        
            if(block_no==0):                
                layers.append( Width_Reduction( in_width = self.d, out_channels = self.channel_counts_out[block_no], non_lin = self.non_lin, ) )
            else:
                layers.append( Block_Height_Reducing_Filtering( n_head = n_head,
                                                                height_in = self.heights_in[block_no],
                                                                height_out = self.heights_out[block_no],
                                                                channel_count_in = self.channel_counts_in[block_no],
                                                                channel_count_out = self.channel_counts_out[block_no],
                                                                pointwise_conv_count = pointwise_conv_count,
                                                                non_lin = non_lin, ) )
        self.net = nn.Sequential(*layers)
        
        self.initial_fully_connected_size = self.channel_counts_out[-1]
        
        self.fc1 = nn.Linear(self.initial_fully_connected_size * 1 * 1, 128 )
        
        self.fc2 = nn.Linear( 128 * 1 * 1, 1)
        
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        x = self.tl_block( x)    # 1,128,2048,1
        
        # x = self.model_transpose( x )
        
        x = torch.transpose(x, 1, 3)  # 1,1,2048,128
        
#############################
        
        x_batch_0 = x.repeat(self.N, 1, 1, 1)  # 2048,1,2048,128
        # x_batch_0 = x.expand(self.N, -1, -1, -1)  # 2048,1,2048,128
        
        ###  x_batch_1 = torch.cat([x[:, :, i, :].repeat(self.N, 1, self.N, 1) for i in range(self.N)], dim=0)
        
        x_batch_1 = x.repeat_interleave(self.N, dim=2).reshape(self.N, 1, self.N, 128) # 2048,1,2048,128

###############
        
        # x_batch_0 = x.expand(-1, self.N, -1, -1).transpose(0, 1)
        
        # indices = torch.arange(self.N).view(self.N, 1, 1, 1)
        
        # x_batch_1 = x.gather(2, indices.expand(-1, -1, self.N, 128))

###############

        x_batch = torch.cat((x_batch_0, x_batch_1), dim=1) # 2048,2,2048,128

#############################    

        # x_batch_0 = self.model_repeat_0( x )
        
        # x_batch_1 = self.model_repeat_1( x )
        
        # x_batch = self.model_cat( x_batch_0, x_batch_1 )
        
#############################

        # x = x.squeeze(dim=3)
        
        # x = torch.transpose(x, 1, 2)

        # x_batch_0 = x.repeat(self.N, 1, 1)

        # x_batch_1 = x.repeat_interleave(self.N, dim=1)
        
        # x_batch = torch.cat( ( x_batch_0.unsqueeze(dim=1), x_batch_1.unsqueeze(dim=1) ), dim=1 )  

#############################   
        
        
        out = self.net( x_batch )
        
        out = out.view( -1, self.initial_fully_connected_size * 1 * 1 )
        out = self.non_lin( self.fc1(out) )
        out = self.fc2(out)
        
        return out
    
class model_transpose(nn.Module):
    def __init__(self, ):
        super(model_transpose, self).__init__()        
        pass       
    def forward(self, x):        
        x = torch.transpose(x, 1, 3)    
        return x

class model_repeat_0(nn.Module):
    def __init__(self, N ):
        super(model_repeat_0, self).__init__()        
        self.N = N        
    def forward(self, x):        
        x_batch_0 = x.repeat(self.N, 1, 1, 1)        
        return x_batch_0
    
class model_repeat_1(nn.Module):
    def __init__(self, N ):
        super(model_repeat_1, self).__init__()        
        self.N = N          
    def forward(self, x):        
        x_batch_1 = x.repeat_interleave(self.N, dim=2).reshape(self.N, 1, self.N, 128)    
        return x_batch_1    
    
class model_cat(nn.Module):
    def __init__(self, ):
        super(model_cat, self).__init__() 
        pass
    def forward(self, x_batch_0, x_batch_1):        
        x_batch = torch.cat((x_batch_0, x_batch_1), dim=1)   
        return x_batch
        
def get_model( config, N, model_width, en_checkpointing, model_adjust_params = None ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        d = 128
        
        in_width = model_width
        
        non_lin = 'ReLU'
        # non_lin = 'LeakyReLU'  
        # non_lin = 'tanh'
        
        tl_best_exp_epochs = [ 88, 88, 96, 82, 84, 89, 98, 55, 48, 79, 81, 80]  # if last place 'last instead of number, since nchunks=1 exp_id is equal to epoch no
        
        if(model_adjust_params != None):            
            raise NotImplementedError(f"The feature '{model_adjust_params}' is not implemented yet.")
        else:  
            
            tl_models = [ 'LTFGC', 'LTFGC', 'LTFGC', 'LTFGC', 'OANET', 'OANET', 'OANET', 'OANET', 'OANET_Iter', 'OANET_Iter', 'OANET_Iter', 'OANET_Iter',  ]
            tl_exp_nos = [ 950, 951, 953, 954, 960, 961, 963, 964, 970, 971, 973, 974,  ]            
            
            n_heads = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]                
            pointwise_conv_counts = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
            
            if( config.model_exp_no in [  0,   1,   3,   4,  10,  11,  13,  14,  20,  21,  23,  24, ] ):
                exp_no_list_index = [  0,   1,   3,   4,  10,  11,  13,  14,  20,  21,  23,  24, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 0 # just for tests, all losses to train tl model are not connected properly
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'last' # not important here
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch = 128
                
                channel_counts_in = [     2,
                                       ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ]
                
                channel_counts_out = [ ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ch*13, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
                
            elif( config.model_exp_no in [  100,   101,   103,   104,  110,  111,  113,  114,  120,  121,  123,  124, ] ):
                exp_no_list_index = [  100,   101,   103,   104,  110,  111,  113,  114,  120,  121,  123,  124, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 0 # just for tests, all losses to train tl model are not connected properly
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'last' # not important here
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [  
                                       2,     192,
                                       256,   320,
                                       384,   448,
                                       512,   768,
                                      1024,  1536,  2048,  3072, ]
                
                channel_counts_out = [ 192,   256,
                                       320,   384,
                                       448,   512,
                                       768,  1024,
                                      1536,  2048,  3072,  4096, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
                
            elif( config.model_exp_no in [  200,   201,   203,   204,  210,  211,  213,  214,  220,  221,  223,  224, ] ):
                exp_no_list_index = [  200,   201,   203,   204,  210,  211,  213,  214,  220,  221,  223,  224, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 1
                tl_checkpoint_best_or_last = 'last'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch = 128
                
                channel_counts_in = [     2,
                                       ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ]
                
                channel_counts_out = [ ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ch*13, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]                
                
            elif( config.model_exp_no in [  300,   301,   303,   304,  310,  311,  313,  314,  320,  321,  323,  324, ] ):
                exp_no_list_index = [  300,   301,   303,   304,  310,  311,  313,  314,  320,  321,  323,  324, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 1
                tl_checkpoint_best_or_last = 'last'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [  
                                         2,   192,
                                       256,   320,
                                       384,   448,
                                       512,   768,
                                      1024,  1536,  2048,  3072, ]
                
                channel_counts_out = [ 192,   256,
                                       320,   384,
                                       448,   512,
                                       768,  1024,
                                      1536,  2048,  3072,  4096, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
                    
            elif( config.model_exp_no in [  400,   401,   403,   404,  410,  411,  413,  414,  420,  421,  423,  424, ] ):
                exp_no_list_index = [  400,   401,   403,   404,  410,  411,  413,  414,  420,  421,  423,  424, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'last'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch = 128
                
                channel_counts_in = [     2,
                                       ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ]
                
                channel_counts_out = [ ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ch*13, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]                
                
            elif( config.model_exp_no in [  500,   501,   503,   504,  510,  511,  513,  514,  520,  521,  523,  524, ] ):
                exp_no_list_index = [  500,   501,   503,   504,  510,  511,  513,  514,  520,  521,  523,  524, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'last'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [  
                                         2,   192,
                                       256,   320,
                                       384,   448,
                                       512,   768,
                                      1024,  1536,  2048,  3072, ]
                
                channel_counts_out = [ 192,   256,
                                       320,   384,
                                       448,   512,
                                       768,  1024,
                                      1536,  2048,  3072,  4096, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
            
            elif( config.model_exp_no in [  600,   601,   603,   604,  610,  611,  613,  614,  620,  621,  623,  624, ] ):
                exp_no_list_index = [  600,   601,   603,   604,  610,  611,  613,  614,  620,  621,  623,  624, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 1
                tl_checkpoint_best_or_last = 'best'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch = 128
                
                channel_counts_in = [     2,
                                       ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ]
                
                channel_counts_out = [ ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ch*13, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]                
                
            elif( config.model_exp_no in [  700,   701,   703,   704,  710,  711,  713,  714,  720,  721,  723,  724, ] ):
                exp_no_list_index = [  700,   701,   703,   704,  710,  711,  713,  714,  720,  721,  723,  724, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 1
                tl_checkpoint_best_or_last = 'best'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [  
                                         2,   192,
                                       256,   320,
                                       384,   448,
                                       512,   768,
                                      1024,  1536,  2048,  3072, ]
                
                channel_counts_out = [ 192,   256,
                                       320,   384,
                                       448,   512,
                                       768,  1024,
                                      1536,  2048,  3072,  4096, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
                    
            elif( config.model_exp_no in [  800,   801,   803,   804,  810,  811,  813,  814,  820,  821,  823,  824, ] ):
                exp_no_list_index = [  800,   801,   803,   804,  810,  811,  813,  814,  820,  821,  823,  824, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'best'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch = 128
                
                channel_counts_in = [     2,
                                       ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ]
                
                channel_counts_out = [ ch*2,    ch*3,
                                       ch*4,    ch*5,
                                       ch*6,    ch*7,
                                       ch*8,    ch*9,
                                      ch*10,   ch*11,  ch*12, ch*13, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]                
                
            elif( config.model_exp_no in [  900,   901,   903,   904,  910,  911,  913,  914,  920,  921,  923,  924, ] ):
                exp_no_list_index = [  900,   901,   903,   904,  910,  911,  913,  914,  920,  921,  923,  924, ].index(config.model_exp_no)
                
                load_model_params_from_checkpoint = 1
                freeze_tl_block_parameters = 0
                tl_checkpoint_best_or_last = 'best'
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [  
                                         2,   192,
                                       256,   320,
                                       384,   448,
                                       512,   768,
                                      1024,  1536,  2048,  3072, ]
                
                channel_counts_out = [ 192,   256,
                                       320,   384,
                                       448,   512,
                                       768,  1024,
                                      1536,  2048,  3072,  4096, ]
                
                if(N==512):
                    heights_in = [512] + heights_in[2:]            
                    heights_out = [512] + heights_out[2:]
                    channel_counts_in = channel_counts_in[0:-2]
                    channel_counts_out = channel_counts_out[0:-2]
                    
                elif(N==1024):
                    heights_in = [1024] + heights_in[1:]            
                    heights_out = [1024] + heights_out[1:]
                    channel_counts_in = channel_counts_in[0:-1]
                    channel_counts_out = channel_counts_out[0:-1]
                elif(N==2048):
                    heights_in = [2048] + heights_in[0:]            
                    heights_out = [2048] + heights_out[0:]
                    channel_counts_in = channel_counts_in[0:]
                    channel_counts_out = channel_counts_out[0:]
            
            else:
                raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")
            
            tl_model = tl_models[exp_no_list_index]
            tl_exp_no = tl_exp_nos[exp_no_list_index]
            tl_best_exp_epoch = tl_best_exp_epochs[exp_no_list_index]
            n_head = n_heads[exp_no_list_index]
            pointwise_conv_count = pointwise_conv_counts[exp_no_list_index]
                
        return model_exp_00( config,
                             N,
                             d,
                             in_width,
                             tl_model,
                             tl_exp_no,
                             tl_best_exp_epoch,
                             load_model_params_from_checkpoint,
                             freeze_tl_block_parameters,
                             tl_checkpoint_best_or_last,
                             n_head,
                             heights_in,
                             heights_out,
                             channel_counts_in,
                             channel_counts_out,
                             pointwise_conv_count,
                             non_lin, )
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
        
        # config.model_exp_no = model_no 
        # print(f'config.model_exp_no: {config.model_exp_no}')
        # model = get_model( config, N, model_width, en_checkpointing ).to(device)
        # get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)
        
        for name, layer in model.named_modules():
            if ( 'conv2d' in name ):
                print(name)
               
else:
    
    from models.models_transfer_learning import get_modified_tl_modelv
    