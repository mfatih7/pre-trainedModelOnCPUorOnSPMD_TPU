import torch
import torch.nn as nn

# from loss_module import loss_functions_n_to_n

class PointCN_Layer_1(nn.Module):
    def __init__(self, w, d):
        super(PointCN_Layer_1, self).__init__()

        self.w = w
        self.d = d
        
        self.Conv2d = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=(1,self.w), stride=(1,1) ) 
        
    def forward(self, x):
        out = self.Conv2d(x)
        return out

class PointCN_CN(nn.Module):
    def __init__(self, d):
        super(PointCN_CN, self).__init__()

        self.d = d        
        self.InstanceNorm2d = nn.InstanceNorm2d(self.d, eps=1e-3)
        
    def forward(self, x):
        out = self.InstanceNorm2d(x)        
        return out

class PointCN_ResNet_Block(nn.Module):
    def __init__(self, d_in, d_out, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):
        super(PointCN_ResNet_Block, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive        
        
        self.contextnorm_1 = PointCN_CN( self.d_in )
        self.batchnorm_1 = nn.BatchNorm2d(self.d_in )
        self.Conv2d_1 = nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) )
        
        self.contextnorm_2 = PointCN_CN( self.d_out )
        self.batchnorm_2 = nn.BatchNorm2d(self.d_out )
        self.Conv2d_2 = nn.Conv2d(in_channels=self.d_out, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) ) 

        if(self.d_in != self.d_out):
            self.short_cut = nn.Conv2d(self.d_in, self.d_out, kernel_size=(1,1) )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):        
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = x
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_1(x)
        
        out = self.relu(self.batchnorm_1(out))
        
        out = self.Conv2d_1(out)
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = out
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_2(out)
            
        out = self.relu(self.batchnorm_2(out))
        
        out = self.Conv2d_2(out)
        
# shortcut
        if(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_active'):
            if(self.d_in != self.d_out):                
                x = self.short_cut(x)
            out = out + x                
        elif(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_inactive'):
            out = out
        
        return out
    
class OANET_pool(nn.Module):
    def __init__(self, d, m):
        super(OANET_pool, self).__init__()

        self.d = d
        self.m = m
        
        self.contextnorm = PointCN_CN( self.d )        
        self.batchnorm = nn.BatchNorm2d( self.d )
        self.relu = nn.ReLU() 
        self.Conv2d = nn.Conv2d( in_channels=self.d, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        self.softmax = nn.Softmax(dim=2)    
        
    def forward(self, x_level_1):      
        
        out = self.relu(self.batchnorm(self.contextnorm(x_level_1)))
        out = self.Conv2d(out)
        Spool = self.softmax(out)      
        
        out = torch.matmul( x_level_1.squeeze(3), torch.transpose(Spool, 1, 2).squeeze(3) ).unsqueeze(3)
        
        return out
    
class OANET_unpool(nn.Module):
    def __init__(self, d, m ):
        super(OANET_unpool, self).__init__()

        self.d = d
        self.m = m
        
        self.contextnorm = PointCN_CN( self.d )
        self.batchnorm = nn.BatchNorm2d( self.d )
        self.relu = nn.ReLU()
        self.Conv2d = nn.Conv2d( in_channels=self.d, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x_level_1, x_level_2):      
        
        out = self.relu(self.batchnorm(self.contextnorm(x_level_1)))
        out = self.Conv2d(out)
        Sunpool = self.softmax(out)
        
        out = torch.matmul( x_level_2.squeeze(3), Sunpool.squeeze(3) ).unsqueeze(3)
        
        return out

class Order_Aware_Filter_Block(nn.Module):
    def __init__(self, m, d):
        super(Order_Aware_Filter_Block, self).__init__()
            
        self.m = m
        self.d = d
        
        self.contextnorm_1 = PointCN_CN( self.d ) 
        self.batchnorm_1 = nn.BatchNorm2d( self.d)
        self.Conv2d_1 = nn.Conv2d( in_channels=self.d, out_channels=self.d, kernel_size=(1,1), stride=(1,1) ) 
        
        self.batchnorm_2 = nn.BatchNorm2d( self.m )
        self.Conv2d_2 = nn.Conv2d( in_channels=self.m, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        
        self.contextnorm_3 = PointCN_CN( self.d ) 
        self.batchnorm_3 = nn.BatchNorm2d( self.d )
        self.Conv2d_3 = nn.Conv2d( in_channels=self.d, out_channels=self.d, kernel_size=(1,1), stride=(1,1) ) 
        
        self.relu = nn.ReLU()        
        
    def forward(self, x):
        
        out = self.relu(self.batchnorm_1(self.contextnorm_1(x)))
        out_short_cut = self.Conv2d_1(out)
        
        out = torch.transpose(out_short_cut, 1, 2)
        
        out = self.relu(self.batchnorm_2(out))
        out = self.Conv2d_2(out)
        
        out = torch.transpose(out, 1, 2)
        
        out = out + out_short_cut
        
        out = self.relu(self.batchnorm_3(self.contextnorm_3(out)))
        out = self.Conv2d_3(out)

        out = out + x
        
        return out

class Order_Aware_Network(nn.Module):
    def __init__(self, n, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):        
        super(Order_Aware_Network, self).__init__()
            
        self.n = n
        self.w = model_width
        self.d = inner_dimension
        self.m = m
        
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive
        
        self.layer_1 = PointCN_Layer_1( self.w, self.d )
        
        self.ResNet_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_4 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.OANET_pool = OANET_pool( self.d, self.m )
        
        self.Order_Aware_Filter_Block_1 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_2 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_3 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_4 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_5 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_6 = Order_Aware_Filter_Block( self.m, self.d )
        
        self.OANET_unpool = OANET_unpool( self.d, self.m )
        
        self.ResNet_7 = PointCN_ResNet_Block( 2*self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_8 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_9 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_10 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_11 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_12 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
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
        
        out = self.layer_last(out)
        
        out = torch.reshape(out, (-1, self.n) )

        return [ out ]
    
class Order_Aware_Network_Iterative(nn.Module):
    def __init__(self, n, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):        
        super(Order_Aware_Network_Iterative, self).__init__()
            
        self.n = n
        self.w = model_width
        self.d = inner_dimension
        self.m = m
        
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive        
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive
        
        self.layer_1_stage_1 = PointCN_Layer_1( self.w, self.d )
        
        self.ResNet_1_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.OANET_pool_stage_1 = OANET_pool( self.d, self.m )
        
        self.Order_Aware_Filter_Block_1_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_2_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_3_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        
        self.OANET_unpool_stage_1 = OANET_unpool( self.d, self.m)
        
        self.ResNet_4_stage_1 = PointCN_ResNet_Block( 2*self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last_stage_1 = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
        #########################################################################################################################################################################################

        self.layer_1_stage_2 = PointCN_Layer_1( self.w+2, self.d )
        
        self.ResNet_1_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.OANET_pool_stage_2 = OANET_pool( self.d, self.m )
        
        self.Order_Aware_Filter_Block_1_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_2_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_3_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        
        self.OANET_unpool_stage_2 = OANET_unpool( self.d, self.m )
        
        self.ResNet_4_stage_2 = PointCN_ResNet_Block( 2*self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last_stage_2 = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
        self.relu = nn.ReLU()        
        self.tanh = nn.Tanh()

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
        
        out_1 = torch.reshape(out, (-1, self.n) )

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
        
        out = self.layer_last_stage_2(out)
        
        out = torch.reshape(out, (-1, self.n) )

        return [ out_1, out ]     

def get_model_OANET( N, model_width, en_checkpointing ):   
    
    inner_dimension = 128
    m = 500
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'    
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return Order_Aware_Network( N, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

def get_model_OANET_Iter( N, model_width, en_checkpointing ):   
    
    inner_dimension = 128
    m = 500
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'    
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return Order_Aware_Network_Iterative( N, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )
    
    from loss_module import loss_functions_n_to_n

    # N = 512
    # model_width = 4
    # inner_dimension = 128
    # m = 500    
    
    # OANET_00 = Order_Aware_Network( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
    #                                 residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    # OANET_01 = Order_Aware_Network( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
    #                                 residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    # OANET_10 = Order_Aware_Network( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
    #                                 residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    # OANET_11 = Order_Aware_Network( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
    #                                 residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    
    N = 512
    model_width = 4
    inner_dimension = 128
    m = 500    
    
    OANET_Iter_00 = Order_Aware_Network_Iterative( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
                                                    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    OANET_Iter_01 = Order_Aware_Network_Iterative( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
                                                    residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    OANET_Iter_10 = Order_Aware_Network_Iterative( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
                                                    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    OANET_Iter_11 = Order_Aware_Network_Iterative( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
                                                    residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
        
    