import torch
import torch.nn as nn

# from loss_module import loss_functions_n_to_n

class LTFGC_Layer_1(nn.Module):
    def __init__(self, w, d):
        super(LTFGC_Layer_1, self).__init__()

        self.w = w
        self.d = d
        
        self.Conv2d = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=(1,self.w), stride=(1,1) ) 
        
    def forward(self, x):
        out = self.Conv2d(x)
        return out

class LTFGC_CN(nn.Module):
    def __init__(self, d):
        super(LTFGC_CN, self).__init__()

        self.d = d        
        self.InstanceNorm2d = nn.InstanceNorm2d(self.d, eps=1e-3)
        
    def forward(self, x):
        out = self.InstanceNorm2d(x)        
        return out

class LTFGC_ResNet_Block(nn.Module):
    def __init__(self, d_in, d_out, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):
        super(LTFGC_ResNet_Block, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive
        
        self.Conv2d_1 = nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) ) 
        self.contextnorm_1 = LTFGC_CN( self.d_out )
        self.batchnorm_1 = nn.BatchNorm2d(self.d_out )
        
        self.Conv2d_2 = nn.Conv2d(in_channels=self.d_out, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) ) 
        self.contextnorm_2 = LTFGC_CN( self.d_out )
        self.batchnorm_2 = nn.BatchNorm2d(self.d_out )

        if(self.d_in != self.d_out):
            self.short_cut = nn.Conv2d(self.d_in, self.d_out, kernel_size=(1,1) )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):        
        
        out = self.Conv2d_1(x)
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = out
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_1(out)
        
        out = self.relu(self.batchnorm_1(out))
        
        out = self.Conv2d_2(out)
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = out
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_2(out)
            
        out = self.relu(self.batchnorm_2(out))
        
        # shortcut
        if(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_active'):
            if(self.d_in != self.d_out):                
                x = self.short_cut(x)
            out = out + x                
        elif(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_inactive'):
            out = out
        
        return out

class LTFGC(nn.Module):
    def __init__(self, n, model_width, inner_dimension, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):
        super(LTFGC, self).__init__()
        
        self.n = n
        self.w = model_width
        self.d = inner_dimension
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive
        
        self.layer_1 = LTFGC_Layer_1( self.w, self.d )
        
        self.ResNet_1 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_4 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_7 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_8 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_9 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_10 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_11 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_12 = LTFGC_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
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
        
        out = self.layer_last(out)
        
        out = torch.reshape(out, (-1, self.n) )
        
        return [ out ]
        
def get_model_LTFGC( N, model_width, en_checkpointing ):   
    
    inner_dimension = 128
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return LTFGC( N, model_width, inner_dimension, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

if __name__ == '__main__':
    
    N = 512
    model_width = 4
    inner_dimension = 128
    
    LTFGC_00 = LTFGC( N, model_width, inner_dimension, CN_active_or_CN_inactive = 'CN_active',
                      residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    LTFGC_01 = LTFGC( N, model_width, inner_dimension, CN_active_or_CN_inactive = 'CN_active',
                      residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    LTFGC_10 = LTFGC( N, model_width, inner_dimension, CN_active_or_CN_inactive = 'CN_inactive',
                      residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    LTFGC_11 = LTFGC( N, model_width, inner_dimension, CN_active_or_CN_inactive = 'CN_inactive',
                      residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    
    