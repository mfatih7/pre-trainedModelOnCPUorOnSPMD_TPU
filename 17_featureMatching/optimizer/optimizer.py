import torch.optim as optim

def get_optimizer( config, optimizer_type, model, learning_rate, ):
    
    if(optimizer_type == 'ADAM'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer_type == 'ADAMW'):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif(optimizer_type == 'SGD'):
        optimizer = optim.SGD( model.parameters(), lr=learning_rate, momentum=config.momentum, )
    else:
        raise ValueError(f"The provided argument is not valid: {optimizer_type}")
    
    if(optimizer_type == 'SGD'):
        print( 'For parameter ' + optimizer_type +', optimizer ' + type(optimizer).__name__ + ' with momentum ' + str(optimizer.param_groups[0]['momentum']) + 
               ' and learning rate ' + str(optimizer.param_groups[0]['lr']) + ' will be used in training')
    else:
        print( 'For parameter ' + optimizer_type +', optimizer ' + type(optimizer).__name__ +
               ' with learning rate ' + str(optimizer.param_groups[0]['lr']) + ' will be used in training')
        
    return optimizer