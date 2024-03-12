import numpy as np
import torch

def get_losses( config, device, labels_device, logits):
    
    # Classification loss
    # The groundtruth epi sqr
    gt_geod_d = labels_device[:, :]
    
    if(device=='cpu' or device=='cuda'):
        is_pos = (gt_geod_d < config.obj_geod_th).type(logits.type())
        is_neg = (gt_geod_d >= config.obj_geod_th).type(logits.type())
    else:
        is_pos = (gt_geod_d < config.obj_geod_th).float()
        is_neg = (gt_geod_d >= config.obj_geod_th).float()
    
    c = is_pos - is_neg
    classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
    # balance
    num_pos = torch.relu(torch.sum(is_pos, dim=0) - 1.0) + 1.0
    num_neg = torch.relu(torch.sum(is_neg, dim=0) - 1.0) + 1.0
    classif_loss_p = torch.sum(classif_losses * is_pos, dim=0)
    classif_loss_n = torch.sum(classif_losses * is_neg, dim=0)
    classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)
    
    # classif_loss.backward()
    
    return classif_loss

def calculate_ess_loss_and_L2loss( config, inputs_model_B, xs_ess, R_device, t_device, virtPt_device ):
    
    e_hat = weighted_8points(xs_ess, inputs_model_B)
        
    
    e_gt_unnorm = torch.reshape(torch.matmul(
        torch.reshape(torch_skew_symmetric(t_device), (-1, 3, 3)),
        torch.reshape(R_device, (-1, 3, 3))
    ), (-1, 9))
    
    e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)
    
    ess_hat = e_hat                
    
    # Essential/Fundamental matrix loss
    virtPt_device = virtPt_device[np.newaxis, :, :]
    pts1_virts, pts2_virts = virtPt_device[:, :, :2], virtPt_device[:,:,2:]
    geod = batch_episym(pts1_virts, pts2_virts, e_hat)
    essential_loss = torch.min(geod, config.geo_loss_margin*geod.new_ones(geod.shape))
    essential_loss = essential_loss.mean()
    # we do not use the l2 loss, just save the value for convenience 
    L2_loss = torch.mean(torch.min(
        torch.sum(torch.pow(ess_hat - e_gt, 2), dim=1),
        torch.sum(torch.pow(ess_hat + e_gt, 2), dim=1)
    ))
    
    
    return essential_loss, L2_loss, e_hat

    # e_hat = loss_functions.weighted_8points(xs_ess, inputs_model_B)
    
    
    
    # e_gt_unnorm = torch.reshape(torch.matmul(
    #     torch.reshape(loss_functions.torch_skew_symmetric(t_device), (-1, 3, 3)),
    #     torch.reshape(R_device, (-1, 3, 3))
    # ), (-1, 9))
    
    # e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)
    
    # ess_hat = e_hat                
    
    # # Essential/Fundamental matrix loss
    # virtPt_device = virtPt_device[np.newaxis, :, :]
    # pts1_virts, pts2_virts = virtPt_device[:, :, :2], virtPt_device[:,:,2:]
    # geod = loss_functions.batch_episym(pts1_virts, pts2_virts, e_hat)
    # essential_loss = torch.min(geod, config.geo_loss_margin*geod.new_ones(geod.shape))
    # essential_loss = essential_loss.mean()
    # # we do not use the l2 loss, just save the value for convenience 
    # L2_loss = torch.mean(torch.min(
    #     torch.sum(torch.pow(ess_hat - e_gt, 2), dim=1),
    #     torch.sum(torch.pow(ess_hat + e_gt, 2), dim=1)
    # ))
    
    # e_hat_2 = predict_essential_matrix_with_8_point_algorithm(
    #                                                             feature_pair_coord = xs_ess,                                            
    #                                                             weights = inputs_model_B,
    #                                                          )
    # L3_loss = torch.mean(torch.min(
    #     torch.sum(torch.pow(e_hat_2 - e_gt, 2), dim=1),
    #     torch.sum(torch.pow(e_hat_2 + e_gt, 2), dim=1)
    # ))

        
    # inputs_model_B.requires_grad = True
    # outputs_model_B = model_B(inputs_model_B)
    # loss_B = criterion_B(outputs_model_B, labels_model_B_device)
    # loss_B.backward()

def weighted_8points(x_in, logits):
    
    logits = torch.squeeze(logits, axis=-1)
    x_in = torch.unsqueeze(x_in, axis=0)
    
    
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

def batch_symeig(X):
    # it is much faster to run symeig on CPU, back to GPU
    # X = X.cpu()
    
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        # RuntimeError: This function was deprecated since version 1.9 and is now removed.
        # The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.
        
        # _, v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='U' ) # if upper else 'L'
        _, v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='L' ) # if upper else 'L'
        
        bv[batch_idx,:,:] = v
        
    # it is much faster to run symeig on CPU, back to GPU
    # bv = bv.cuda()
    return bv

def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
    return ys

def torch_skew_symmetric(v):

    # zero = torch.zeros_like(v[:, 0])

    # M = torch.stack([
    #     zero, -v[:, 2], v[:, 1],
    #     v[:, 2], zero, -v[:, 0],
    #     -v[:, 1], v[:, 0], zero,
    # ], dim=1)

    # return M
    
    zero = torch.zeros_like(v[0, 0])

    M = torch.stack([
        zero, -v[2, 0], v[1, 0],
        v[2, 0], zero, -v[0, 0],
        -v[1, 0], v[0, 0], zero,
    ], dim=0)

    return M


def tune_logits_with_ess_loss( config, logits, feedback_from_network_2):
    
    # Manuel Optimization
    logits_2 = logits - config.learning_rate * feedback_from_network_2
    
    return logits_2

    