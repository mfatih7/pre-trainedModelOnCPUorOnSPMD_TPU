import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.datasets import get_dataset
from datasets.datasets import collate_fn2

from models.models import get_model
from models.models import get_model_structure

from samplers.CustomBatchSampler import get_sampler

from optimizer.optimizer import get_optimizer

from loss_module import loss_functions
from mAP import mAP
import checkpoint
import plots

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp

from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs

def test( 
            config,
            experiment_no,
            learning_rate,
            n_epochs,
            num_workers,
            model_type,
            en_grad_checkpointing,
            N_images_in_batch,
            N,
            batch_size,
            optimizer_type,
            chkpt_mgr, ):
    
    if(config.tpu_profiler == 'enable'):        
        server = xp.start_server(9012)
    
    device = xm.xla_device()
    
    model_width = config.model_width
    model = get_model( config, model_type, N, model_width, en_grad_checkpointing )
    
    # device_for_model_structure = 'cpu'
    # get_model_structure( config, device_for_model_structure, model, N, model_width, en_grad_checkpointing)
    
    model = model.to(device)

    optimizer = get_optimizer( config, optimizer_type, model, learning_rate, )
    
### SPMD ###############################################################
    
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    
    if(config.spmd_type == 'model'):
    
        mesh_shape = (2, num_devices // 2, 1, 1)
        
        mesh = xs.Mesh(device_ids, mesh_shape, ('w', 'x', 'y', 'z'))
        partition_spec = (0, 1, 2, 3)  # Apply sharding along all axes
        
        for name, layer in model.named_modules():
            if ( 'conv2d' in name ):
              xs.mark_sharding(layer.weight, mesh, partition_spec)
        
    elif(config.spmd_type == 'batch'):
        mesh_shape = (num_devices, 1, 1, 1)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
    
########################################################################
    
    n_steps = checkpoint.get_number_of_checkpoint_files_for_test_spmd(config, chkpt_mgr)
    
    success_checkpoint = np.zeros( (1, n_steps, config.n_chunks, 4) )
    loss_checkpoint = np.zeros( (1, n_steps, config.n_chunks, 3) )
    proc_time_checkpoint = np.zeros( (1, n_steps, config.n_chunks) )
    mAP_checkpoint = np.zeros( (1, n_steps, config.n_chunks, 3, 360) )
    
    if(config.n_chunks == 1):
        dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=0 )
        
        sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size )
            
        dataloader_test = DataLoader(   dataset = dataset_test,
                                        sampler = sampler_test,
                                        pin_memory = True,
                                        num_workers = num_workers,
                                        collate_fn=collate_fn2,)
                                        
        if(config.spmd_type == 'batch'):
            mp_dataloader_test = pl.MpDeviceLoader( dataloader_test, device, input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)), )            
            dataloader_test = mp_dataloader_test
    
    for epoch in range( n_steps ):
        
        model = checkpoint.load_test_checkpoint_spmd( config, device, model, optimizer, step=epoch, chkpt_mgr=chkpt_mgr, )
    
        for chunk in range(0, config.n_chunks):
        
            loss_cls_test = 0
            loss_geo_test = 0
            loss_ess_test = 0
            loss_count_test = 0       
            
            confusion_matrix_at_epoch_test_device  = torch.zeros( (2,2), device = device, requires_grad = False )
            
### Generating dataset, sampler and dataloader for the current test chunk
            
            if(config.n_chunks > 1):
                dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=chunk )
                
                sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size )
                    
                dataloader_test = DataLoader(   dataset = dataset_test,
                                                sampler = sampler_test,
                                                pin_memory = True,
                                                num_workers = num_workers,
                                                collate_fn=collate_fn2,)
                                                
                if(config.spmd_type == 'batch'):
                    mp_dataloader_test = pl.MpDeviceLoader( dataloader_test, device, input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)), )            
                    dataloader_test = mp_dataloader_test
                
            start_time_test = time.perf_counter()       
                
            model.eval()  # Sets the model to evaluation mode
            with torch.no_grad():        
                for i, data in enumerate(dataloader_test):
                    
                    xs_device = data['xs'].to(device)
                    labels_device = data['ys'].to(device)
                    
                    xs_ess =  data['xs_ess'].to(device)
                    R_device =  data['R'].to(device)
                    t_device =  data['t'].to(device)
                    virtPt_device =  data['virtPt'].to(device) 
                    
                    logits = model(xs_device)
                    
                    classif_loss = loss_functions.get_losses( config, device, labels_device, logits)
                    
                    geo_loss, ess_loss, e_hat = loss_functions.calculate_ess_loss_and_L2loss( config, logits, xs_ess, R_device, t_device, virtPt_device )
                    
                    confusion_matrix_at_epoch_test_device[0,0] += torch.sum( torch.logical_and( logits<0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[0,1] += torch.sum( torch.logical_and( logits>0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[1,0] += torch.sum( torch.logical_and( logits<0, labels_device<config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[1,1] += torch.sum( torch.logical_and( logits>0, labels_device<config.obj_geod_th ) )

## BARRIER TO TRIG IR GRAPH EXECUTION #########################################################################
                    xm.mark_step()
###############################################################################################################   

                    loss_cls_test = loss_cls_test * loss_count_test + classif_loss.detach().cpu().numpy() * batch_size
                    loss_ess_test = loss_ess_test * loss_count_test + ess_loss.detach().cpu().numpy() * N
                    loss_geo_test = loss_geo_test * loss_count_test + geo_loss.detach().cpu().numpy() * N
                    loss_count_test = loss_count_test + batch_size
                    loss_cls_test = loss_cls_test / loss_count_test
                    loss_ess_test = loss_ess_test / loss_count_test  
                    loss_geo_test = loss_geo_test / loss_count_test
                    
                    err_q, err_t, err_qt = mAP.calculate_err_q_err_t( config=config,
                                                                      xs_ess=data['xs_ess'].cpu(),
                                                                      R=data['R'].cpu(),
                                                                      t=data['t'].cpu(),
                                                                      E_hat=e_hat.cpu(),
                                                                      y_hat=logits.detach().cpu() )
                    mAP_checkpoint = checkpoint.update_mAP_checkpoint(mAP_checkpoint, err_q, err_t, err_qt, epoch=epoch, chunk=chunk)
                            
                    if( ( (i*batch_size) % 1000000 ) > ( ((i+1)*batch_size) % 1000000 ) or (i+1) == len(dataloader_test) ):
                        
                        tot_it_test = torch.sum(confusion_matrix_at_epoch_test_device)
                        acc_test = torch.sum(confusion_matrix_at_epoch_test_device[0,0]+confusion_matrix_at_epoch_test_device[1,1]) / tot_it_test * 100
                        pre_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[:,1]) * 100
                        rec_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[1,:]) * 100
                        f1_test = 2 * pre_test * rec_test / ( pre_test + rec_test )
                            
                        print("Exp {} Test Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                                .format(    experiment_no,
                                            epoch,
                                            n_steps-1,
                                            chunk,
                                            config.n_chunks-1,
                                            i,
                                            len(dataloader_test)-1,
                                            learning_rate,
                                            loss_cls_test,
                                            loss_geo_test,
                                            loss_ess_test,
                                            int(torch.sum(confusion_matrix_at_epoch_test_device[0,0]+confusion_matrix_at_epoch_test_device[1,1])),
                                            int(tot_it_test),
                                            acc_test,
                                            pre_test,
                                            rec_test,
                                            f1_test,
                                            ) )
## BARRIER TO TRIG IR GRAPH EXECUTION #########################################################################
                    xm.mark_step()
###############################################################################################################

            success_checkpoint[0, epoch, chunk, :] = np.array([acc_test.detach().cpu().numpy(), pre_test.detach().cpu().numpy(), rec_test.detach().cpu().numpy(), f1_test.detach().cpu().numpy()])
            loss_checkpoint[0, epoch, chunk, :] = np.array([loss_cls_test, loss_geo_test, loss_ess_test])
            proc_time_checkpoint[0, epoch, chunk] = time.perf_counter() - start_time_test
            
            print("-" * 40)
        
    plots.plot_success_and_loss( config, epoch, config.n_chunks-1, success_checkpoint, loss_checkpoint)
    
    plots.plot_mAP( config, epoch, config.n_chunks-1, mAP_checkpoint, ref_angles = [5, 10, 20])   
    
    plots.plot_proc_time( config, epoch, config.n_chunks-1, proc_time_checkpoint)       
    
    checkpoint.save_test_checkpoint( config, success_checkpoint, loss_checkpoint, proc_time_checkpoint, mAP_checkpoint)
            
    return 0
