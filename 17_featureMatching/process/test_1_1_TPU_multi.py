import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.datasets import get_dataset
from datasets.datasets import collate_fn2

from models.models import get_model
from models.models import get_model_structure

from samplers.CustomBatchSampler import get_sampler

from loss_module import loss_functions
from mAP import mAP
import checkpoint
import plots

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.debug.profiler as xp

def test(index, FLAGS):
    
    config                  = FLAGS['config']
    experiment_no           = FLAGS['experiment_no']
    learning_rate           = FLAGS['learning_rate']
    n_epochs                = FLAGS['n_epochs']
    num_workers             = FLAGS['num_workers']
    model_type              = FLAGS['model_type']
    en_grad_checkpointing   = FLAGS['en_grad_checkpointing']
    N_images_in_batch       = FLAGS['N_images_in_batch']
    N                       = FLAGS['N']
    batch_size              = FLAGS['batch_size']
    
    if(config.tpu_profiler == 'enable'):        
        server = xp.start_server(9012)

    world_size = xm.xrt_world_size()
    ordinal = xm.get_ordinal()    
    
    torch.manual_seed(1234)
    
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()  
        
    xm.master_print(f"Master Print by Process {index} using core {ordinal}/{world_size}")
    print( f"Core {ordinal}/{world_size} DEB PNT 0" )
    
    # Barrier to prevent master from exiting before workers connect.
    xm.rendezvous('init')
    
    model_width = config.model_width
    model = get_model( config, model_type, N, model_width, en_grad_checkpointing )
    model = model.to(device)

    print( f"Core {ordinal}/{world_size} DEB PNT 1" )
    
    # if xm.is_master_ordinal():
    #     device_for_model_structure = 'cpu'
    #     get_model_structure( config, device_for_model_structure, model, N, model_width, en_grad_checkpointing)        
    #     model = model.to(device)

    print( f"Core {ordinal}/{world_size} DEB PNT 2" )
    xm.rendezvous('get_model_structure')

    checkpoint_files = checkpoint.get_all_checkpoint_files_for_test(config)
    
    success_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 4) )
    loss_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 3) )
    proc_time_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks) )
    mAP_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 3, 360) )
    
    if(config.n_chunks == 1):
        dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=0, world_size=world_size, ordinal=ordinal )
        
        total_dataset_size_test = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( len(dataset_test), dtype=torch.int64).to(device) )
        
        sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size, common_dataset_size = total_dataset_size_test.item() // world_size )
            
        dataloader_test = DataLoader(   dataset = dataset_test,
                                        sampler = sampler_test,
                                        pin_memory = True,
                                        num_workers = num_workers,
                                        collate_fn=collate_fn2,)
    
    for epoch in range( len(checkpoint_files) ):
        
        model = checkpoint.load_test_checkpoint( config, device, model, checkpoint_file_with_path=checkpoint_files[epoch] )
    
        for chunk in range(0, config.n_chunks):
        
            loss_cls_test = 0
            loss_geo_test = 0
            loss_ess_test = 0
            loss_count_test = 0       
            
            confusion_matrix_at_epoch_test_device  = torch.zeros( (2,2), device = device, requires_grad = False )
            
### Generating dataset, sampler and dataloader for the current test chunk
            
            if(config.n_chunks > 1):
                dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=chunk, world_size=world_size, ordinal=ordinal )
                
                total_dataset_size_test = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( len(dataset_test), dtype=torch.int64).to(device) )
                
                sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size, common_dataset_size = total_dataset_size_test.item() // world_size )
                    
                dataloader_test = DataLoader(   dataset = dataset_test,
                                                sampler = sampler_test,
                                                pin_memory = True,
                                                num_workers = num_workers,
                                                collate_fn=collate_fn2,)
                
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
                    
                    err_q, err_t, err_qt = mAP.calculate_err_q_err_t( config=config, xs_ess=data['xs_ess'], R=data['R'], t=data['t'], E_hat=e_hat, y_hat=logits.detach() )
                    mAP_checkpoint = checkpoint.update_mAP_checkpoint(mAP_checkpoint, err_q, err_t, err_qt, epoch=epoch, chunk=chunk)
                            
                    if( ( (i*batch_size) % 100000 ) > ( ((i+1)*batch_size) % 100000 ) or (i+1) == len(dataloader_test) ):
                        
                        tot_it_test = torch.sum(confusion_matrix_at_epoch_test_device)
                        acc_test = torch.sum(confusion_matrix_at_epoch_test_device[0,0]+confusion_matrix_at_epoch_test_device[1,1]) / tot_it_test * 100
                        pre_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[:,1]) * 100
                        rec_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[1,:]) * 100
                        f1_test = 2 * pre_test * rec_test / ( pre_test + rec_test )
                            
                        xm.master_print("Exp {} Test Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                                .format(    experiment_no,
                                            epoch,
                                            len(checkpoint_files)-1,
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

            success_checkpoint[0, epoch, chunk, :] = np.array([acc_test.detach().cpu().numpy(), pre_test.detach().cpu().numpy(), rec_test.detach().cpu().numpy(), f1_test.detach().cpu().numpy()])
            loss_checkpoint[0, epoch, chunk, :] = np.array([loss_cls_test, loss_geo_test, loss_ess_test])
            proc_time_checkpoint[0, epoch, chunk] = time.perf_counter() - start_time_test
            
    
    xm.master_print(f'success_checkpoint before merging is {success_checkpoint[-1,-1,-1,0]}')
    
    success_checkpoint = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( success_checkpoint, dtype=torch.float64).to(device) )
    loss_checkpoint = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( loss_checkpoint, dtype=torch.float64).to(device) )
    proc_time_checkpoint = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( proc_time_checkpoint, dtype=torch.float64).to(device) )
    mAP_checkpoint = xm.all_reduce( reduce_type = xm.REDUCE_SUM, inputs = torch.tensor( mAP_checkpoint, dtype=torch.float64).to(device) )
    
    xm.master_print(f'success_checkpoint after summation is {success_checkpoint[-1,-1,-1,0]}')
    
    success_checkpoint = success_checkpoint / world_size
    loss_checkpoint = loss_checkpoint / world_size
    proc_time_checkpoint = proc_time_checkpoint / world_size
    mAP_checkpoint = mAP_checkpoint / world_size
    
    xm.master_print(f'success_checkpoint after averaging is {success_checkpoint[-1,-1,-1,0]}')
    
    success_checkpoint = success_checkpoint.detach().cpu().numpy()
    loss_checkpoint = loss_checkpoint.detach().cpu().numpy()
    proc_time_checkpoint = proc_time_checkpoint.detach().cpu().numpy()
    mAP_checkpoint = mAP_checkpoint.detach().cpu().numpy()
    
    if xm.is_master_ordinal():
        
        plots.plot_success_and_loss( config, epoch, config.n_chunks-1, success_checkpoint, loss_checkpoint)
        
        plots.plot_mAP( config, epoch, config.n_chunks-1, mAP_checkpoint, ref_angles = [5, 10, 20])   
        
        plots.plot_proc_time( config, epoch, config.n_chunks-1, proc_time_checkpoint)
        
        checkpoint.save_test_checkpoint( config, success_checkpoint, loss_checkpoint, proc_time_checkpoint, mAP_checkpoint)
        
    xm.rendezvous('update_plots_and_checkpoints')    
    
    xm.master_print("-" * 40)
        
    return 0
