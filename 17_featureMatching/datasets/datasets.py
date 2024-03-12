# import cProfile
# import pstats

import sys
sys.path.append('..')

import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import h5py
import pickle

def collate_fn2(batch):
    
    batch_dict = batch[0]    
    for key in batch_dict:        
        batch_dict[key] = torch.from_numpy(batch_dict[key]).float()    
    return batch_dict

def collate_fn(batch):
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['Rs'], data['ts'], data['xs'], data['ys'], data['virtPts'], data['sides']  = [], [], [], [], [], []
    for sample in batch:
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        data['virtPts'].append(sample['virtPt'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp, replace=False)
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'][sub_idx,:])
        else:
            data['xs'].append(sample['xs'])
            data['ys'].append(sample['ys'])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'])

    for key in ['Rs', 'ts', 'xs', 'ys', 'virtPts']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    if data['sides'] != []:
        data['sides'] = torch.from_numpy(np.stack(data['sides'])).float()
    return data

class Collate_Fn(Dataset):
    def __init__(self, N, n_to_1_en ):
        
        self.N = N
        self.n_to_1_en = n_to_1_en
        
    def collate_fn(self, batch):
        
        data = {}
        data['Rs'], data['ts'], data['xs'], data['ys'], data['virtPts'], data['sides']  = [], [], [], [], [], []
        for sample in batch:
            data['Rs'].append(sample['R'])
            data['ts'].append(sample['t'])
            data['virtPts'].append(sample['virtPt'])
            
            size_to_be_added = sample['xs'].shape[1]
            sub_idx = np.zeros(self.N, dtype='int32')
            cur_num_kp = 0
            while 1:
                if( cur_num_kp+size_to_be_added <= self.N):
                    sub_idx[cur_num_kp:cur_num_kp+size_to_be_added] = np.random.choice(size_to_be_added, size_to_be_added, replace=False)
                    cur_num_kp = cur_num_kp+size_to_be_added
                else:
                    sub_idx[cur_num_kp:self.N] = np.random.choice(size_to_be_added, self.N-cur_num_kp, replace=False)
                    break
                
            data['xs'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
            if len(sample['side']) != 0:
                data['sides'].append(sample['side'][sub_idx,:])
                
        if( self.n_to_1_en == True ):      
            
            tmp_x = np.zeros( (self.N, 2, self.N, 4), dtype='float32' )
            if len(sample['side']) != 0:
                tmp_sides = np.zeros( (self.N, 2, self.N, 2), dtype='float32' )
                
            for j in range( len( data['xs'] ) ):
                tmp_x[:,1:2,:,:] = np.tile( data['xs'][j], (self.N, 1, 1, 1) )
                if len(sample['side']) != 0:
                    tmp_sides[:,1:2,:,:] = np.tile( data['sides'][j], (self.N, 1, 1, 1) )
                
                for k in range( self.N ):
                    row_x = data['xs'][j][0][k][:]
                    tmp_x[k, 0:1,:,:] = np.tile( row_x, (1, 1, self.N, 1) )                    
                    if len(sample['side']) != 0:
                        row_side = data['sides'][j][k][:]
                        tmp_sides[k,0:1,:,:] = np.tile( row_side, (1, 1, self.N, 1) )
                
                data['xs'][j] = tmp_x
                if len(sample['side']) != 0:
                    data['sides'][j] = tmp_sides            
        
        for key in ['Rs', 'ts', 'xs', 'ys', 'virtPts']:
            data[key] = torch.from_numpy(np.stack(data[key])).float()
        if data['sides'] != []:
            data['sides'] = torch.from_numpy(np.stack(data['sides'])).float()
        return data    
        
        ## For debugging numpy variables without changing to torch
        # for key in ['Rs', 'ts', 'xs', 'ys', 'virtPts']:
        #     data[key] = np.stack(data[key])
        # if data['sides'] != []:
        #     data['sides'] = np.stack(data['sides'])
        # return data

class CorrespondencesDataset(Dataset):
    def __init__(self, config, N_images_in_batch, N, batch_size, train_val_test, chunk, world_size, ordinal):
        
        # print('Dataset __init__() method called')       
        
        self.config = config
        self.train_val_test = train_val_test
        
        if( self.config.use_hdf5_or_picle == 'hdf5' ):
            if(chunk==0 or not('self.indices_chunks' in locals()) ):
                self.indices_chunks = self.get_indices_for_chunks( )
            if(self.train_val_test == 'train' or (self.train_val_test == 'val' and self.config.validation_chunk_or_all == 'chunk') ):
                indices_chunk = np.array( self.indices_chunks[chunk] )
                data_chunk = self.get_data_for_chunk_from_hdf5( indices_chunk)
            # For Validation use all
            else:
                indices_chunk_merged = [item for sublist in self.indices_chunks for item in sublist]
                indices_chunk = np.array(indices_chunk_merged).reshape( (-1) )
                data_chunk = self.get_data_for_chunk_from_hdf5( indices_chunk )
        elif( self.config.use_hdf5_or_picle == 'pickle' ):
            
            if( config.device != 'tpu' or config.tpu_cores != 'multi' ):
            
                if(self.train_val_test == 'train' or ( self.train_val_test == 'val' and self.config.validation_chunk_or_all == 'chunk') ):
                    file_indices = list( range( chunk, config.n_chunk_files, config.n_chunks ) )
                else:
                    file_indices = list( range( 0, config.n_chunk_files, 1 ) )
    
                for file_index_id, file_index in enumerate( file_indices ):
                    tmp = self.get_data_for_chunk_from_pickle( file_index )
                    if(file_index_id==0):
                        data_chunk = tmp
                    else:
                        for e in range( len(tmp) ):
                            data_chunk[e].extend(tmp[e])
                            
                print(f'File indices for {train_val_test} dataset are {file_indices}')
            else:
                
                n_chunk_files_per_chunk = int( config.n_chunk_files/config.n_chunks )
                
                if(self.train_val_test == 'train' or ( self.train_val_test == 'val' and self.config.validation_chunk_or_all == 'chunk') ):
                    file_indices = list( range( chunk*n_chunk_files_per_chunk+ordinal,
                                                (chunk+1)*n_chunk_files_per_chunk,
                                                world_size ) )
                else:
                    file_indices = list( range( ordinal,
                                                config.n_chunk_files,
                                                world_size ) )
                    
                print(f'File indices for {train_val_test} dataset of TPU core {ordinal}/{(world_size-1)} are {file_indices}')
    
                for file_index_id, file_index in enumerate( file_indices ):
                    tmp = self.get_data_for_chunk_from_pickle( file_index )
                    if(file_index_id==0):
                        data_chunk = tmp
                    else:
                        for e in range( len(tmp) ):
                            data_chunk[e].extend(tmp[e])
        
        self.input_type         = config.input_type
        self.N_images_in_batch  = N_images_in_batch
        self.N                  = N
        self.batch_size         = batch_size
        
        self.size_dataset = len( data_chunk[0] )
        
        self.xs_chunk = data_chunk[0]
        self.ys_chunk = data_chunk[1]
        self.ratios_chunk = data_chunk[2]
        self.mutuals_chunk = data_chunk[3]
        self.R_chunk = data_chunk[4]
        self.t_chunk = data_chunk[5]

        if(self.config.en_tl_on_cpu==1):
            import models.models_tl_cpu
            self.tl_model = models.models_tl_cpu.get_model( self.config, self.N, )
            self.tl_model.eval()
        
        print('Size of ' + self.train_val_test + ' dataset is ' + str(self.size_dataset) )

    def correctMatches(self, e_gt):
        step = 0.1
        xx,yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
        # Points in first image before projection
        pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
        # Points in second image before projection
        pts2_virt_b = np.float32(pts1_virt_b)
        pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1,-1,2), pts2_virt_b.reshape(1,-1,2)

        pts1_virt_b, pts2_virt_b = cv2.correctMatches(e_gt.reshape(3,3), pts1_virt_b, pts2_virt_b)

        return pts1_virt_b.squeeze(), pts2_virt_b.squeeze()
    
    def __getitem__(self, indices):
        
        if(self.input_type == '1_to_1'):
# For GPU operation to fit the GPU memory
            if( self.N_images_in_batch == 1 and self.N > self.batch_size ):
                
                if( indices[0] < ( 1 * self.size_dataset * self.N ) ):    # 0,2 -> data and distance to epipolar line(label)
                
                    indices = np.array(indices)
                
                    sample = int( indices[0] / self.N )
                    batch_start_ind_in_sample = indices[0] % self.N
                    
                    # print( str(sample) + ' ' + str(batch_start_ind_in_sample) )
                
                    # xs = np.asarray( self.data['xs'][str(sample)] )
                    # ys = np.asarray( self.data['ys'][str(sample)] )
                    
                    xs = self.xs_chunk[sample]
                    ys = self.ys_chunk[sample]
                    
                    if self.config.use_ratio == 0 and self.config.use_mutual == 0:
                        pass
                    elif self.config.use_ratio == 1 and self.config.use_mutual == 0:
                        # mask = np.asarray(self.data['ratios'][str(sample)]).reshape(-1)  < self.config.ratio_test_th
                        mask = self.ratios_chunk[sample].reshape(-1)  < self.config.ratio_test_th
                        xs = xs[:,mask,:]
                        ys = ys[mask,:]
                    elif self.config.use_ratio == 0 and self.config.use_mutual == 1:
                        # mask = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1).astype(bool)
                        mask = self.mutuals_chunk[sample].reshape(-1).astype(bool)
                        xs = xs[:,mask,:]
                        ys = ys[mask,:]
                    elif self.config.use_ratio == 0 and self.config.use_mutual == 2:
                        # side = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1, 1)
                        side = self.mutuals_chunk[sample].reshape(-1, 1)
                    elif self.config.use_ratio == 2 and self.config.use_mutual == 0:
                        # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                        side = self.ratios_chunk[sample].reshape(-1,1) 
                    elif self.config.use_ratio == 2 and self.config.use_mutual == 2:
                        # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                        # side = np.concatenate( ( side, np.asarray(self.data['mutuals'][str(sample)]).reshape(-1,1) ), axis=-1)
                        
                        side = self.ratios_chunk[sample].reshape(-1,1)
                        side = np.concatenate( ( side, self.mutuals_chunk[sample].reshape(-1,1) ), axis=-1)
                    else:
                        raise NotImplementedError
                    
                    if not( self.config.use_ratio == 0 and self.config.use_mutual == 0 ):
                        xs = np.concatenate( (xs, side[np.newaxis,:,:] ), axis=-1)
                    
                    context = xs
                    
                    while True:
                        if(context.shape[1]>=self.N):
                            context = context[:,:self.N,:]
                            break
                        elif(context.shape[1]<self.N):
                            context = np.tile( context, (1, 2, 1) )
                
                    xs_1 = np.tile( context, (self.batch_size, 1, 1) )
                    xs_1 = xs_1[:, np.newaxis, :, :]
                    
                    candidate = np.zeros( (self.batch_size, self.N, context.shape[2]), dtype=np.float32 )            
                    for cand_ind_index, cand_ind in enumerate( range(batch_start_ind_in_sample, batch_start_ind_in_sample+self.batch_size) ):
                        candidate[cand_ind_index,:,:] = np.repeat(context[:,cand_ind:cand_ind+1,:], self.N, axis=1)
                    
                    xs_2 = candidate[:, np.newaxis, :, :]
                    
                    xs = np.concatenate( (xs_1, xs_2), axis=1 )
                    
                    
                    while True:
                        if(ys.shape[0]>=self.N):
                            ys = ys[:self.N,:]
                            break
                        elif(ys.shape[0]<self.N):
                            ys = np.tile( ys, (2, 1) )
                            
                    ys = ys[batch_start_ind_in_sample:batch_start_ind_in_sample+self.batch_size, :]
                    
                    return { 'xs':xs, 'ys':ys }

                elif( indices[0] < ( 2 * self.size_dataset * self.N ) ):    # 1 -> virt_pts(essential loss), R,t(essential loss 2)
                        
                    indices = np.array(indices) - ( 1 * self.size_dataset * self.N )
                    sample = int( indices[0] / self.N )
                    
                    # print( str(sample) )
                    
                    # xs_ess = np.asarray( self.data['xs'][str(sample)] )
                    xs_ess = self.xs_chunk[sample]
                    
                    while True:
                        if(xs_ess.shape[1]>=self.N):
                            xs_ess = xs_ess[:,:self.N,:]
                            break
                        elif(xs_ess.shape[1]<self.N):
                            xs_ess = np.tile( xs_ess, (1, 2, 1) )
                    
                    # R = np.asarray(self.data['Rs'][str(sample)])
                    # t = np.asarray(self.data['ts'][str(sample)])
                    
                    R = self.R_chunk[sample]
                    t = self.t_chunk[sample]
                    
                    t_float64 = t.astype('float64')
                    t_skew_symmetric = np.reshape( np.array( [ np.array( [ 0 ] ), -t_float64[2], t_float64[1] ,
                                                                t_float64[2], np.array( [ 0 ] ), -t_float64[0] ,
                                                                -t_float64[1], t_float64[0], np.array( [ 0 ] ) ]  ), (3,3) )

                    e_gt_unnorm = np.reshape( np.matmul( t_skew_symmetric, np.reshape( R.astype('float64'), (3, 3) ) ), (3, 3) )            
                    e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)

                    pts1_virt, pts2_virt = self.correctMatches(e_gt)

                    pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')
                    
                    return {'xs_ess': xs_ess, 'R':R, 't':t, 'virtPt':pts_virt }        
                    
                else:
                    raise ValueError(f"Index {indices[0]} is out of range")

# For TPU operation, all data related to an image pair pumped within the same batch                
            elif( self.N_images_in_batch >= 1 and self.N == self.batch_size ):
                
                if(len(indices)==1):
                    
                    sample = indices[0]             
                    
                    xs = self.xs_chunk[sample]
                    ys = self.ys_chunk[sample]
                    
                    if self.config.use_ratio == 0 and self.config.use_mutual == 0:
                        pass
                    elif self.config.use_ratio == 1 and self.config.use_mutual == 0:
                        # mask = np.asarray(self.data['ratios'][str(sample)]).reshape(-1)  < self.config.ratio_test_th
                        mask = self.ratios_chunk[sample].reshape(-1)  < self.config.ratio_test_th
                        xs = xs[:,mask,:]
                        ys = ys[mask,:]
                    elif self.config.use_ratio == 0 and self.config.use_mutual == 1:
                        # mask = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1).astype(bool)
                        mask = self.mutuals_chunk[sample].reshape(-1).astype(bool)
                        xs = xs[:,mask,:]
                        ys = ys[mask,:]
                    elif self.config.use_ratio == 0 and self.config.use_mutual == 2:
                        # side = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1, 1)
                        side = self.mutuals_chunk[sample].reshape(-1, 1)
                    elif self.config.use_ratio == 2 and self.config.use_mutual == 0:
                        # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                        side = self.ratios_chunk[sample].reshape(-1,1) 
                    elif self.config.use_ratio == 2 and self.config.use_mutual == 2:
                        # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                        # side = np.concatenate( ( side, np.asarray(self.data['mutuals'][str(sample)]).reshape(-1,1) ), axis=-1)
                        
                        side = self.ratios_chunk[sample].reshape(-1,1)
                        side = np.concatenate( ( side, self.mutuals_chunk[sample].reshape(-1,1) ), axis=-1)
                    else:
                        raise NotImplementedError
                    
                    if not( self.config.use_ratio == 0 and self.config.use_mutual == 0 ):
                        xs = np.concatenate( (xs, side[np.newaxis,:,:] ), axis=-1)
                    
                    context = xs
                    
                    while True:
                        if(context.shape[1]>=self.N):
                            context = context[:,:self.N,:]
                            break
                        elif(context.shape[1]<self.N):
                            context = np.tile( context, (1, 2, 1) )
                    
                    if( self.config.en_batch_build_with_context == 1 ):

                        if(self.config.en_tl_on_cpu==1):
                            x_tl = context[np.newaxis, :, :, :]

                            with torch.no_grad():
                                context = self.tl_model( torch.from_numpy( x_tl ) )
                            context = context.numpy()
                            context = np.squeeze(context, axis=1)            
                                
                        xs_1 = np.tile( context, (self.N, 1, 1) )
                        xs_1 = xs_1[:, np.newaxis, :, :]
                        
                        candidate = np.zeros( (self.N, self.N, context.shape[2]), dtype=np.float32 )            
                        for cand_ind in range( self.N ):
                            candidate[cand_ind,:,:] = np.repeat(context[:,cand_ind:cand_ind+1,:], self.N, axis=1)
                        
                        xs_2 = candidate[:, np.newaxis, :, :]
                        
                        xs = np.concatenate( (xs_1, xs_2), axis=1 )
                        
                    elif( self.config.en_batch_build_with_context == 0 ):
                        
                        xs = context[np.newaxis, :, :, :]
                    
                    while True:
                        if(ys.shape[0]>=self.N):
                            ys = ys[:self.N,:]
                            break
                        elif(ys.shape[0]<self.N):
                            ys = np.tile( ys, (2, 1) )
                    
                    xs_ess = self.xs_chunk[sample]
                    
                    while True:
                        if(xs_ess.shape[1]>=self.N):
                            xs_ess = xs_ess[:,:self.N,:]
                            break
                        elif(xs_ess.shape[1]<self.N):
                            xs_ess = np.tile( xs_ess, (1, 2, 1) )
                    
                    R = self.R_chunk[sample]
                    t = self.t_chunk[sample]
                    
                    t_float64 = t.astype('float64')
                    t_skew_symmetric = np.reshape( np.array( [ np.array( [ 0 ] ), -t_float64[2], t_float64[1] ,
                                                                t_float64[2], np.array( [ 0 ] ), -t_float64[0] ,
                                                                -t_float64[1], t_float64[0], np.array( [ 0 ] ) ]  ), (3,3) )
    
                    e_gt_unnorm = np.reshape( np.matmul( t_skew_symmetric, np.reshape( R.astype('float64'), (3, 3) ) ), (3, 3) )            
                    e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)
    
                    pts1_virt, pts2_virt = self.correctMatches(e_gt)
    
                    pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')
                    
                    return { 'xs':xs, 'ys':ys, 'xs_ess': xs_ess, 'R':R, 't':t, 'virtPt':pts_virt } 
                    # return { 'xs':xs, 'ys':ys }   
                    
                else:
                    raise NotImplementedError(f"The feature '{self.input_type}' is not implemented yet.")
                
        elif(self.input_type == 'n_to_n'):
            
            xs_          = np.zeros( (self.N_images_in_batch, 1, self.N, self.config.model_width), dtype=np.float32 )
            ys_          = np.zeros( (self.N_images_in_batch, self.N, 1), dtype=np.float32 )
            xs_ess_      = np.zeros( (self.N_images_in_batch, 1, self.N, 4), dtype=np.float32 )
            R_           = np.zeros( (self.N_images_in_batch, 3, 3), dtype=np.float32 )
            t_           = np.zeros( (self.N_images_in_batch, 3, 1), dtype=np.float32 )
            pts_virt_    = np.zeros( (self.N_images_in_batch, 400, 4), dtype=np.float32 )
            
            ######################################################################################################
            
            for sample_idx, sample in enumerate(indices):
                
                xs = self.xs_chunk[sample]
                ys = self.ys_chunk[sample]
                
                if self.config.use_ratio == 0 and self.config.use_mutual == 0:
                    pass
                elif self.config.use_ratio == 1 and self.config.use_mutual == 0:
                    # mask = np.asarray(self.data['ratios'][str(sample)]).reshape(-1)  < self.config.ratio_test_th
                    mask = self.ratios_chunk[sample].reshape(-1)  < self.config.ratio_test_th
                    xs = xs[:,mask,:]
                    ys = ys[mask,:]
                elif self.config.use_ratio == 0 and self.config.use_mutual == 1:
                    # mask = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1).astype(bool)
                    mask = self.mutuals_chunk[sample].reshape(-1).astype(bool)
                    xs = xs[:,mask,:]
                    ys = ys[mask,:]
                elif self.config.use_ratio == 0 and self.config.use_mutual == 2:
                    # side = np.asarray(self.data['mutuals'][str(sample)]).reshape(-1, 1)
                    side = self.mutuals_chunk[sample].reshape(-1, 1)
                elif self.config.use_ratio == 2 and self.config.use_mutual == 0:
                    # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                    side = self.ratios_chunk[sample].reshape(-1,1) 
                elif self.config.use_ratio == 2 and self.config.use_mutual == 2:
                    # side = np.asarray(self.data['ratios'][str(sample)]).reshape(-1,1)
                    # side = np.concatenate( ( side, np.asarray(self.data['mutuals'][str(sample)]).reshape(-1,1) ), axis=-1)
                    
                    side = self.ratios_chunk[sample].reshape(-1,1)
                    side = np.concatenate( ( side, self.mutuals_chunk[sample].reshape(-1,1) ), axis=-1)
                else:
                    raise NotImplementedError
                
                if not( self.config.use_ratio == 0 and self.config.use_mutual == 0 ):
                    xs = np.concatenate( (xs, side[np.newaxis,:,:] ), axis=-1)
                
                context = xs
                
                xs_ess = self.xs_chunk[sample]
                
                while True:
                    if(context.shape[1]>=self.N):
                        context = context[:,:self.N,:]
                        ys = ys[:self.N,:]
                        xs_ess = xs_ess[:,:self.N,:]
                        break
                    elif(context.shape[1]<self.N):
                        context = np.tile( context, (1, 2, 1) )
                        ys = np.tile( ys, (2, 1) )
                        xs_ess = np.tile( xs_ess, (1, 2, 1) )
                
                R = self.R_chunk[sample]
                t = self.t_chunk[sample]
                
                t_float64 = t.astype('float64')
                t_skew_symmetric = np.reshape( np.array( [ np.array( [ 0 ] ), -t_float64[2], t_float64[1] ,
                                                            t_float64[2], np.array( [ 0 ] ), -t_float64[0] ,
                                                            -t_float64[1], t_float64[0], np.array( [ 0 ] ) ]  ), (3,3) )

                e_gt_unnorm = np.reshape( np.matmul( t_skew_symmetric, np.reshape( R.astype('float64'), (3, 3) ) ), (3, 3) )            
                e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)

                pts1_virt, pts2_virt = self.correctMatches(e_gt)

                pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')

                xs_[sample_idx, :, :, :] = context
                ys_[sample_idx, :, :] = ys
                xs_ess_[sample_idx, :, :] = xs_ess
                R_[sample_idx, :, :] = R
                t_[sample_idx, :, :] = t
                pts_virt_[sample_idx, :, :] = pts_virt
                
            return { 'xs':xs_, 'ys':ys_, 'xs_ess': xs_ess_, 'R':R_, 't':t_, 'virtPt':pts_virt_ }
                
                
        # xs = np.asarray(self.data['xs'][str(index)])
        # ys = np.asarray(self.data['ys'][str(index)])
        # R = np.asarray(self.data['Rs'][str(index)])
        # t = np.asarray(self.data['ts'][str(index)])
        # side = []
        # if self.config.use_ratio == 0 and self.config.use_mutual == 0:
        #     pass
        # elif self.config.use_ratio == 1 and self.config.use_mutual == 0:
        #     mask = np.asarray(self.data['ratios'][str(index)]).reshape(-1)  < self.config.ratio_test_th
        #     xs = xs[:,mask,:]
        #     ys = ys[mask,:]
        # elif self.config.use_ratio == 0 and self.config.use_mutual == 1:
        #     mask = np.asarray(self.data['mutuals'][str(index)]).reshape(-1).astype(bool)
        #     xs = xs[:,mask,:]
        #     ys = ys[mask,:]
        # elif self.config.use_ratio == 0 and self.config.use_mutual == 2:
        #     side = np.asarray(self.data['mutuals'][str(index)]).reshape(-1, 1)
        # elif self.config.use_ratio == 2 and self.config.use_mutual == 0:
        #     side = np.asarray(self.data['ratios'][str(index)]).reshape(-1,1)
        # elif self.config.use_ratio == 2 and self.config.use_mutual == 2:
        #     side.append(np.asarray(self.data['ratios'][str(index)]).reshape(-1,1)) 
        #     side.append(np.asarray(self.data['mutuals'][str(index)]).reshape(-1,1))
        #     side = np.concatenate(side,axis=-1)
        # else:
        #     raise NotImplementedError

        # t_float64 = t.astype('float64')
        # t_skew_symmetric = np.reshape( np.array( [ np.array( [ 0 ] ), -t_float64[2], t_float64[1] ,
        #                                            t_float64[2], np.array( [ 0 ] ), -t_float64[0] ,
        #                                            -t_float64[1], t_float64[0], np.array( [ 0 ] ) ]  ), (3,3) )

        # e_gt_unnorm = np.reshape( np.matmul( t_skew_symmetric, np.reshape( R.astype('float64'), (3, 3) ) ), (3, 3) )            
        # e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)

        # pts1_virt, pts2_virt = self.correctMatches(e_gt)

        # pts_virt = np.concatenate([pts1_virt, pts2_virt], axis=1).astype('float64')
        # return {'R':R, 't':t, 'xs':xs, 'ys':ys, 'virtPt':pts_virt, 'side':side}
        
        
    # def reset(self):
    #     if self.data is not None:
    #         self.data.close()
    #     self.data = None

    def __len__(self):
        return self.size_dataset       

    # def __del__(self):
    #     if self.data is not None:
    #         self.data.close()
    
    def get_indices_for_chunks( self ):
        
        if(self.train_val_test=='train'):
            n_image_pairs = self.config.n_image_pairs_train
        elif(self.train_val_test=='val'):
            n_image_pairs = self.config.n_image_pairs_val
        elif(self.train_val_test=='test'):
            n_image_pairs = self.config.n_image_pairs_test
            
        n_image_pairs_per_chunk = n_image_pairs / self.config.n_chunks
        
        indices_for_chunks = []
        for c in range(self.config.n_chunks):
            indices_for_chunks.append( list( range( math.floor(n_image_pairs_per_chunk*c), math.floor(n_image_pairs_per_chunk*(c+1) ) ) ) )
        return indices_for_chunks
    
    def get_data_for_chunk_from_hdf5( self, indices_chunk):
        
        if(self.config.input_data_storage_local_or_bucket=='local'):
            input_path_local_or_bucket = self.config.input_path_local
        elif(self.config.input_data_storage_local_or_bucket=='bucket'):
            input_path_local_or_bucket = self.config.input_path_bucket
            
        if(self.train_val_test=='train'):
            file_name = os.path.join( input_path_local_or_bucket, self.config.file_name_train )
        elif(self.train_val_test=='val'):
            file_name = os.path.join( input_path_local_or_bucket, self.config.file_name_val )
        elif(self.train_val_test=='test'):
            file_name = os.path.join( input_path_local_or_bucket, self.config.file_name_test )
            
        if(self.config.input_data_storage_local_or_bucket=='local'):
            data = h5py.File( file_name, 'r', libver='latest', swmr=True )
        elif(self.config.input_data_storage_local_or_bucket=='bucket'):        
            from google.cloud import storage
            import io
            
            storage_client = storage.Client()        
            blob = storage_client.bucket(self.config.bucket_name).blob(file_name)
            
            print(blob.name)
    
            hdf5_bytes = blob.download_as_bytes()        
            file_like_object = io.BytesIO(hdf5_bytes)
            data = h5py.File(file_like_object, 'r')        
        
        size_chunk = indices_chunk.shape[0]
    
        xs_chunk = [ np.asarray( data['xs'][str(indices_chunk[0])] ) ]
        ys_chunk = [ np.asarray( data['ys'][str(indices_chunk[0])] ) ]
        ratios_chunk = [ np.asarray( data['ratios'][str(indices_chunk[0])] ) ]
        mutuals_chunk = [ np.asarray( data['mutuals'][str(indices_chunk[0])] ) ]
        R_chunk = [ np.asarray( data['Rs'][str(indices_chunk[0])] ) ]
        t_chunk = [ np.asarray( data['ts'][str(indices_chunk[0])] ) ]
        
        xs_chunk = [ xs_chunk[0].copy() for _ in range(size_chunk) ]
        ys_chunk = [ ys_chunk[0].copy() for _ in range(size_chunk) ]
        ratios_chunk = [ ratios_chunk[0].copy() for _ in range(size_chunk) ]
        mutuals_chunk = [ mutuals_chunk[0].copy() for _ in range(size_chunk) ]
        R_chunk = [ R_chunk[0].copy() for _ in range(size_chunk) ]
        t_chunk = [ t_chunk[0].copy() for _ in range(size_chunk) ]
        
        # for i in range(self.size_chunk):
        #     print(str(i))
        #     self.xs_chunk[i] = np.asarray( self.data['xs'][str(indices_chunk[i])] )
        #     self.ys_chunk[i] = np.asarray( self.data['ys'][str(indices_chunk[i])] )
        #     self.ratios_chunk[i] = np.asarray(self.data['ratios'][str(indices_chunk[i])])
        #     self.mutuals_chunk[i] = np.asarray(self.data['mutuals'][str(indices_chunk[i])])
        #     self.R_chunk[i] = np.asarray(self.data['Rs'][str(indices_chunk[i])])
        #     self.t_chunk[i] = np.asarray(self.data['ts'][str(indices_chunk[i])])
            
        for i in range(size_chunk):                
            xs_chunk[i] = np.asarray( data['xs'][str(indices_chunk[i])] )
        print('Loading xs from hdf5 file completed')
            
        for i in range(size_chunk):
            ys_chunk[i] = np.asarray( data['ys'][str(indices_chunk[i])] )
        print('Loading ys from hdf5 file completed')
             
        for i in range(size_chunk):
            ratios_chunk[i] = np.asarray(data['ratios'][str(indices_chunk[i])])
        print('Loading ratios from hdf5 file completed')
             
        for i in range(size_chunk):
            mutuals_chunk[i] = np.asarray(data['mutuals'][str(indices_chunk[i])])
        print('Loading mutuals from hdf5 file completed')
             
        for i in range(size_chunk):
            R_chunk[i] = np.asarray(data['Rs'][str(indices_chunk[i])])
        print('Loading Rs from hdf5 file completed')
             
        for i in range(size_chunk):
            t_chunk[i] = np.asarray(data['ts'][str(indices_chunk[i])])
        print('Loading ts from hdf5 file completed')
        
        data.close()
        print('Size of Chunk ' + str(size_chunk) )
        
        return xs_chunk, ys_chunk, ratios_chunk, mutuals_chunk, R_chunk, t_chunk

    def get_data_for_chunk_from_pickle( self, chunk):
        
        if(self.config.input_data_storage_local_or_bucket=='local'):
            file_name_with_path = os.path.join(self.config.input_path_pickle_local, self.train_val_test + f'_{chunk:04d}' + '.pkl')
        
            with open(file_name_with_path, 'rb') as file:
                data = pickle.load(file)
                
        elif(self.config.input_data_storage_local_or_bucket=='bucket'):
            file_name_with_path = os.path.join(self.config.input_path_pickle_bucket, self.train_val_test + f'_{chunk:04d}' + '.pkl')
            
            from google.cloud import storage
            
            storage_client = storage.Client()
            
            blob = storage_client.bucket(self.config.bucket_name).blob(file_name_with_path)
    
            print(blob.name)
    
            # Download the blob as bytes
            pickle_bytes = blob.download_as_bytes()
    
            # Deserialize the bytes object
            data = pickle.loads(pickle_bytes)
            
        return data 

# Make sure that any custom collate_fn, worker_init_fn or dataset code is declared as top level
# definitions, outside of the __main__ check. This ensures that they are available in worker processes.
# (this is needed since functions are pickled as references only, not bytecode.)

def get_dataset( config, N_images_in_batch, N, batch_size, train_val_test, chunk, world_size=1, ordinal=0 ):    
    return CorrespondencesDataset(config, N_images_in_batch, N, batch_size, train_val_test, chunk, world_size, ordinal)



if __name__ == '__main__':

    # N_images_in_batch   = 1
    # N                   = 512
    # batch_size          = 64
    
    # N_images_in_batch   = 2
    # N                   = 512
    # batch_size          = 512
    
    N_images_in_batch   = 1
    N                   = 512
    batch_size          = 512
    
    # N_images_in_batch   = 32
    # N                   = 512
    # batch_size          = 32
    
    chunk = 0
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config    
    config = get_config()
    
    dataset = CorrespondencesDataset(config, N_images_in_batch, N, batch_size, 'train', chunk, world_size=1, ordinal=0 )
    
### GET DATA WITH DIRECTLY FROM DATASET 1_to_1 ######################################################################

    A_00 = dataset.__getitem__( [ 0 ] )           # Getting all for single sample
    A_10 = dataset.__getitem__( [ 0, 0+1 ] )      # Getting all for multiple sample

    A_10 = dataset.__getitem__( [ 0  ] )                        # Getting data and labels
    A_11 = dataset.__getitem__( [ 0 + dataset.__len__()*N ] )   # Getting R, t, xs_ess, virt_pts

### GET DATA WITH DIRECTLY FROM DATASET 1_to_1 ######################################################################

### GET DATA WITH DIRECTLY FROM DATASET n_to_n ######################################################################

    # A_00 = dataset.__getitem__( [number for number in range(0,32,1)] )

### GET DATA WITH DIRECTLY FROM DATASET n_to_n ######################################################################
    
### GET DATA WITH SAMPLER AND DATALOADER 1_to_1 or n_to_n ######################################################################

    # from samplers.CustomBatchSampler import get_sampler
    
    # sampler = get_sampler( config, dataset, N_images_in_batch, N, batch_size )
    
    # from torch.utils.data import DataLoader
    
    # dataloader = DataLoader(    dataset = dataset,
    #                             sampler = sampler,
    #                             pin_memory = True,
    #                             num_workers = 0,
    #                             collate_fn = collate_fn2,)

    # dataiter = iter(dataloader)
    # for sample in range(50):
        
    #     B = next(dataiter)
    #     print(B.keys())

### GET DATA WITH SAMPLER AND DATALOADER 1_to_1 or n_to_n ######################################################################

