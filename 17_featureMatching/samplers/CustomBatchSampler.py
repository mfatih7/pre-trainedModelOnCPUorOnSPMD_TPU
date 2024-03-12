from torch.utils.data import BatchSampler
import numpy as np

class CustomBatchSampler(BatchSampler):
    def __init__(self, config, n_image_pairs, train_val_test, N_images_in_batch, N, batch_size ):
        
        self.n_samples          = n_image_pairs
        self.train_val_test     = train_val_test
        self.input_type         = config.input_type
        self.N_images_in_batch  = N_images_in_batch
        self.N                  = np.int64( N )
        self.batch_size         = np.int64( batch_size )

        if(self.input_type == '1_to_1'):            
# For GPU operation to fit the GPU memory
            if( self.N_images_in_batch == 1 and self.N > self.batch_size ):
                if(train_val_test=='train'):
                    self.num_repeats    = np.int64( 3 )
                else:
                    self.num_repeats    = np.int64( 2 )
    
                self.batches_in_N       = np.int64( self.N / self.batch_size ) ### overflow error if int
                
# For TPU operation, all data related to an image pair pumped within the same batch
            elif( self.N_images_in_batch >= 1 and self.N == self.batch_size ):
                pass
            else:
                raise ValueError(f"The provided arguments are not valid: {self.input_type} {self.N_images_in_batch} {self.N} {self.batch_size}")
        elif(self.input_type == 'n_to_n'):
            
            if( self.N_images_in_batch == self.batch_size ):
                pass
            else:
                raise ValueError(f"The provided arguments are not valid: {self.N_images_in_batch} {self.batch_size}")
        else:
            raise ValueError(f"The provided argument is not valid: {self.input_type}")

    def __iter__(self):
        
        shuffled_samples = np.random.permutation( self.n_samples )  # Shuffle the samples
        
        if(self.input_type == '1_to_1'):            
            if( self.N_images_in_batch == 1 and self.N > self.batch_size ):
        
                for chosen_sample in shuffled_samples:
                    
                    if( self.num_repeats>2 ):                
                        count_repeats = [0, 1, 2]
                    else:
                        count_repeats = [0, 1]
                    
                    for count_repeat in count_repeats:  # 0,2 -> data and distance to epipolar line(label), 1 -> virt_pts(essential loss), R,t(essential loss 2)
                    
                        if( count_repeat==0 or count_repeat == 2 ):
                        
                            for batch in range( self.batches_in_N ):
                                start_idx = ( self.n_samples * self.N * 0) + ( chosen_sample * self.N ) + batch * self.batch_size
                                indices = list( range( start_idx, start_idx + self.batch_size ) )
                                yield indices
                                
                        elif( count_repeat==1 ):
                        
                            start_idx = ( self.n_samples * self.N * count_repeat) + ( chosen_sample * self.N )
                            indices = list( range( start_idx, start_idx + self.N) )
                            yield indices

            elif( self.N_images_in_batch >= 1 and self.N == self.batch_size ):
                
                for chosen_sample_idx, chosen_sample in enumerate(shuffled_samples):
                    
                    if( chosen_sample_idx % self.N_images_in_batch == 0):
                        indices = []
                    
                    indices.append(chosen_sample)
                    
                    if( chosen_sample_idx % self.N_images_in_batch == self.N_images_in_batch-1 ):
                        yield indices
                            
        elif(self.input_type == 'n_to_n'):
            
            for chosen_sample_idx, chosen_sample in enumerate(shuffled_samples):
                
                if( chosen_sample_idx % self.N_images_in_batch == 0):
                    indices = []
                
                indices.append(chosen_sample)
                
                if( chosen_sample_idx % self.N_images_in_batch == self.N_images_in_batch-1 ):
                    yield indices

    def __len__(self):
        
        if(self.input_type == '1_to_1'):
            if( self.N_images_in_batch == 1 and self.N > self.batch_size ):
                length = self.n_samples * self.batches_in_N
                if( self.num_repeats == 2 ):
                    length = 1 * length + self.n_samples
                elif( self.num_repeats == 3 ):
                    length = 2 * length + self.n_samples
                return length
                
            elif( self.N_images_in_batch >= 1 and self.N == self.batch_size ):                
                return self.n_samples // self.N_images_in_batch
            
        elif(self.input_type == 'n_to_n'):
            return self.n_samples // self.N_images_in_batch
            
# Make sure that any custom collate_fn, worker_init_fn or dataset code is declared as top level
# definitions, outside of the __main__ check. This ensures that they are available in worker processes.
# (this is needed since functions are pickled as references only, not bytecode.)

def get_sampler( config, dataset, N_images_in_batch, N, batch_size, common_dataset_size = 0 ):
    
    train_val_test = dataset.train_val_test
    
    if( config.device != 'tpu' or config.tpu_cores != 'multi'):
        n_image_pairs = np.int64( len( dataset ) )
    else:
        n_image_pairs = np.int64( common_dataset_size )
        print( f'Length of {train_val_test} dataset for TPU multi core sampler is reduced to {n_image_pairs} ')
    
    return CustomBatchSampler( config, n_image_pairs, train_val_test, N_images_in_batch, N, batch_size )

if __name__ == '__main__':
    # customBatchSampler = CustomBatchSampler( 1000, 'train', 2, 2048, 1024 )
    customBatchSampler = CustomBatchSampler( 1000, 'train', 1, 512, 512 )
    
    # customBatchSampler = CustomBatchSampler( 1000, 'train', 64, 512, 64 )
    # customBatchSampler = CustomBatchSampler( 1000, 'train', 32, 1024, 32 )
