import numpy as np
import matplotlib.pyplot as plt
import os

import time

def plot_success_and_loss( config, epoch, chunk, success_checkpoint, loss_checkpoint):
    
    if(epoch>0 or chunk>0):
    
        folder_name = config.output_path_local
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)
        
        colors = ['blue', 'red', 'green', 'black']
        
        if(success_checkpoint.shape[0]==2):
            if(config.validation == 'enable'):
                sets = ['train', 'val']
                linestyles = ['-', ':']
            else:
                sets = ['train']
                linestyles = ['-']
        else:
            sets = ['test']
            linestyles = ['--']
        x_count = success_checkpoint.shape[1] * success_checkpoint.shape[2]
            
        success = np.zeros( (len(sets), success_checkpoint.shape[3], x_count) )
        loss = np.zeros( (len(sets), loss_checkpoint.shape[3], x_count) )
        
        for set_ind, set_ in enumerate(sets):
            
            for s in range(success.shape[1]):            
                success[set_ind,s,:] = np.reshape(success_checkpoint[set_ind,:,:,s], (1,1,-1))
                
            for l in range(loss.shape[1]):            
                loss[set_ind,l,:] = np.reshape(loss_checkpoint[set_ind,:,:,l], (1,1,-1))
        
        valid_data_count = epoch * config.n_chunks + chunk + 1
        if(success_checkpoint.shape[0]==2):
            success = success[:,:,:valid_data_count]
            loss = loss[:,:,:valid_data_count]
        
        x_line = np.arange(0, valid_data_count, 1)
        
        # success plot
        success_legend_names = ['acc', 'pre', 'rec', 'f1']
        
        plt.figure()
        plt.xlabel('Chunk')
        plt.title( 'Success' )
        
        for set_ind, set_ in enumerate(sets):            
            for s in range(success.shape[1]):
                
                y_line = success[set_ind,s,:]
                linestyle = linestyles[set_ind]
                color=colors[s]
                success_legend_name = set_ + '_' + success_legend_names[s]      
                
                plt.plot(x_line, y_line, color=color, linestyle=linestyle, label=success_legend_name)
        plt.grid(True)
        if(success_checkpoint.shape[0]==2):
            plt.legend(fontsize = 'x-small', loc="lower right", ncol=4,)
        else:
            plt.legend(fontsize = 'x-small', loc="lower right", ncol=4,)
        if(success_checkpoint.shape[0]==2):
            plot_file_name = 'success' + '.png'
        else:
            plot_file_name = 'success_test' + '.png'
        plt.savefig( os.path.join(folder_name, plot_file_name), dpi=300)

        # loss plot
        
        loss_legend_names = ['cls', 'geo', 'ess']    
        
        lines = []
        
        for l in range(loss.shape[1]):
            for set_ind, set_ in enumerate(sets):
                
                # if(set_=='train' or l==0):
                    
                if(set_ind==0):
                    if(l==0):
                        fig, ax0 = plt.subplots()
                    elif(l==1):
                        ax1 = ax0.twinx()
                    elif(l==2):
                        ax2 = ax0.twinx()
                        ax2.spines['right'].set_position( ('axes', 1.30) )
            
                y_line = loss[set_ind,l,:]
                color = colors[l]
                linestyle = linestyles[set_ind]
                loss_legend_name = set_ + '_' + loss_legend_names[l]
            
                if(l==0):
                    line = ax0.plot(x_line, y_line, color=color, linestyle=linestyle, label=loss_legend_name)
                elif(l==1):                     
                    line = ax1.plot(x_line, y_line, color=color, linestyle=linestyle, label=loss_legend_name)
                elif(l==2):
                    line = ax2.plot(x_line, y_line, color=color, linestyle=linestyle, label=loss_legend_name)
                lines.append(line)
                
        ax0.grid(True)
        lines0, labels0 = ax0.get_legend_handles_labels()
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if(success_checkpoint.shape[0]==2):
            if(success_checkpoint.shape[0]==2):
                plt.legend(lines0+lines1+lines2, labels0+labels1+labels2, fontsize = 'x-small', loc="upper right", ncol=2,)
            else:
                plt.legend(lines0+lines1+lines2, labels0+labels1+labels2, fontsize = 'x-small', loc="upper right", ncol=1,)
        else:
            plt.legend(lines0+lines1+lines2, labels0+labels1+labels2, fontsize = 'x-small', loc="upper right", ncol=3,)
        
        ax0.set_ylabel('Cls Loss', color=colors[0])
        ax1.set_ylabel('Geo Loss', color=colors[1])
        ax2.set_ylabel('Ess Loss', color=colors[2])
        
        ax0.tick_params(axis='y', colors=colors[0])
        ax1.tick_params(axis='y', colors=colors[1])
        ax2.tick_params(axis='y', colors=colors[2])
        
        ax0.spines['left'].set_color(colors[0])
        ax1.spines['right'].set_color(colors[1])
        ax2.spines['right'].set_color(colors[2])
        
        fig.tight_layout()
        if(loss_checkpoint.shape[0]==2):
            plot_file_name = 'loss' + '.png'
        else:
            plot_file_name = 'loss_test' + '.png'
        plt.savefig( os.path.join(folder_name, plot_file_name), dpi=300)
        
def plot_mAP( config, epoch, chunk, mAP_checkpoint, ref_angles):
    
    if(epoch>0 or chunk>0):
        
        folder_name = config.output_path_local
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)
        
        colors = ['blue', 'red', 'green']
        
        if(mAP_checkpoint.shape[0]==2):
            sets = ['train', 'val']
            linestyles = ['-', ':']
        else:
            sets = ['test']
            linestyles = ['--']
        x_count = mAP_checkpoint.shape[1] * mAP_checkpoint.shape[2]
        
        error_angle_legend_names = ['R', 't', 'Rt']        
        
        for ref_angle_ind, ref_angle in enumerate(ref_angles):
            mAP = np.zeros( (len(sets), mAP_checkpoint.shape[3], x_count) )
            
            for set_ind, set_ in enumerate(sets):
                for err_ang in range(mAP_checkpoint.shape[3]):
                    for e in range(mAP_checkpoint.shape[1]):
                        for c in range(mAP_checkpoint.shape[2]):
                            mAP[set_ind, err_ang, e*mAP_checkpoint.shape[2]+c] = np.sum(mAP_checkpoint[set_ind, e, c, err_ang, 0:ref_angle]) / np.sum(mAP_checkpoint[set_ind, e, c, err_ang, :])
            mAP = mAP[:,:,0:epoch*config.n_chunks+chunk+1]
            x_line = np.arange(0, epoch*config.n_chunks+chunk+1, 1)
            # mAP plot
            
            plt.figure()
            plt.xlabel('Chunk')
            plt.title( 'mAP_' + f'{ref_angle:02d}' + '_degree' )
            
            for set_ind, set_ in enumerate(sets):            
                for s in range(mAP.shape[1]):
                    
                    y_line = mAP[set_ind,s,:]
                    linestyle= linestyles[set_ind]
                    color=colors[s]
                    mAP_legend_name = set_ + '_' + error_angle_legend_names[s]      
                    
                    plt.plot(x_line, y_line, color=color, linestyle=linestyle, label=mAP_legend_name)
            plt.grid(True)
            plt.legend(fontsize = 'x-small', loc="lower right", ncol=3,)
            if(mAP.shape[0]==2):
                plot_file_name = 'mAP_' + f'{ref_angle:02d}' + '_degree.png'
            else:
                plot_file_name = 'mAP_test_' + f'{ref_angle:02d}' + '_degree.png'
            plt.savefig( os.path.join(folder_name, plot_file_name), dpi=300)
        
def plot_proc_time( config, epoch, chunk, proc_time_checkpoint):
    
    if(epoch>0 or chunk>0):
    
        folder_name = os.path.join(config.output_path_local, 'process_time')
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)
            
        if(proc_time_checkpoint.shape[0]==2):
            if(config.validation == 'enable'):
                sets = ['train', 'val']
                linestyles = ['-', '-']
            else:
                sets = ['train']
                linestyles = ['-']
        else:
            sets = ['test']
            linestyles = ['--']
            
        colors = ['blue', 'red']
            
        proc_time_checkpoint = proc_time_checkpoint[:, 0:epoch+1, 0:chunk+1]
        
        proc_time_plot = np.zeros( ( len(sets), proc_time_checkpoint.shape[1]*proc_time_checkpoint.shape[2]) )
        
        for i in range( len(sets) ):
            proc_time_plot[i,:] = proc_time_checkpoint[i,:,:].reshape( (1, -1) )
            
        x_line = np.arange(0, proc_time_plot.shape[1], 1)
        
        lines = []
        for i in range( len(sets) ):
            if(i==0):
                fig, ax0 = plt.subplots()
            elif(i==1):
                ax1 = ax0.twinx()
        
            y_line = proc_time_plot[i,:]
            color = colors[i]
            linestyle = linestyles[i]
            loss_legend_name = sets[i]
    
            if(i==0):
                line = ax0.plot(x_line, y_line, color=color, linestyle=linestyle, label=loss_legend_name)
            elif(i==1):                     
                line = ax1.plot(x_line, y_line, color=color, linestyle=linestyle, label=loss_legend_name)
            lines.append(line)
        
        ax0.grid(True)
        if(proc_time_checkpoint.shape[0]==1):
            lines0, labels0 = ax0.get_legend_handles_labels()
            plt.legend(lines0, labels0, fontsize = 'small', loc="upper right", ncol=2,)
            
            ax0.set_ylabel('Test Time(sec)', color=colors[0])
            
            ax0.tick_params(axis='y', colors=colors[0])
            
            ax0.spines['left'].set_color(colors[0])
            
            plot_file_name = 'process_time_test' + '.png'
        else:
            lines0, labels0 = ax0.get_legend_handles_labels()
            if(config.validation == 'enable'):
                lines1, labels1 = ax1.get_legend_handles_labels()
                plt.legend(lines0+lines1, labels0+labels1, fontsize = 'small', loc="upper right", ncol=2,)
            else:
                plt.legend(lines0, labels0, fontsize = 'small', loc="upper right", ncol=1,)
                
            ax0.set_ylabel('Train Time(sec)', color=colors[0])
            if(config.validation == 'enable'):
                ax1.set_ylabel('Val Time(sec)', color=colors[1])
            
            ax0.tick_params(axis='y', colors=colors[0])
            if(config.validation == 'enable'):
                ax1.tick_params(axis='y', colors=colors[1])
            
            ax0.spines['left'].set_color(colors[0])
            if(config.validation == 'enable'):
                ax1.spines['right'].set_color(colors[1])
            
            fig.tight_layout()
            plot_file_name = 'process_time' + '.png'
        plt.savefig( os.path.join(folder_name, plot_file_name), dpi=300)

if __name__ == '__main__':
    while True:
        time.sleep(1)