
import tensorflow as tf
from gn_inverse_dynamics.train_model.train_model import TrainModel
from gn_inverse_dynamics.train_model.run_functions import *
from gn_inverse_dynamics.utils import mygraphutils as mygraphutils



class TorqueErrorModel(TrainModel):
    def __init__(self, args):
        super().__init__(args, "TorqueErrorModel")

    def run(self):             
        # wrapper functions to be used
        def update_step(inputs_tr, targets_tr):
            with tf.GradientTape() as tape:
                output_ops_tr = self.model(inputs_tr, self.num_processing_steps)
                # Loss.
                loss_ops_tr = self.create_loss_ops(targets_tr, output_ops_tr)
                #loss_tr = tf.math.reduce_sum(loss_ops_tr) / self.num_processing_steps   
                loss_tr = loss_ops_tr[self.num_processing_steps-1]

            gradients = tape.gradient(loss_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return output_ops_tr, loss_tr 

        def val_loss(inputs_tr, targets_tr):
            output_ops_tr = self.model(inputs_tr, self.num_processing_steps)
            # Loss.
            loss_ops_tr = self.create_loss_ops(targets_tr, output_ops_tr)
            #loss_tr = tf.math.reduce_sum(loss_ops_tr) / self.num_processing_steps
            loss_tr = loss_ops_tr[self.num_processing_steps-1]
            return output_ops_tr, loss_tr

        ## step functions
        compiled_update_step = tf.function(update_step, 
                                input_signature=self.dataset_batch_signature)
        compiled_val_loss = tf.function(val_loss, 
                                input_signature=self.dataset_batch_signature) 

        ## save input data for future
        for train_traj_tr in self.dataset_batch :
            inputs_batch_tr = mygraphutils.graph_reshape(train_traj_tr[0])
            break
        for single_traj_tr in self.dataset:
            input_tr = single_traj_tr[0]
            break

        self.save_sample_input(inputs_batch_tr, "inputs_batch_tr")
        self.save_sample_input(input_tr, "input_tr")
                                                    

        ## training
        print("============ start !!! =============")
        self.TOTAL_TIMER.reset()
        batch_loss_sum = run_one_epoch(compiled_val_loss, self.dataset_batch)
        self.min_loss = batch_loss_sum #0.02  
        print("T {:.1f}, Ltr of initial raw model = {}".format( 
                                self.TOTAL_TIMER.elapsed(), batch_loss_sum ) )
        break_count = 0
        for epoch in range(0, self.epoch_size):
            self.update_dataset_batch() #shuffle
            batch_loss_sum = run_one_epoch(compiled_update_step, self.dataset_batch)
            self.logf.record_time_loss(self.TOTAL_TIMER.elapsed(), batch_loss_sum)

            if(self.validation_test):
                val_batch_loss_sum = run_one_epoch(compiled_val_loss, self.valdataset_batch) 
                self.logf_val.record_time_loss(self.TOTAL_TIMER.elapsed(), val_batch_loss_sum)
                print("T {:.1f}, epoch_iter = {:02d}, Ltr {:.4f}, ValLtr {:.4f}".format(
                        self.TOTAL_TIMER.elapsed(), epoch, batch_loss_sum, val_batch_loss_sum))

                if(val_batch_loss_sum*0.7 > batch_loss_sum):
                    break_count = break_count + 1
                    print(" cnt = {:02d}, val_batch_loss_sum*0.7 > batch_loss_sum ", break_count)
                else:
                    break_count = 0
                
                if(break_count > 3):
                    print(" cnt = {:02d}", break_count)
                    break
                    
            else:
                print("T {:.1f}, epoch_iter = {:02d}, Ltr {:.4f}".format(
                    self.TOTAL_TIMER.elapsed(), epoch, batch_loss_sum))



            self.save_model(batch_loss_sum)