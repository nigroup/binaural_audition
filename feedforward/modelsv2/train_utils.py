import pdb

import numpy as np

import dataloader
import settings
import accuracy_utils as acc_u



def create_generator(dloader):
    while True:
        b_x, b_y = dloader.next_batch()
        if b_x is None or b_y is None:
            return
        yield b_x, b_y


def create_dataloaders(LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, BATCHSIZE, TIMESTEPS, EPOCHS, NFEATURES, NCLASSES,
                       VAL_FOLDS, VAL_STATEFUL):
    train_loader = dataloader.DataLoader('train', LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, batchsize=BATCHSIZE,
                              timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES)
    train_loader_len = train_loader.len()
    print('Number of batches per epoch (training): ' + str(train_loader_len))

    val_loader = dataloader.DataLoader('val', LABEL_MODE, VAL_FOLDS, TRAIN_SCENES, epochs=EPOCHS, batchsize=BATCHSIZE,
                            timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES, val_stateful=VAL_STATEFUL)

    val_loader_len = val_loader.len()
    print('Number of batches per epoch (validation): ' + str(val_loader_len))

    return train_loader, val_loader


class Phase:

    def __init__(self, train_or_val, graphModel, session, dloader, OUTPUT_THRESHOLD, MASK_VAL, EPOCHS, val_fold_str, metric='BAC',
                 ret=('final', 'per_class')):

        self.prefix = train_or_val
        if train_or_val == 'train':
            self.train = True
        elif train_or_val == 'val':
            self.train = False
        else:
            raise ValueError('unknown train_or_val: {}'.format(train_or_val))

        self.graphModel = graphModel
        self.session = session
        self.dloader = dloader
        self.e = 0
        self.OUTPUT_THRESHOLD = OUTPUT_THRESHOLD
        self.MASK_VAL = MASK_VAL
        self.EPOCHS = EPOCHS

        self.val_fold_str = val_fold_str

        self.metric = metric
        self.ret = ret

        self.gen = create_generator(dloader)
        self.dloader_len = dloader.len()

        self.losses = []
        self.accs = []
        self.class_accs = []

    @property
    def epoch_str(self):
        return 'epoch: {} / {}'.format(self.e + 1, self.EPOCHS)





    def run(self):
        def take_block_based_label(y):
            return y[:,:,-1,:,:]

        def transform_before_accuracy(out, y_true):
            ''' transform into: (according to Heiner)
            y_true: bs* bl*classes*2(labels, ids)
            out: bs*bl*classes
            '''
            out = out[0]
            out = out.reshape(y_true.shape[0],y_true.shape[1],out.shape[1])

            out = np.swapaxes(out, 0, 1)
            y_true = np.swapaxes(y_true, 0, 1)
            return out,y_true

        #self.model.reset_states()

        # training phase
        scene_instance_id_metrics_dict = dict()
        #for iteration in range(1, self.dloader_len[self.e] + 1):
        for iteration in range(1, 2 + 1):

            it_str = '{}_iteration: {} / {}'.format(self.prefix, iteration, self.dloader_len[self.e])

            b_x, b_y = next(self.gen)
            b_y = take_block_based_label(b_y)
            b_y_no_sceneIds = b_y[:,:,:,0].reshape((-1, settings.n_labels))


            if self.train:
                #train
                _,loss =  self.session.run([self.graphModel.optimiser,self.graphModel.cross_entropy], feed_dict={self.graphModel.x_dataloader: b_x, self.graphModel.y_dataloader: b_y_no_sceneIds }) #self.graphModel.cross_entropy_class_weights : data.cross_entropy_class_weights})
                #predict on batch
                out =  self.session.run([self.graphModel.thresholded], feed_dict={self.graphModel.x_dataloader: b_x, self.graphModel.y_dataloader:b_y_no_sceneIds})
            else:
                #loss onn batch, no train
                loss =  self.session.run([self.graphModel.optimiser,self.graphModel.cross_entropy], feed_dict={self.graphModel.x_dataloader: b_x, self.graphModel.y_dataloader: b_y_no_sceneIds}) #self.graphModel.cross_entropy_class_weights : data.cross_entropy_class_weights})
                #predict on batch
                out =  self.session.run([self.graphModel.thresholded], feed_dict={self.graphModel.x_dataloader: b_x, self.graphModel.y_dataloader:b_y_no_sceneIds })

            self.losses.append(loss)

            out, y_true = transform_before_accuracy(out,b_y)
            print("out calculated")

            acc_u.calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict, out, y_true ,self.OUTPUT_THRESHOLD, self.MASK_VAL)
            print("metrics calculated")

            loss_str = 'loss: {}'.format(loss)
            loss_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(self.val_fold_str, self.epoch_str, it_str, loss_str)
            print(loss_log_str)

            if not self.train and not self.dloader.val_stateful:
                pass
                #self.model.reset_states()

        if self.train:
            final_acc = 0

            final_acc = acc_u.train_accuracy(scene_instance_id_metrics_dict, metric=self.metric)
        else:
            final_acc, class_accuracies = acc_u.val_accuracy(scene_instance_id_metrics_dict, metric=self.metric,ret=self.ret)
            final_acc = 0
            class_accuracies = 0
            self.class_accs.append(class_accuracies)
        self.accs.append(final_acc)

        acc_str = '{}_accuracy: {}'.format(self.prefix, final_acc)
        acc_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(self.val_fold_str, self.epoch_str, '', acc_str)
        print(acc_log_str)

        # increase epoch
        self.e += 1

