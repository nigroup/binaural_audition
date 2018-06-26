from heiner import model_extension as m_ext
from heiner import accuracy_utils as acc_u
from heiner.dataloader import DataLoader


def create_generator(dloader):
    while True:
        # ret is either b_x, b_y or b_x, b_y, keep_states
        ret = dloader.next_batch()
        if ret[0] is None or ret[1] is None:
            return
        yield ret


def create_dataloaders(LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, BATCHSIZE, TIMESTEPS, EPOCHS, NFEATURES, NCLASSES,
                       VAL_FOLDS, VAL_STATEFUL, BUFFER):
    train_loader = DataLoader('train', LABEL_MODE, TRAIN_FOLDS, TRAIN_SCENES, batchsize=BATCHSIZE,
                              timesteps=TIMESTEPS, epochs=EPOCHS, features=NFEATURES, classes=NCLASSES,
                              buffer=BUFFER)
    train_loader_len = train_loader.len()
    print('Number of batches per epoch (training): ' + str(train_loader_len))

    val_loader = DataLoader('val', LABEL_MODE, VAL_FOLDS, TRAIN_SCENES, epochs=EPOCHS, batchsize=BATCHSIZE,
                            timesteps=TIMESTEPS, features=NFEATURES, classes=NCLASSES, val_stateful=VAL_STATEFUL,
                            buffer=BUFFER)

    val_loader_len = val_loader.len()
    print('Number of batches per epoch (validation): ' + str(val_loader_len))

    return train_loader, val_loader


class Phase:

    def __init__(self, train_or_val, model, dloader, OUTPUT_THRESHOLD, MASK_VAL, EPOCHS, val_fold_str, metric='BAC',
                 ret=('final', 'per_class')):

        self.prefix = train_or_val
        if train_or_val == 'train':
            self.train = True
        elif train_or_val == 'val':
            self.train = False
        else:
            raise ValueError('unknown train_or_val: {}'.format(train_or_val))

        self.model = model
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
        self.model.reset_states()

        # training phase
        scene_instance_id_metrics_dict = dict()

        for iteration in range(1, self.dloader_len[self.e] + 1):
            it_str = '{}_iteration: {} / {}'.format(self.prefix, iteration, self.dloader_len[self.e])

            b_x, b_y = next(self.gen)
            if self.train:
                loss, out = m_ext.train_and_predict_on_batch(self.model, b_x, b_y[:, :, :, 0])
            else:
                loss, out = m_ext.test_and_predict_on_batch(self.model, b_x, b_y[:, :, :, 0])
            self.losses.append(loss)

            acc_u.calculate_class_accuracies_metrics_per_scene_instance_in_batch(scene_instance_id_metrics_dict,
                                                                                 out, b_y,
                                                                                 self.OUTPUT_THRESHOLD, self.MASK_VAL)

            loss_str = 'loss: {}'.format(loss)
            loss_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(self.val_fold_str, self.epoch_str, it_str, loss_str)
            print(loss_log_str)

            if not self.train and not self.dloader.val_stateful:
                self.model.reset_states()

        if self.train:
            final_acc = acc_u.train_accuracy(scene_instance_id_metrics_dict, metric=self.metric)
        else:
            final_acc, class_accuracies = acc_u.val_accuracy(scene_instance_id_metrics_dict, metric=self.metric,
                                                             ret=self.ret)
            self.class_accs.append(class_accuracies)
        self.accs.append(final_acc)

        acc_str = '{}_accuracy: {}'.format(self.prefix, final_acc)
        acc_log_str = '{:<20}  {:<20}  {:<20}  {:<20}'.format(self.val_fold_str, self.epoch_str, '', acc_str)
        print(acc_log_str)

        # increase epoch
        self.e += 1
