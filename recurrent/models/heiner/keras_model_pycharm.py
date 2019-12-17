import keras_model
import keras_model_final


if __name__ == '__main__':
    model_name = 'LDNN_pycharm'
    # model_name = 'LDNN_v1'

    keras_model.run_experiment(False, 1, 'BAC', '3', 1, False, 2000, model_name=model_name)

    # keras_model_final.run_final_experiment(False, '2', 2, 3, model_name='LDNN_final_new_weighting', model_name_old='LDNN_v1')