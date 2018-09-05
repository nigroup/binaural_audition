import heiner.keras_model as keras_model
import heiner.keras_model_final as keras_model_final


if __name__ == '__main__':
    # model_name = 'LDNN_pycharm'
    model_name = 'LDNN_v1'

    # keras_model.run_experiment(False, 1, 'BAC', '0', 5, False, 1000, model_name=model_name)

    keras_model_final.run_final_experiment(False, '2', 15, 1, model_name='LDNN_final', model_name_old='LDNN_v1')