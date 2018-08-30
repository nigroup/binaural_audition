import heiner.keras_model as keras_model


if __name__ == '__main__':
    # model_name = 'LDNN_pycharm'
    model_name = 'LDNN_v1'

    keras_model.run_experiment(False, 1, 'BAC', '0', 5, False, 1000, model_name=model_name)
