import argparse
import os
from testset_evaluation import evaluate_testset


def call_evaluate_testset(id_in_model_name, model_name, use_val=False):
    save_path_mn = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name,
                                'hcomb_' + str(id_in_model_name))
    save_path_mn_metrics = os.path.join('/home/spiess/twoears_proj/models/heiner/model_directories', model_name,
                                        'hcomb_' + str(id_in_model_name), 'final')

    from heiner import hyperparameters
    params = hyperparameters.H()
    params.load_from_dir(save_path_mn)

    from heiner import utils
    metrics = utils.load_metrics(save_path_mn_metrics)

    # 'test_sens_spec_class_scene': np.array(test_phase.sens_spec_class_scene)

    if use_val:
        sens_per_scene_class = metrics['val_sens_spec_class_scene'][-1, :, :, 0]
        spec_per_scene_class = metrics['val_sens_spec_class_scene'][-1, :, :, 1]
    else:
        sens_per_scene_class = metrics['test_sens_spec_class_scene'][-1, :, :, 0]
        spec_per_scene_class = metrics['test_sens_spec_class_scene'][-1, :, :, 1]

    name = 'LDNN_id_{}_in_{})'.format(id_in_model_name, model_name)

    plotconfig = {'class_std': False, 'show_explanation': True}

    folder = os.path.join(save_path_mn, 'analysis', name)
    os.makedirs(folder, exist_ok=True)

    evaluate_testset(folder, name, plotconfig, sens_per_scene_class, spec_per_scene_class, collect=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--id_in_mn',
                        required=True,
                        type=int,
                        default=0,
                        dest="id_in_model_name",
                        metavar="<id in model directory>",
                        help="-")
    parser.add_argument('-mn', '--model_name',
                        required=False,
                        type=str,
                        default='LDNN_final',
                        dest='model_name',
                        metavar='<model name>',
                        help='The model name for final model.')

    args = parser.parse_args()

    call_evaluate_testset(**vars(args))


if __name__ == '__main__':
    # main()
    call_evaluate_testset(3, 'LDNN_final_nnw', use_val=True)
