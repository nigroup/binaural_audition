# this plotting code is shared by heiner/changbin/moritz
# => input: sensitivity and specificity per scene per class each as array with shape (nscenes, nclasses) = (168, 13)
# => output: each function creates a plot (that can be put into a figure/subplot as determined by calling code)


# TODO: put here a managable data struct that holds all scene specifications from binAudLSTM_testSceneParameters.txt

# TODO: add class names (alarm, ...)

def plot_metric_over_snr_per_nsrc(sens_per_scene_class, spec_per_scene_class, metric_name):
    if metric_name == 'BAC':
        metric_per_class_scene = (sens_per_scene_class + spec_per_scene_class)/2.
    elif metric_name == 'sens':
        metric_per_class_scene = sens_per_scene_class
    elif metric_name == 'spec':
        metric_per_class_scene = spec_per_scene_class
    else: 
        # remark: BAC2 is not required for test set evaluation
        raise ValueError('the metric {} is not supported (need one of BAC, sens, spec)'.format(metric_name))

    # class averages here only
    metric_per_scene = np.mean(metric_per_class_scene, axis=1)

    # TODO: plotting stuff here -- without plt.figure / without plt.savefig



def plot_metric_over_snr_per_class(sens_per_scene_class, spec_per_scene_class, metric_name):
    if metric_name == 'BAC':
        metric_per_class_scene = (sens_per_scene_class + spec_per_scene_class)/2.
    elif metric_name == 'sens':
        metric_per_class_scene = sens_per_scene_class
    elif metric_name == 'spec':
        metric_per_class_scene = spec_per_scene_class
    else: 
        # remark: BAC2 is not required for test set evaluation
        raise ValueError('the metric {} is not supported (need one of BAC, sens, spec)'.format(metric_name))

    # TODO: plotting stuff here -- without plt.figure / without plt.savefig



def plot_metric_over_nsrc_per_class(sens_per_scene_class, spec_per_scene_class, metric_name):
    if metric_name == 'BAC':
        metric_per_class_scene = (sens_per_scene_class + spec_per_scene_class)/2.
    elif metric_name == 'sens':
        metric_per_class_scene = sens_per_scene_class
    elif metric_name == 'spec':
        metric_per_class_scene = spec_per_scene_class
    else: 
        # remark: BAC2 is not required for test set evaluation
        raise ValueError('the metric {} is not supported (need one of BAC, sens, spec)'.format(metric_name))

    # TODO: plotting stuff here -- without plt.figure / without plt.savefig
