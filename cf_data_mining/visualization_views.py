from django.shortcuts import render
from cf_core import helpers
import numpy as np
import datetime
import sklearn

from . import dataset


def display_classifier(request, input_dict, output_dict, widget):
    """Displays a classifier/model
    """

    classifier = input_dict['classifier']
    output_dict = {}

    output_dict['model_as_string'] = classifier.print_classifier()

    return render(request, 'visualizations/display_classifier.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})


def export_dataset_to_csv(request, input_dict, output_dict, widget):
    """Visualization for exporting dataset to CSV file and download of the file

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """

    output_dict = {}
    dataset = input_dict['dataset']

    csv = []
    for i, sample in enumerate(dataset.data):
        # join n_sample and n_feature array
        csv.append(np.append(sample, dataset.target[i]))

    destination = helpers.get_media_root() + '/' + str(request.user.id) + \
        '/' + str(widget.id) + '.csv'
    np.savetxt(destination, csv, delimiter=",")

    filename = str(request.user.id) + '/' + str(widget.id) + '.csv'
    output_dict['filename'] = filename
    return render(request, 'visualizations/string_to_file.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})


def display_dataset(request, input_dict, output_dict, widget):
    """Visualization displaying a dataset using a table viewer

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """
    display_data = helper_display_dataset(input_dict['data'])
    return render(request, 'visualizations/display_dataset.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': display_data})


def helper_display_dataset(bunch):
    """Helper for display_dataset visualization

    :param bunch:
    :return: a dict
    """

    data = bunch["data"]
    target = bunch.get("target")
    targetPredicted = bunch.get("targetPredicted")

    csv = []

    nrows, ncols = data.shape
    for i in range(0, nrows):
        csv.append([])

    # Features:
    for j in range(0, ncols):
        if dataset.is_feature_nominal(bunch, j):
            # nominal feat
            for i in range(0, nrows):
                if np.isnan(data[i][j]):
                    csv[i].append(np.nan)
                else:
                    val = int(data[i][j])
                    csv[i].append(bunch.feature_value_names[
                                  j][val] + (" [%d]" % val))
        else:
            for i in range(0, nrows):
                csv[i].append(data[i][j])

    if target is not None:
        if dataset.is_target_nominal(bunch):
            # nominal target
            for i in range(0, nrows):
                if np.isnan(target[i]).all():
                    csv[i].append(np.nan)
                else:
                    if isinstance(target[i], np.ndarray):
                        csv[i].append(['{}={}'.format(n, v) for n, v in list(zip(bunch.target_names, target[i]))])
                    else:
                        val = int(target[i])
                        csv[i].append(bunch.target_names[val] + (" [%d]" % val))
        else:
            for i in range(0, nrows):
                csv[i].append(target[i])

    if targetPredicted is not None:
        for i in range(0, nrows):
            csv[i].append(targetPredicted[i])

    if "feature_names" in bunch:
        attrs = bunch.feature_names
    else:
        # name of attributes
        attrs = ["attribute" + str(i) for i in range(len(data[0]))]

    # attrs.append('class')
    metas = ''
    data_new = csv  # fill table with data
    return {'attrs': attrs, 'metas': metas, 'data_new': data_new, 'target_var': target is not None, 'target_var_predicted': targetPredicted is not None}


def display_clustering_table_form(request, input_dict, output_dict, widget):
    """Visualization displaying a dataset table along with cluster associations for each instance

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """
    display_data = helper_display_dataset(input_dict)
    bunch = input_dict["data"]
    display_data['attrs'].append('class')
    display_data['class_var'] = 'cluster_id'

    for i, row in enumerate(display_data['data_new']):
        row.append(bunch['cluster_id'][i])

    return render(request, 'visualizations/display_dataset.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': display_data})


# def display_decision_tree(request, input_dict, output_dict, widget):
#     """Visualization displaying a decision tree
#
#     :param request:
#     :param input_dict:
#     :param output_dict:
#     :param widget:
#     :return:
#     """
#
#     # png_file = 'decision_tree.png'
#     #
#     # # param: used to force reload of image
#     # my_dict = {'pngfile': png_file, 'param': str(
#     #     datetime.datetime.now().time())}
#
#     return render(request, 'visualizations/display_decision_tree.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})



def display_decision_tree(request, input_dict, output_dict, widget):
    """Visualization displaying a decision tree

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """

    from sklearn import tree
    import tempfile
    import subprocess
    import os

    from mothra.settings import PUBLIC_DIR
    from workflows.helpers import ensure_dir

    from workflows.engine import ValueNotSet

    dataset = input_dict.get('dataset')
    if dataset is not None and dataset != ValueNotSet:
        feature_names = dataset.get('feature_names')
        class_names = dataset.get('target_names')
    else:
        feature_names = class_names = None

    text = tree.export_text(input_dict['classifier'].classifier,
                            feature_names=list(feature_names),
                            show_weights=True)

    # with tempfile.NamedTemporaryFile(mode='w', suffix='.dot') as fp:
    #     tree.export_graphviz(input_dict['classifier'].classifier,
    #                          out_file=fp.name)
    #                          # feature_names=feature_names,
    #                          # class_names=class_names)
    #     fp.flush()
    #     path, fname = os.path.split(fp.name)
    #     base, ext = os.path.splitext(fname)
    #     pngfile = os.path.join(MEDIA_ROOT, 'sklearn', base + '.png')
    #     print(pngfile)
    #     ensure_dir(pngfile)
    #     subprocess.Popen(['dot', '-Tpng', '-o', pngfile, fp.name])
    # print(pngfile)
    output_dict['treetext'] = text
    return render(request, 'visualizations/display_decision_tree.html', {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})
