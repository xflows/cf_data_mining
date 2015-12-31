from django.shortcuts import render
from cf_core import helpers
import numpy as np
import datetime
import dataset

def display_classifier(request,input_dict,output_dict,widget):
    """Displays a classifier/model
    """

    classifier = input_dict['classifier']
    output_dict = {}
    output_dict['model_as_string'] = classifier.print_classifier()

    return render(request, 'visualizations/display_classifier.html',{'widget':widget,'input_dict':input_dict,'output_dict':output_dict})


def export_dataset_to_csv(request,input_dict,output_dict,widget):
    """Visualization for exporting dataset to CSV file and download of the file

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """

    output_dict={}
    dataset= input_dict['dataset']

    csv=[]
    for i,sample in enumerate(dataset.data):
        csv.append(np.append(sample,dataset.target[i])) #join n_sample and n_feature array

    destination = helpers.get_media_root()+'/'+str(request.user.id)+'/'+str(widget.id)+'.csv'
    np.savetxt(destination, csv, delimiter=",")

    filename = str(request.user.id)+'/'+str(widget.id)+'.csv'
    output_dict['filename'] = filename
    return render(request, 'visualizations/string_to_file.html',{'widget':widget,'input_dict':input_dict,'output_dict':output_dict})


def display_dataset(request,input_dict,output_dict,widget):
    """Visualization displaying a dataset using a table viewer

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """
    display_data = helper_display_dataset(input_dict['data'])
    return render(request, 'visualizations/display_dataset.html',{'widget':widget,'input_dict':input_dict,'output_dict':display_data})


def helper_display_dataset(bunch):
    """Helper for display_dataset visualization

    :param bunch:
    :return: a dict
    """

    data = bunch["data"]
    target = bunch["target"]

    csv=[]

    nrows, ncols = data.shape
    for i in range(0,nrows):
        csv.append([])

    # Features:
    for j in range(0,ncols):
        if dataset.is_feature_nominal(bunch, j):
            #nominal feat
            for i in range(0,nrows):
                if np.isnan(data[i][j]):
                    csv[i].append( np.nan )
                else:
                    val = int(data[i][j])
                    csv[i].append( bunch.feature_value_names[j][val]+(" [%d]" % val) )
        else:
            for i in range(0,nrows):
                csv[i].append( data[i][j] )

    # Target:
    if dataset.is_target_nominal(bunch):
        #nominal target
        for i in range(0,nrows):
            if np.isnan(target[i]):
                csv[i].append( np.nan )
            else:
                val = int(target[i])
                csv[i].append( bunch.target_names[val]+(" [%d]" % val) )
    else:
        for i in range(0,nrows):
            csv[i].append( target[i] )

    # attrs = ["attribute" for i in range(len(data[0]))] #name of attributes
    attrs = bunch.feature_names
    # attrs.append('class')
    metas = ''
    data_new = csv #fill table with data

    return {'attrs':attrs, 'metas':metas, 'data_new':data_new, 'class_var':'class'}


def display_clustering_table_form(request,input_dict,output_dict,widget):
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
    display_data['class_var']='cluster_id'

    for i, row in enumerate( display_data['data_new'] ):
        row.append( bunch['cluster_id'][i] )

    return render(request, 'visualizations/display_dataset.html',{'widget':widget,'input_dict':input_dict,'output_dict':display_data})


def display_decision_tree(request,input_dict,output_dict,widget):
    """Visualization displaying a decision tree

    :param request:
    :param input_dict:
    :param output_dict:
    :param widget:
    :return:
    """

    png_file = 'decision_tree.png'


    my_dict = {'pngfile':png_file, 'param':str( datetime.datetime.now().time() )} # param: used to force reload of image

    return render(request, 'visualizations/display_decision_tree.html',{'widget':widget,'input_dict':input_dict,'output_dict':my_dict })
