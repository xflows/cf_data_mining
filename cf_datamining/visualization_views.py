from django.shortcuts import render

def export_dataset_to_CSV(request,input_dict,output_dict,widget):
    from cf_base import helpers
    import numpy

    output_dict={}
    dataset= input_dict['scikitDataset']

    csv=[]
    count=0
    for i,sample in enumerate(dataset.data):
        csv.append(numpy.append(sample,dataset.target[i])) #join n_sample and n_feature array

    destination = helpers.get_media_root()+'/'+str(request.user.id)+'/'+str(widget.id)+'.csv'
    numpy.savetxt(destination, csv, delimiter=",")

    filename = str(request.user.id)+'/'+str(widget.id)+'.csv'
    output_dict['filename'] = filename
    return render(request, 'visualizations/string_to_file.html',{'widget':widget,'input_dict':input_dict,'output_dict':output_dict})


def display_dataset(request,input_dict,output_dict,widget):
    display_data = helper_display_dataset(input_dict)
    return render(request, 'visualizations/scikitAlgorithms_displayDS.html',{'widget':widget,'input_dict':input_dict,'output_dict':display_data})


def helper_display_dataset(data):
    """ helper_display_dataset """

    #get data to fill table
    bunch = data['data']
    print type(bunch)

    data = bunch["data"]
    target = bunch["target"]

    # join data in the right format
    import numpy as np
    csv=[]
    # row=0
    # for sample in data:
    #     csv.append(numpy.append(sample,target[row])) #join n_sample and n_feature array
    #     row+=1

    nrows, ncols = data.shape
    for i in range(0,nrows):
        csv.append([])

    # Features:
    for j in range(0,ncols):
        if bunch.has_key("feature_value_names") and len(bunch.feature_value_names[j])>0:
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
    if len(bunch.target_names)>0:
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
    display_data = helper_display_dataset(input_dict)
    bunch = input_dict["data"]
    display_data['attrs'].append('class')
    display_data['class_var']='cluster_id'

    for i, row in enumerate( display_data['data_new'] ):
        row.append( bunch['cluster_id'][i] )


    return render(request, 'visualizations/scikitAlgorithms_displayDS.html',{'widget':widget,'input_dict':input_dict,'output_dict':display_data})


def display_decision_tree(request,input_dict,output_dict,widget):
    pngFile = 'decisionTree-scikit.png'

    import datetime
    my_dict = {'pngfile':pngFile, 'param':str( datetime.datetime.now().time() )} # param: used to force reload of image

    return render(request, 'visualizations/scikitAlgorithms_displayDecisTree.html',{'widget':widget,'input_dict':input_dict,'output_dict':my_dict })

