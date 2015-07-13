def scikitAlgorithms_displayDS(request,input_dict,output_dict,widget):
    data = input_dict['data']
    return render(request, 'visualizations/scikitAlgorithms_displayDS.html',{'widget':widget,'input_dict':input_dict,'output_dict':helperDisplayDS(output_dict)})
