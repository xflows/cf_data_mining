from django.shortcuts import render

def select_data(request, input_dict, output_dict, widget):
    bunch = input_dict['data']

    attrs = {}
    if bunch.has_key('feature_names'):
        for i,f in enumerate(bunch.feature_names):
            if bunch.has_key('feature_value_names') and len(bunch.feature_value_names[i])>0:
                vals = [str(v) for v in bunch.feature_value_names[i]]
                attrs[f] = {'values': vals, 'type': 'Discrete', 'feature': 1}
            else:
                attrs[f] = {'values': [], 'type': 'Continuous', 'feature': 1}

    attrs_as_list = attrs.items() # do not sort the features

    return render(request, 'interactions/select_data.html', 
                  {'widget' : widget, 'attrs' : attrs_as_list})

