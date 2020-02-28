from django.shortcuts import render
from . import dataset

def select_data(request, input_dict, output_dict, widget):
    bunch = input_dict['data']

    attrs = {}
    if 'feature_names' in bunch:
        for i,f in enumerate(bunch.feature_names):
            if 'feature_value_names' in bunch and len(bunch.feature_value_names[i])>0:
                vals = [str(v) for v in bunch.feature_value_names[i]]
                attrs[f] = {'values': vals, 'type': 'Discrete', 'feature': 1}
            else:
                attrs[f] = {'values': [], 'type': 'Continuous', 'feature': 1}

    # Target:
    if dataset.is_target_nominal(bunch):
        #nominal target
        attrs['class'] = {'values': bunch.target_names, 'type': 'Discrete', 'feature': 0}
    else:
        attrs['class'] = {'values': [], 'type': 'Continuous', 'feature': 0}

    attrs_as_list = list(attrs.items()) # do not sort the features

    return render(request, 'interactions/select_data_scikit.html',
                  {'widget' : widget, 'attrs' : attrs_as_list})
