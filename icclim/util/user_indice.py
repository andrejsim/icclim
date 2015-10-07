import util.calc as calc

# map with required parameters (for user defined indices)  
map_calc_params_required = {
                              'max': [],
                              'min': [],
                              'sum': [],
                              'mean': [],
                              'nb_events': ['logical_operation', 'thresh'], 
                              'max_number_consecutive_events': ['logical_operation', 'thresh'],
                              'run_mean': ['extreme_mode', 'window_width'],
                              'run_sum': ['extreme_mode', 'window_width'],
                              'anomalie': ['ref_time_range']
                              }   

# map with optional parameters (for user defined indices)  
map_calc_params_optional = {
                              'max': ['coef', 'logical_operation', 'thresh', 'date_event'], # filter: ['>=', 1.0]
                              'min': ['coef', 'logical_operation', 'thresh', 'date_event'],
                              'sum': ['coef', 'logical_operation', 'thresh'], #'date_event'],
                              'mean': ['coef', 'logical_operation', 'thresh'], #'date_event'],
                              'nb_events': ['coef', 'date_event'],
                              #'nb_events': ['coef', 'cond2', 'logical_operator_conditions', 'date_event'], 
                              # if 'cond2' --> 'logical_operator_conditions' is required, 
                              # if 2 vars ---> 'cond2' is applied to the second var
                              # threshold in cond1 or cond2 could be a single number, list of numbers, or percentile thresh
                              # threshold must be the same units !!!
                              # 'max_number_consecutive_events': ['coef', 'cond2', 'logical_operator_conditions', 'date_event'],
                              'max_number_consecutive_events': ['coef', 'date_event'],
                              'run_mean': ['coef', 'date_event'],
                              'run_sum': ['coef', 'date_event'],
                              'anomalie': []
                              }   

def check_params(user_indice):
    '''
    Check if a set of user parameters is correct for selected calc_operation
    '''
    
    if 'indice_name' not in user_indice.keys():
        raise IOError(" 'indice_name' is required for a user defines indice")
    
    elif 'calc_operation' not in user_indice.keys():
        raise IOError(" 'calc_operation' is required for a user defines indice")

    given_params = get_given_params(user_indice)
    
    calc_op = user_indice['calc_operation']
    required_params = map_calc_params_required[calc_op]
    
    if (calc_op not in ['max', 'min', 'mean', 'sum']) and (set(required_params).intersection(given_params) != set(required_params)):
        raise IOError('All theses parameters are required: {0}'.format(required_params))
    
    if calc_op=='nb_events' and type(user_indice['thresh'])==str:
        if ('out_units' or 'var_type') not in user_indice.keys():
            raise IOError("If threshold value is a percentile,  'out_units' and 'var_type' are required")
        


def get_given_params(user_indice):
    given_params_list = user_indice.keys()
    
    given_params_list.remove('indice_name')
    #given_params_list.remove('calc_operation') 

    return given_params_list


def set_params(user_indice):
    given_params = get_given_params(user_indice)
    
    class F:
        pass
    global obj
    obj = F()
    
    # we set all default parameters 
    setattr(obj, 'logical_operation', None)
    setattr(obj, 'thresh', None)
    setattr(obj, 'coef', 1.0)
    setattr(obj, 'fill_val', None)
    setattr(obj, 'date_event', False)
    setattr(obj, 'out_units', None)
    setattr(obj, 'var_type', None)
    
    
    for p in given_params:
        setattr(obj, p, user_indice[p])
        
    setattr(obj, p, user_indice[p])
    


def get_user_indice(user_indice, arr, fill_val, dt_arr=None, pctl_thresh=None):
    ### 'dt_arr' and 'pctl_thresh' are required for percentile-based indices, i.e. when a threshold is a percentile values
    ### 'pctl_thresh' could be a dictionary with daily percentiles (for temperature variables) 
    ### or an 2D array with percentiles (for precipitation variables)
    
    #check_params(user_indice) 
    set_params(user_indice)  
    
    #print obj.logical_operation, obj.thresh, fill_val, obj.date_event, obj.coef
    
    
    if obj.calc_operation in ['min', 'max', 'mean', 'sum']:
        # simple_stat(arr, stat_operation, logical_operation=None, thresh=None, coef=1.0, fill_val=None, index_event=False)
        res = calc.simple_stat(arr, 
                        stat_operation=obj.calc_operation,
                        logical_operation=obj.logical_operation,
                        thresh=obj.thresh,
                        coef=obj.coef,
                        fill_val=fill_val,
                        index_event=obj.date_event)

    elif obj.calc_operation == 'nb_events':
        
        if type(obj.thresh) != str: # thresh is float or int 
            # get_nb_days(arr, logical_operation, thresh, coef=1.0, fill_val=None, index_event=False)
            res = calc.get_nb_days(arr,
                              logical_operation=obj.logical_operation,
                              thresh=obj.thresh,
                              coef=obj.coef,
                              fill_val=fill_val,
                              index_event=obj.date_event) 
        else: # thresh is str: 'p90' or 'p20'
            
            if obj.var_type == 't':
                # TXXXp(arr, dt_arr, percentile_dict, logical_operation, fill_val=None, out_unit="days")
                
                #### ADD: coef=obj.coef, index_event=obj.date_event
                res = calc.TXXXp(arr, 
                                 dt_arr, 
                                 logical_operation=obj.logical_operation,
                                 percentile_dict=pctl_thresh,  
                                 fill_val=fill_val, 
                                 out_unit=obj.out_units,
                                 index_event=obj.date_event)
                
            elif obj.var_type == 'p':
                # RXXp(arr, percentile_arr, logical_operation='gt', pr_thresh = 1.0, fill_val=None, out_unit="days")
                
                #### ADD: coef=obj.coef, index_event=obj.date_event
                
                res = calc.RXXp(arr, 
                           logical_operation=obj.logical_operation,
                           percentile_arr=pctl_thresh, 
                            
                           pr_thresh = 1.0, # only for wet days, i.e. values >= 1.0
                           
                           fill_val=fill_val, 
                           out_unit=obj.out_units,
                           index_event=obj.date_event)
                
            
                        
    elif obj.calc_operation == 'max_number_consecutive_events':
        # get_max_nb_consecutive_days(arr, logical_operation, thresh, coef=1.0, fill_val=None, index_event=False)
        res = calc.get_max_nb_consecutive_days(arr, 
                                          logical_operation=obj.logical_operation, 
                                          thresh=obj.thresh, 
                                          coef=obj.coef, 
                                          fill_val=fill_val, 
                                          index_event=obj.date_event)       
        
        
    elif obj.calc_operation in ['run_mean', 'run_sum']:
        if obj.calc_operation == 'run_mean':
            stat_m = 'mean'
        elif obj.calc_operation == 'run_sum':
            stat_m = 'sum'
        
        # get_run_stat(arr, window_width, stat_mode, extreme_mode, coef=1.0, fill_val=None, index_event=False)
        res = calc.get_run_stat(arr, 
                           window_width=obj.window_width, 
                           stat_mode=stat_m, 
                           extreme_mode=obj.extreme_mode, 
                           coef=obj.coef, 
                           fill_val=fill_val, 
                           index_event=obj.date_event)
        
#     elif obj.calc_operation == 'anomalie':
#         res = calc.get_anomalie(arr, arr2)

    
    return res


# user_indice = {'indice_name': 'TEST',
#                'calc_operation': 'nb_events',
#                 'logical_operation': 'e',
#                 'thresh': 0,
#                #'stat_oper': 'max',
#                #'window': 6,
#                #'coef': 1,
#                #'date_event': True
#                } 
# 
# import numpy 
# arr = numpy.random.randint(-5, 10, size=(5,2,3))*1.0
# print arr
# print "==="
# 
# print get_user_indice(user_indice)