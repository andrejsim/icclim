# basic function used for computing indices 

import numpy

import ctypes
from numpy.ctypeslib import ndpointer
import os
import util_dt

# TODO: remove reshape after calling C function

 
my_rep = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + os.sep
libraryC = ctypes.cdll.LoadLibrary(my_rep+'libC.so')


def get_first_occurrence(arr, logical_operation, thresh):
    '''
    Return the first occurrence (index) of values satisfying the condition (logical_operation, thresh)
    in the 3D array along axis=0    
    '''
    
    if logical_operation == 'gt':
        res=numpy.argmax(arr>thresh, axis=0)
        
    elif logical_operation == 'get':
        res=numpy.argmax(arr>=thresh, axis=0)   
            
    elif logical_operation == 'lt':
        res=numpy.argmax(arr<thresh, axis=0)    
            
    elif logical_operation == 'let':
        res=numpy.argmax(arr<=thresh, axis=0) 
             
    elif logical_operation == 'e':
        res=numpy.argmax(arr==thresh, axis=0)
        
    return res


def get_last_occurrence(arr, logical_operation, thresh):
    '''
    Return the last occurrence (index) of values satisfying the condition (logical_operation, thresh)
    in the 3D array along axis=0    
    ''' 
    
    arr_inverted = arr[::-1,:,:]
    
    # first occurrence in the inverted array   
    if logical_operation == 'gt':
        firs_occ=numpy.argmax(arr_inverted>thresh, axis=0)
        
    elif logical_operation == 'get':
        firs_occ=numpy.argmax(arr_inverted>=thresh, axis=0)   
            
    elif logical_operation == 'lt':
        firs_occ=numpy.argmax(arr_inverted<thresh, axis=0)    
            
    elif logical_operation == 'let':
        firs_occ=numpy.argmax(arr_inverted<=thresh, axis=0) 
             
    elif logical_operation == 'e':
        firs_occ=numpy.argmax(arr_inverted==thresh, axis=0)
    
    res=arr.shape[0]-firs_occ-1
        
    return res    


def get_binary_arr(arr1, arr2, logical_operation):
    '''
    Compare "arr1" with "arr2" and return a binary array with the result.
    
    :param arr1: array to comparer with arr2
    :type arr1: numpy.ndarray
    :param arr2: reference array or threshold 
    :type arr2: numpy.ndarray or float or int
    :rtype: binary numpy.ndarray
    
    ..warning:: "arr1" and "arr2" must have the same shape
    
    '''

    if logical_operation == 'gt':
        binary_arr = arr1 > arr2
        
    elif logical_operation == 'get':
            binary_arr = arr1 >= arr2
            
    elif logical_operation == 'lt':
            binary_arr = arr1 < arr2
            
    elif logical_operation == 'let':
            binary_arr = arr1 <= arr2 
            
    elif logical_operation == 'e':
            binary_arr = arr1 == arr2
    
    binary_arr = binary_arr.astype(int) # True/False ---> 1/0

    # if binary_arr is masked array, we fill masked values with 0
    if isinstance(binary_arr, numpy.ma.MaskedArray):
        binary_arr = binary_arr.filled(0.0)
    
    return binary_arr


def get_masked_arr(arr, fill_val):
    '''
    If a masked array is passed, this function does nothing.
    If a filled array is passed (fill_value must be passed also), it will be transformed into a masked array.
    
    '''
    if isinstance(arr, numpy.ma.MaskedArray):               # numpy.ma.MaskedArray
        masked_arr = arr
    else:                                                   # numpy.ndarray
        if (fill_val==None):
            raise(ValueError('If input array is not a masked array, a "fill_value" must be provided.'))
        mask_arr = (arr==fill_val)
        masked_arr = numpy.ma.masked_array(arr, mask=mask_arr, fill_value=fill_val)
    
    return masked_arr


def simple_stat(arr, stat_operation, logical_operation=None, thresh=None, coef=1.0, fill_val=None, index_event=False):
    
    '''    
    Used for computing: TG, TX, TN, TXx, TNx, TXn, TNn, PRCPTOT, SD
    
    :param arr:
    :type arr:
    
    :param stat_operation: Statistical operation to be applied to `arr`: 'min', 'max', 'mean', 'sum' 
    :type stat_operation: str
    
    :param coef: Coefficient to be multiplied on `arr`
    :type coef: float
    
    :param fill_val: Fill value
    :type fill_val: float

    :param threshold:
    :type threshold:
    
    :param logical_operation:
    :type logical_operation:
    
    'thresh' and 'logical_operation' will filter values, 
    for example if thresh=20 and logical_operation='lt',
    this function will filter all values < 20 before doing statistical operation.
    
    
    :param index_event: If True, returns the index where an event is found (only for 'max' and 'min') 
    :type index_event: bool


    
    '''
    
    arr_masked = get_masked_arr(arr, fill_val) * coef                # numpy.ma.MaskedArray with fill_value=fill_val (if numpy.ndarray passed) or fill_value=arr.fill_value (if numpy.ma.MaskedArray is passed)
                
    # condition: arr <logical_operation> <thresh>         
    if thresh != None:
        if logical_operation=='gt':
            mask_a = arr_masked <= thresh
        elif logical_operation=='get':
            mask_a = arr_masked < thresh
        elif logical_operation=='lt':
            mask_a = arr_masked >= thresh
        if logical_operation=='let':
            mask_a = arr_masked > thresh
        
        arr_masked = numpy.ma.array(arr_masked, mask=mask_a, fill_value=arr_masked.fill_value)
    
    
    if stat_operation=="mean":
        RESULT = arr_masked.mean(axis=0)                              # fill_value is changed: RESULT is a new numpy.ma.MaskedArray with default fill_value=999999 (!) => next line is to keep the fill_value of arr_masked
    elif stat_operation=="min":
        RESULT = arr_masked.min(axis=0)                              # fill_value is changed: RESULT is a new numpy.ma.MaskedArray with default fill_value=999999 (!) => next line is to keep the fill_value of arr_masked
        if index_event==True:
            index_event_arr=numpy.argmin(arr_masked, axis=0) # numpy.argmin works as well for masked arrays
        
    elif stat_operation=="max":
        RESULT = arr_masked.max(axis=0)                              # fill_value is changed: RESULT is a new numpy.ma.MaskedArray with default fill_value=999999 (!) => next line is to keep the fill_value of arr_masked
        if index_event==True:
            index_event_arr=numpy.argmax(arr_masked, axis=0) # numpy.argmax works as well for masked arrays
    elif stat_operation=="sum":
        RESULT = arr_masked.sum(axis=0)                              # fill_value is changed: RESULT is a new numpy.ma.MaskedArray with default fill_value=999999 (!) => next line is to keep the fill_value of arr_masked

    numpy.ma.set_fill_value(RESULT, arr_masked.fill_value)
    
    if not isinstance(arr, numpy.ma.MaskedArray):
        RESULT = RESULT.filled(fill_value=arr_masked.fill_value)      # numpy.ndarray filled with input fill_val
    
    if index_event==True and stat_operation in ['min', 'max']:
        return [RESULT, index_event_arr]
    else:
        return RESULT

# add params: out_inits (days, %), var_type = 'p'/'t'
# thresh: float or 'pXX'--> XXth pctl
# if type(thresh) is str ==> required params: out_inits, var_type
# if var_type == 'p' (precipitation): only one percentile value will be computed, without bootstrapping
# if var_type == 't' (temperature): daily percentiles dictionary will be created, with bootstrapping
def get_nb_days(arr, logical_operation, thresh, coef=1.0, fill_val=None, index_event=False, out_units=None, var_type=None):
    '''
    Used for computing: SU, TR, FD, ID, RR1, R10mm, R20mm, SD1, SD5cm, SD50cm
    
    :param thresh: temperature or precipitation threshold (must be the same unit as arr) 
    :type thresh: float
    '''
    
    if index_event==True:
        index_event_bounds=[]
    
    arr_masked = get_masked_arr(arr, fill_val) * coef
    arr_bin = get_binary_arr(arr_masked, thresh, logical_operation) # numpy.ndarray
    RESULT = arr_bin.sum(axis=0) # numpy.ndarray                    
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        RESULT = numpy.ma.array(RESULT, mask=RESULT==arr_masked.fill_value, fill_value=arr_masked.fill_value)
    
    if index_event==True:

        first_occurrence_event=get_first_occurrence(arr_bin, logical_operation='e', thresh=1)
        last_occurrence_event=get_last_occurrence(arr_bin, logical_operation='e', thresh=1)

        index_event_bounds=[first_occurrence_event, last_occurrence_event]
        
        return [RESULT, index_event_bounds]
    
    
    else:
        return RESULT

# #arr = numpy.random.randint(100, size=(200,2,3))
# 
# arr = 200.*numpy.random.random_sample(size=(300,300,400))
# 
# 
# res = get_nb_days(arr=arr, thresh=150, logical_operation='gt', coef=1.0, fill_val=32, date_event=True)
# 
# # print res[0]
# # print res[1]

def get_run_stat(arr, window_width, stat_mode, extreme_mode, coef=1.0, fill_val=None, index_event=False):
    
    '''
    Used for computing: RX5day
    '''
    
    assert(arr.ndim == 3)
    
    if index_event==True:
        index_event_bounds=[]
    
    arr_masked = get_masked_arr(arr, fill_val) * coef
    arr_filled = arr_masked.filled(fill_value=arr_masked.fill_value) # array must be filled for passing in C function
    
    
    ## array data type should be 'float32' to pass it to C function  
    if arr_filled.dtype != 'float32':
        arr_filled = numpy.array(arr_filled, dtype='float32')
    
    C_get_run_stat = libraryC.get_run_stat_3d
    C_get_run_stat.restype = None
    C_get_run_stat.argtypes = [ndpointer(ctypes.c_float), # const float *indata
                                ctypes.c_int, # int _sizeT
                                ctypes.c_int, # int _sizeI
                                ctypes.c_int, # int _sizeJ
                                ndpointer(ctypes.c_double), # double *outdata
                                ctypes.c_int, # int w_width
                                ctypes.c_float, # float fill_val
                                ctypes.c_char_p, # char * stat_mode
                                ctypes.c_char_p, # char * extreme_mode
                                ndpointer(ctypes.c_int) # int *index_event                                                              
                                ]
    
    res = numpy.zeros([arr_filled.shape[1], arr_filled.shape[2]]) # reserve memory
    first_index_event = numpy.zeros([arr_filled.shape[1], arr_filled.shape[2]], dtype='int32') # reserve memory
    
    C_get_run_stat(arr_filled, 
                   arr_filled.shape[0], 
                   arr_filled.shape[1], 
                   arr_filled.shape[2], 
                   res, 
                   window_width, 
                   fill_val,
                   stat_mode, 
                   extreme_mode, 
                   first_index_event)
    
    res = res.reshape(arr_filled.shape[1], arr_filled.shape[2])
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        res = numpy.ma.array(res, mask=res==arr_masked.fill_value, fill_value=arr_masked.fill_value)
    
    
    if index_event==False:
        return res
    else:
        first_index_event = first_index_event.reshape(arr_filled.shape[1], arr_filled.shape[2])
        
        last_index_event = first_index_event + (window_width-1)
        last_index_event[first_index_event==-1]=-1 # first_index_event=-1, i.e. no event found ==> last_index_event=-1
  
        index_event_bounds=[first_index_event, last_index_event]
        return [res, index_event_bounds] # [2D, [2D, 2D]]
    
### TODO: index_event in find_max_len_consec_sequence_3d
def get_max_nb_consecutive_days(arr, logical_operation, thresh, coef=1.0, fill_val=None, index_event=False):

    '''
    Used for computing: CSU, CFD, CDD, CWD
    '''
    
    if index_event==True:
        index_event_bounds=[]
    
    arr_masked = get_masked_arr(arr, fill_val) * coef
    arr_filled = arr_masked.filled(fill_value=arr_masked.fill_value) # array must be filled for passing in C function
    
    ######

        
    # array data type should be 'float32' to pass it to C function  
    if arr_filled.dtype != 'float32':
        arr_filled = numpy.array(arr_filled, dtype='float32')
    
    C_find_max_len_consec_sequence_3d = libraryC.find_max_len_consec_sequence_3d
    C_find_max_len_consec_sequence_3d.restype = None
    C_find_max_len_consec_sequence_3d.argtypes = [ndpointer(ctypes.c_float), # const float *indata
                                                    ctypes.c_int, # int _sizeT
                                                    ctypes.c_int, # int _sizeI
                                                    ctypes.c_int, # int _sizeJ
                                                    ndpointer(ctypes.c_double), # double *outdata
                                                    ctypes.c_float, # float thresh
                                                    ctypes.c_float, # float fill_val
                                                    ctypes.c_char_p, # char *operation
                                                    ndpointer(ctypes.c_int), # int *index_event_start
                                                    ndpointer(ctypes.c_int), # int *index_event_end
                                                    ] 
    
    RESULT = numpy.zeros([arr_filled.shape[1], arr_filled.shape[2]]) # reserve memory
    first_index_event = numpy.zeros([arr_filled.shape[1], arr_filled.shape[2]], dtype='int32') # reserve memory
    last_index_event = numpy.zeros([arr_filled.shape[1], arr_filled.shape[2]], dtype='int32') # reserve memory

    
    C_find_max_len_consec_sequence_3d(arr_filled, 
                                      arr_filled.shape[0], 
                                      arr_filled.shape[1], 
                                      arr_filled.shape[2], 
                                      RESULT, 
                                      thresh, 
                                      fill_val, 
                                      logical_operation, 
                                      first_index_event, 
                                      last_index_event)

    RESULT = RESULT.reshape(arr_filled.shape[1], arr_filled.shape[2])
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        RESULT = numpy.ma.array(RESULT, mask=RESULT==arr_masked.fill_value, fill_value=arr_masked.fill_value)

    if index_event==False:
        return RESULT
    else:
        first_index_event = first_index_event.reshape(arr_filled.shape[1], arr_filled.shape[2])
        last_index_event = last_index_event.reshape(arr_filled.shape[1], arr_filled.shape[2])      
        index_event_bounds=[first_index_event, last_index_event]
        return [RESULT, index_event_bounds] # [2D, [2D, 2D]] 




def TXXXp(arr, dt_arr, percentile_dict, logical_operation, fill_val=None, out_unit="days", index_event=False):
    
    TXXXp = numpy.zeros((arr.shape[1], arr.shape[2]))
    
    if index_event==True:
        index_event_bounds=[]
        bin_arr_3D = numpy.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
    
    arr_masked = get_masked_arr(arr, fill_val)
    
    i=0
    for dt in dt_arr:
        
        # current calendar day
        m = dt.month
        d = dt.day

        # we take the 2D array corresponding to the current calendar day
        current_perc_arr = percentile_dict[m,d]
                    
        # we are looking for the values which are g/ge/l/le than the XXth percentile  
        bin_arr = get_binary_arr(arr_masked[i,:,:], current_perc_arr, logical_operation=logical_operation) 
        TXXXp = TXXXp + bin_arr
        
        if index_event==True:
            bin_arr_3D[i,:,:] = bin_arr
        
        
        
        i+=1
   
    if out_unit == "days":
        TXXXp = TXXXp
    elif out_unit == "%":
        TXXXp = TXXXp*(100./len(dt_arr))
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        TXXXp = numpy.ma.array(TXXXp, mask=TXXXp==arr_masked.fill_value, fill_value=arr_masked.fill_value)
    
    if index_event==True:
        first_occurrence_event=get_first_occurrence(bin_arr_3D, logical_operation='e', thresh=1)
        last_occurrence_event=get_last_occurrence(bin_arr_3D, logical_operation='e', thresh=1)

        index_event_bounds=[first_occurrence_event, last_occurrence_event]
        
        return [TXXXp, index_event_bounds]   
    
    else:    
        return TXXXp  
    

def WCSDI(arr, dt_arr, percentile_dict, logical_operation, fill_val=None, N=6):
    '''
    Calculate the WSDI/CSDI indice (warm/cold-spell duration index).
    This function calls C function "WSDI_CSDI_3d" from libC.c
 
    '''

 
    arr_masked = get_masked_arr(arr, fill_val)
    
    # step1: we get a 3D binary array from arr (if arr value > corresponding 90th percentile value: 1, else: 0)    
    bin_arr = numpy.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
    
    i=0
    for dt in dt_arr:
        
        # current calendar day
        m = dt.month
        d = dt.day

        current_perc_arr = percentile_dict[m,d]
        
        # we are looking for the values which are greater than the 90th percentile  
        bin_arr_current_slice = get_binary_arr(arr_masked[i,:,:], current_perc_arr, logical_operation=logical_operation) 
#         bin_arr_current_slice = bin_arr_current_slice.filled(fill_value=0) # we fill the bin_arr_current_slice with zeros
        bin_arr[i,:,:] = bin_arr_current_slice

        i+=1
    
    # step2: now we will pass our 3D binary array (bin_arr) to C function WSDI_CSDI_3d
    
    # array data type should be 'float32' to pass it to C function  
    if bin_arr.dtype != 'float32':
        bin_arr = numpy.array(bin_arr, dtype='float32')
    
    
    WSDI_CSDI_C = libraryC.WSDI_CSDI_3d    
    WSDI_CSDI_C.restype = None
    WSDI_CSDI_C.argtypes = [ndpointer(ctypes.c_float),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ndpointer(ctypes.c_double),
                            ctypes.c_int] 
        
    WCSDI = numpy.zeros([arr.shape[1], arr.shape[2]]) # reserve memory
        
    WSDI_CSDI_C(bin_arr, bin_arr.shape[0], bin_arr.shape[1], bin_arr.shape[2], WCSDI, N)
    
    WCSDI = WCSDI.reshape(arr.shape[1], arr.shape[2])
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        WCSDI = numpy.ma.array(WCSDI, mask=WCSDI==arr_masked.fill_value, fill_value=arr_masked.fill_value)

    return WCSDI    


# def get_wet_days(arr, pr_thresh = 1.0, logical_operation='gt', fill_val=None):
#     '''
#     Return binary 3D array with the same same as arr, where 1 is a wet day.
#     '''
#     arr_masked = get_masked_arr(arr, fill_val)  # mm/s
#     arr_masked = arr_masked*60*60*24            # mm/day
#     
#     # we need to check only wet days (i.e. days with RR >= 1 mm)
#     # so, we mask all values < 1 mm with the same fill_value
#     mask_arr_masked = arr_masked < pr_thresh # mask
#     arr_masked_masked = numpy.ma.array(arr_masked, mask=mask_arr_masked, fill_value=arr_masked.fill_value)
#   
#     # we are looking for the values which are greater than the Xth percentile  
#     bin_arr = get_binary_arr(arr_masked_masked[:,:,:], percentile_arr, logical_operation=logical_operation)
    
    
    

def RXXp(arr, percentile_arr, logical_operation='gt', pr_thresh = 1.0, fill_val=None, out_unit="days", index_event=False):
    '''
    Calculate a RXXp indice, where XX is a percentile value.

    '''
    #RXXp = numpy.zeros((arr.shape[1], arr.shape[2]))
    
    if index_event==True:
        index_event_bounds=[]

    arr_masked = get_masked_arr(arr, fill_val)  # mm/day
    
    # we need to check only wet days (i.e. days with RR >= 1 mm)
    # so, we mask all values < 1 mm with the same fill_value
    mask_arr_masked = arr_masked < pr_thresh # mask
    arr_masked_masked = numpy.ma.array(arr_masked, mask=mask_arr_masked, fill_value=arr_masked.fill_value)
  
    # we are looking for the values which are greater than the Xth percentile  
    bin_arr = get_binary_arr(arr_masked_masked, percentile_arr, logical_operation=logical_operation) 
    RXXp = numpy.sum(bin_arr, axis=0)
     
    if out_unit == "days":
        RXXp = RXXp
    elif out_unit == "%":
        RXXp = RXXp*(100./arr.shape[0])
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(arr, numpy.ma.MaskedArray):
        RXXp = numpy.ma.array(RXXp, mask=RXXp==arr_masked.fill_value, fill_value=arr_masked.fill_value)
    
    if index_event==True:
        first_occurrence_event=get_first_occurrence(bin_arr, logical_operation='e', thresh=1)
        last_occurrence_event=get_last_occurrence(bin_arr, logical_operation='e', thresh=1)

        index_event_bounds=[first_occurrence_event, last_occurrence_event]
        
        return [RXXp, index_event_bounds]   
    
    else:
        return RXXp    


def RXXpTOT(arr, percentile_arr, logical_operation='gt', pr_thresh = 1.0, fill_val=None, input_units="mm/day"):
    '''
    Calculate a RXXpTOT indice, where XX is a percentile value.
    '''
    
    #RXXpTOT = numpy.zeros((arr.shape[1], arr.shape[2]))

    arr_masked = get_masked_arr(arr, fill_val)  # # mm/day  

    # we need to check only wet days (i.e. days with RR >= 1 mm)
    # so, we mask all values < 1 mm with the same fill_value
    mask_arr_masked = arr_masked < pr_thresh # mask
    arr_masked_masked = numpy.ma.array(arr_masked, mask=mask_arr_masked, fill_value=arr_masked.fill_value)
  
    # we are looking for the values which are greater than the Xth percentile  
    bin_arr = get_binary_arr(arr_masked_masked, percentile_arr, logical_operation=logical_operation)
    
    # we inverse bin_arr to get a mask (i.e. to mask values which are less or equal than the Xth percentile)
    maska = numpy.logical_not(bin_arr)
    
    # we apply the mask to arr_masked_masked
    arr_m = numpy.ma.array(arr_masked_masked, mask=maska, fill_value=arr_masked.fill_value)
    
    RXXpTOT = numpy.sum(arr_m, axis=0)

    if isinstance(arr, numpy.ma.MaskedArray):
        RXXpTOT = numpy.ma.array(RXXpTOT, mask=RXXpTOT==arr_masked.fill_value, fill_value=arr_masked.fill_value)
        
    
    return RXXpTOT

#### TODO: correct
def CD_CW_WD_WW(t_arr, t_percentile_dict, t_logical_operation, p_arr, p_percentile_dict, p_logical_operation, dt_arr, 
                pr_thresh = 1.0, fill_val1=None, fill_val2=None, out_unit="days"):
    '''
    Calculates the CD/CW/WD/WW indices.    
    '''
    # we intitialize the indice array
    RESULT = numpy.zeros((t_arr.shape[1], t_arr.shape[2]))
        
    
    # 1) we mask both arrays: t_arr and p_arr
    t_arr_masked = get_masked_arr(t_arr, fill_val1)
    p_arr_masked = get_masked_arr(p_arr, fill_val2)

    # 2) p_arr: mm/s ---> mm/day ; we are looking only for wet days (RR > 1 mm), i.e. we mask values < 1 mm
    #p_arr_masked = p_arr_masked*60*60*24            # mm/day
    mask_p_arr = p_arr_masked<pr_thresh
    p_arr_masked_masked = numpy.ma.array(p_arr_masked, mask=mask_p_arr) 

    
    i=0
    for dt in dt_arr:
        
        # current calendar day
        m = dt.month
        d = dt.day
        

        t_current_perc_arr = t_percentile_dict[m,d]
        p_current_perc_arr = p_percentile_dict[m,d]

        # 3) we compare daily mean temperature (t_arr) with its XXth percentile (t_percentile_dict)                   ==> result 1          
        t_bin_arr = get_binary_arr(t_arr_masked[i,:,:], t_current_perc_arr, logical_operation=t_logical_operation) 
        
        # 4) we compare daily precipitation amount at wet day (p_arr) with its XXth percentile (p_percentile_dict)    ==> result 2        
        p_bin_arr = get_binary_arr(p_arr_masked_masked[i,:,:], p_current_perc_arr, logical_operation=p_logical_operation) 
    
        # 5) result 1 AND result 2 ==> RESULT        
        t_bin_arr_AND_p_bin_arr = numpy.logical_and(t_bin_arr, p_bin_arr) # masked array              
        #t_bin_arr_AND_p_bin_arr_filled = t_bin_arr_AND_p_bin_arr.filled(fill_value=0)
        
#         RESULT = RESULT + t_bin_arr_AND_p_bin_arr_filled
        RESULT = RESULT + t_bin_arr_AND_p_bin_arr
        
        i+=1
    
    
    if out_unit == "days":
        RESULT = RESULT
    elif out_unit == "%":
        RESULT = RESULT*(100./len(dt_arr))
    
    # RESULT must be numpy.ma.MaskedArray if arr is numpy.ma.MaskedArray
    if isinstance(t_arr, numpy.ma.MaskedArray):
        RESULT = numpy.ma.array(RESULT, mask=RESULT==t_arr_masked.fill_value, fill_value=t_arr_masked.fill_value)
    
    
    return RESULT


def get_date_event_arr(dt_arr, index_arr, time_calendar, time_units, fill_val):
    ## dt_arr: 1D numpy array with datetime.datetime objects
    ## index_arr: 2D array with indices
    ## return: 2D array with with numeric dates 
    
    res = numpy.zeros((index_arr.shape[0], index_arr.shape[1]))
    
    for i in range(index_arr.shape[0]):
        for j in range(index_arr.shape[1]):     
            index =  index_arr[i,j] 
            
            if index==-1:
                date_num = fill_val 
            else:
                date_dt =  dt_arr[index]            
                date_num = util_dt.date2num(dt=date_dt, calend=time_calendar, units=time_units)
            res[i,j] = date_num
            
    return res


# def get_anomalie(arr, arr2, fill_val, fill_val2):
#     arr1_masked = get_masked_arr(arr, fill_val)
#     arr2_masked = get_masked_arr(arr2, fill_val2)
#     
#     anomalie = 