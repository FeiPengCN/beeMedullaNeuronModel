####
# code to accompany the paper: 'Randomly weighted receptor inputs can explain the large diversity of colour-coding
# neurons in the bee visual system', by Vera Vasas, Fei Peng, HaDi MaBouDi, Lars Chittka, Scientific Reports, 2019
# for questions, contact Vera Vasas at v.vasas@qmul.co.uk or vvasas@gmail.com
####

##### CALCULATIONS DESCRIBED IN THE PAPER: PARAMETER EXPLORATION, COMPARING THE MODEL PREDICTIONS
##### TO EMPIRICAL DATA, GENERATING A LIBRARY OF RANDOMLY WIRED NEURONS


import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


import response_profile
import parameters

######

par=parameters
stim_cat=parameters.wavelength

# variables to use for the functions below

# outfile_csv_path='' # name of the csv file where to save the results
# outfile_plot_library='' # name of the library where to save the results
#
# thresholds_1=[[]] # list of threshold parameters between receptors and TM cells
# threshold_1=[]  # one set of threshold parameters between receptors and TM cells

# weights_1=[[]] # list of weight parameters between receptors and TM cells
# weight_UV=[-1.,0.,0.] # weights from the receptors to the UV-sensitive TM cells
# weight_blue=[0.,-1.,0.] # weights from the receptors to the blue-sensitive TM cells
# weight_green=[0.,0.,-1.] # weights from the receptors to the green-sensitive TM cells
# weight_broad=[0.,0.,0.] # weights from the receptors to the broad-band-sensitive TM cells
#
# thresholds_2=[[]] # list of threshold parameters between TM cells and amacrine/large-field cells
# threshold_2=[] # one set of threshold parameters between TM cells and amacrine/large-field cells
#
# weights_2=[[]] # list of weight parameters between TM cells and amacrine/large-field cells
# weight_2=[] # one set of weight parameters between TM cells and amacrine/large-field cells
#
# empirical_index=1 # which empirical data set to use

# trial_no=5500 # when generating a random library: number of random neurons to be generated


#

def ExploreParameters1st(par,stim_cat,thresholds_1,weights_1,outfile_csv_path=None):
    "Explore a range of parameters for the connection between receptors and the TM cells"
    
    if outfile_csv_path!=None: # save the results if an out file path is given
        with open(outfile_csv_path, 'ab') as out_file:
            writer=csv.writer(out_file, delimiter=',')
            header_row=['threshold1_1','threshold1_2','weight_UV_rec','weight_blue_rec','weight_green_rec']
            header_row.extend(stim_cat.tolist())
            writer.writerow((header_row))  
    
    for gain_threshold in thresholds_1:
        for pre_weight in weights_1:
            test=response_profile.ResponseProfile(par.gain_function_1['gain_type'],gain_threshold,stim_cat,pre_weight)
            test.ActivationCalculation(parameters.receptor_responses_list)
            if outfile_csv_path!=None:
                with open(outfile_csv_path, 'ab') as out_file:
                    writer=csv.writer(out_file, delimiter=',')
                    test_result=[]
                    test_result.extend(gain_threshold)
                    test_result.extend(pre_weight)
                    test_result.extend(test.activation_list)
                    writer.writerow((test_result))

    if len(thresholds_1)==1 and len(weights_1) == 1: # if there is only one calculation, show the plot and the values
        plt.close()
        plt.plot(stim_cat,test.activation_list)
        plt.show()
        print test.activation_list        

def ExploreParameters2nd(par,stim_cat,threshold_1,weight_UV,weight_blue,weight_green,weight_broad,thresholds_2,weights_2,outfile_csv_path=None):
    "Explore a range of parameters for the connection between the TM cells and the amacrine/large-field cells"
    
    if outfile_csv_path!=None: # save the results if an out file path is given
        with open(outfile_csv_path, 'ab') as out_file:
            writer=csv.writer(out_file, delimiter=',')
            header_row=['threshold1_1','threshold1_2','weight_to_UV_narrow','weight_to_blue_narrow','weight_to_green_narrow','weight_to_broad',
                        'threshold2_1','threshold2_2','weight_UV_narrow','weight_blue_narrow','weight_green_narrow','weight_broad']
            header_row.extend(stim_cat.tolist())
            header_row.extend(stim_cat.tolist())
            writer.writerow((header_row))  

    for gain_threshold in thresholds_2:
        for pre_weight in weights_2:
            
            test=response_profile.ResponseProfile2ndOrder(par,threshold_1,weight_UV,weight_blue,weight_green,
                weight_broad)
            test.ActivationCalculation(par,gain_threshold,pre_weight)
            test.second.GetPeaks(5) # !!window size for smoothing is set to a constant here
            
            
            if outfile_csv_path!=None:
                with open(outfile_csv_path, 'ab') as out_file:
                    writer=csv.writer(out_file, delimiter=',')
                    test_result=[]
                    test_result.extend(threshold_1)
                    test_result.append(weight_UV)
                    test_result.append(weight_blue)
                    test_result.append(weight_green)
                    test_result.append(weight_broad)
                    test_result.extend(gain_threshold)
                    test_result.extend(pre_weight)
                    test_result.extend(test.second.activation_list)
                    test_result.extend(test.second.peaks)
                    writer.writerow((test_result))
                    
    if len(thresholds_2)==1 and len(weights_2) == 1: # if there is only one calculation, show the plot and the values
        plt.close()
        plt.plot(stim_cat,test.second.activation_list)
        plt.show()
        print test.second.activation_list        


def CompareEmpirical2Model(par,threshold_1,weight_UV,weight_blue,weight_green, weight_broad,
                            threshold_2,weight_2,empirical_index):
    "Compare the fit between a model and an empirical spectral tuning curve"
    
    example=response_profile.ResponseProfile2ndOrder(par,threshold_1,weight_UV,weight_blue,weight_green, weight_broad)
    example.ActivationCalculation(par,threshold_2,weight_2)
    example.second.SquaredError(par.empirical_curves_edited[empirical_index][:,0], par.empirical_curves_edited[empirical_index][:,1]) # edited empirical profile
    plt.close()
    plt.plot(example.second.stimulus_category, example.second.activation_list)
    plt.plot(par.empirical_curves[empirical_index][:,0], par.empirical_curves[empirical_index][:,1])
    plt.plot(par.empirical_curves_edited[empirical_index][:,0], par.empirical_curves_edited[empirical_index][:,1])
    plt.show()
    print 'example.second.activation_list', example.second.activation_list
    print 'example.second.squared_error', example.second.squared_error
    print 'example.second.explained_squared',example.second.explained_squared


def Generate2ndOrderRandomLibrary(par, threshold_1,weight_UV,weight_blue,weight_green, weight_broad,
                           trial_no,outfile_csv_path):
    "Generate a library of second order responses given parameters and first order weights/thresholds"
        
    precision=2 # round double numbers to this decimal precision
    initialized_neuron=response_profile.ResponseProfile2ndOrder(par,threshold_1,weight_UV,weight_blue,weight_green,
                weight_broad)  # the weight up to the first order neurons are the same and stored here
    
    with open(outfile_csv_path, 'ab') as out_file:
            writer=csv.writer(out_file, delimiter=',')
            header_row=['adaptation_UV','adaptation_blue','adaptation_green','threshold1_1','threshold1_2',
                        'weight_to_UV_narrow','weight_to_blue_narrow','weight_to_green_narrow','weight_to_broad',
                        'threshold2_1','threshold2_2','weight_UV_narrow','weight_blue_narrow','weight_green_narrow']
            header_row.extend(par.wavelength.tolist())
            header_row.extend(par.wavelength.tolist())
            writer.writerow((header_row))  
    
    for trial in np.arange(trial_no):
            
            random_weights=np.random.uniform(-1.,1.,3)
            trial_presynaptic_weights= np.around(random_weights/sum(abs(random_weights)),precision) # set weights so they are random and their absolutes add up to 1
            
            trial_total_presynaptic_list=[]
            for presynaptic_all in initialized_neuron.first_responses: # go through each stimulus input
            # go through each presynaptic neuron's input and sum them up
                   trial_total_presynaptic = sum([a*b for a,b in zip(presynaptic_all,trial_presynaptic_weights)])
                   trial_total_presynaptic_list.append(trial_total_presynaptic)
  
            
            max_positive_threshold=np.ceil(max(trial_total_presynaptic_list)*np.power(10.,precision)) / np.power(10.,precision) # round UP to given precision (default 2); so that the max threshold is always larger than the max or the presynaptic input (could be less because of rounding errors)
            max_negative_threshold=np.ceil(abs(min(trial_total_presynaptic_list))*np.power(10.,precision)) / np.power(10.,precision) #same for peak in the negative region
            max_threshold=max(max_positive_threshold,max_negative_threshold)  # 
            
                      
            #### TWO OPTIONS TO SET THRESHOLDS HERE
            #thresholds_ok=False     # find thresholds which are random and meaningful
            #while thresholds_ok==False:
            #    threshold_2_1=round(np.random.uniform(0.,max_threshold),precision)
            #    threshold_2_2=round(np.random.uniform(0.,max_threshold),precision)
            #    if threshold_2_2>=threshold_2_1:
            #        thresholds_ok=True
            
            # setting the second threshold to the max
            threshold_2_1=round(np.random.uniform(0.,max_threshold),precision)
            threshold_2_2=max_threshold
            
            

            post_neuron_response_list =[response_profile.PiecewiseLinearTransformation(total_presynaptic,threshold_2_1,threshold_2_2) for total_presynaptic in trial_total_presynaptic_list] # calculate the neron response profile with the given parameters
                                 
            test_neuron=initialized_neuron
            test_neuron.second=response_profile.ResponseProfile(par.gain_function_2['gain_type'],[threshold_2_1,threshold_2_2],par.wavelength, trial_presynaptic_weights)
            test_neuron.second.activation_list=post_neuron_response_list
            test_neuron.second.GetPeaks(5) # !! window size for smoothing is set to a constant here
                                
            with open(outfile_csv_path, 'ab') as out_file:
                   writer=csv.writer(out_file, delimiter=',')
                   test_result=[]
                   test_result.extend([par.receptor_adaptation_UV,par.receptor_adaptation_blue,par.receptor_adaptation_green])
                   test_result.extend(threshold_1)
                   test_result.append(weight_UV)
                   test_result.append(weight_blue)
                   test_result.append(weight_green)
                   test_result.append(weight_broad)
                   test_result.extend([threshold_2_1,threshold_2_2])
                   test_result.extend(trial_presynaptic_weights)
                   test_result.extend(test_neuron.second.activation_list)
                   test_result.extend(test_neuron.second.peaks)
                   writer.writerow((test_result))
