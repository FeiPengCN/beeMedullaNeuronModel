####
# code to accompany the paper: 'Randomly weighted receptor inputs can explain the large diversity of colour-coding
# neurons in the bee visual system', by Vera Vasas, Fei Peng, HaDi MaBouDi, Lars Chittka, Scientific Reports, 2019
# for questions, contact Vera Vasas at v.vasas@qmul.co.uk or vvasas@gmail.com
####

##### CALCULATIONS DESCRIBED IN THE PAPER: FINDING THE BEST FIT PARAMETERS


import csv
import numpy as np
import matplotlib.pyplot as plt

import parameters


par=parameters

       
#####

# variables to use for the FindBestFit function below
#
# outfile_csv_path='' # name of the csv file where to save the results
# outfile_plot_library='' # name of the library where to save the results
#
# empirical_index=1 # which empirical data set to use
#
# learning_rate weights=0.001
# learning_rate_alfa=0.001


##### helper functions

def GetPresynapticInput(receptor_responses, weight_third, baseline_UV=0., baseline_blue=0., baseline_green=0., weight_UV=[-1., 0.,0.], weight_blue= [0.,-1.,0.], weight_green=[0.,0.,-1.]):
    "Calculate the total presynaptic input to the third order cell"
    
    total_presynaptic= sum([a*b for a,b in zip(receptor_responses,weight_UV)]) # total input is weights multiplied by receptor responses given in the data file as input parameters
    firing_rate_tm_UV=total_presynaptic
                
    total_presynaptic= sum([a*b for a,b in zip(receptor_responses,weight_blue)]) # total input is weights multiplied by receptor responses given in the data file as input parameters
    firing_rate_tm_blue=total_presynaptic
                
    total_presynaptic= sum([a*b for a,b in zip(receptor_responses,weight_green)]) # total input is weights multiplied by receptor responses given in the data file as input parameters
    firing_rate_tm_green=total_presynaptic
    
    total_presynaptic= sum([a*b for a,b in zip([firing_rate_tm_UV,firing_rate_tm_blue,firing_rate_tm_green],weight_third)])
    
    presynaptic_inputs={'firing_rate_tm_uv':firing_rate_tm_UV,'firing_rate_tm_blue': firing_rate_tm_blue,'firing_rate_tm_green' :firing_rate_tm_green,'total_presynaptic': total_presynaptic}
    
    return(presynaptic_inputs)
    

def ActivationFunction(total_presynaptic, alfa, baseline_third=0.5, noise_st_dev=0.0000000001):
    "Calculate the firing rate of a third order cell with the specified parameters using a sigmoid activation function"

    noise= np.random.normal(loc=0.0, scale=noise_st_dev) # normal distribution noise
    half_response_point=np.log(1./99.) / alfa +0.75 # half response point is set so response is 0.95 at 0.75 input (ie threshold changes but max stimulus is at 0.9)
    
    if total_presynaptic < 0.:
        
        response_third_order =  - 1. / (1. + np.exp( -alfa * (-total_presynaptic - half_response_point) ) ) + noise
    
    else:
        
        response_third_order= 1. / (1. + np.exp( -alfa * (total_presynaptic - half_response_point) ) ) + noise
    
    return(response_third_order)
    
    
def DerivatedActivationFunction(total_presynaptic, alfa):
    "Derivation of the sigmoid activation function above"
     
    half_response_point=np.log(1./99.) / alfa +0.75 # half response point is set so response is 0.95 at 0.75 input (ie threshold changes but max stimulus is at 0.9)
    
    if total_presynaptic < 0.:
        
         derivated_response_third_order =   alfa * np.exp(-alfa*(-total_presynaptic - half_response_point)) / np.power(1. + np.exp(-alfa*(-total_presynaptic - half_response_point)),2.)
                 
    else:
        
         derivated_response_third_order = alfa * np.exp(-alfa*(total_presynaptic - half_response_point)) / np.power(1. + np.exp(-alfa*(total_presynaptic - half_response_point)),2.)
 
    return (derivated_response_third_order) 
    
def Derivated2AlfaActivationFunction(total_presynaptic, alfa):
    "Derivation of the sigmoid activation function above"
     
    half_response_point=np.log(1./99.) / alfa +0.75 # half response point is set so response is 0.95 at 0.75 input (ie threshold changes but max stimulus is at 0.9)
    
    if total_presynaptic < 0.:
        
        derivated_response_third_order = - (-total_presynaptic - half_response_point) * np.exp(-alfa*(-total_presynaptic - half_response_point)) / np.power(1. + np.exp(-alfa*(-total_presynaptic - half_response_point)),2.) #Patrick
                 
    else:
        
        derivated_response_third_order = (total_presynaptic - half_response_point) * np.exp(-alfa*(total_presynaptic - half_response_point)) / np.power(1. + np.exp(-alfa*(total_presynaptic - half_response_point)),2.) #Patrick
        
    return (derivated_response_third_order) 

def GetOneDerivatedCostWeight(weight_from_tm,firing_rate_tm,response_third_order,derivated_response_third_order,observed_response):
    "Calculated how much one particular observation adds to the derivated cost function for the weight"
    one_cost = 2. * firing_rate_tm * derivated_response_third_order * (response_third_order - observed_response) 

    return(one_cost)
    
def GetOneDerivatedCostAlfa(response_third_order,derivated_response_third_order_to_alfa,observed_response):
    "Calculated how much one particular observation adds to the derivated cost function for alfa (the steepness)"
    one_cost = 2. *  derivated_response_third_order_to_alfa * (response_third_order - observed_response) 

    return(one_cost)

## function for finding the  best fit parameters

def FindBestFit(par, empirical_index, learning_rate_weights,learning_rate_alfa,
                outfile_csv_path=None,outfile_plot_library=None,
                start_weights=[0.,0.,0.], start_alfa=10., min_alfa=7.5, increment_threshold=0.0001,
                baseline_UV=1., baseline_blue=1., baseline_green=1.,
                weight_UV=[-1., 0.,0.], weight_blue= [0.,-1.,0.], weight_green=[0.,0.,-1.]):
    "Find the weights for the inputs for empirical data based on the analytical solution"
    
    print 'empirical index', empirical_index
    
    #initialize parameters
    weights_t=start_weights
    alfa_t=start_alfa
    
    total_cost_tminus=0. # first cost t-1 for comparison: with the initial parameters
    for observation in par.empirical_curves_edited_analytical[empirical_index]:
            
             # the input and observed output for that observation
             receptor_responses=observation[1:4]
             observed_response=observation[4]
             
             # calculate the model predictions for that observation with the current parameters
             presynaptic_inputs=GetPresynapticInput(receptor_responses, weights_t, baseline_UV, baseline_blue, baseline_green, weight_UV, weight_blue, weight_green)
             response_third_order= ActivationFunction(presynaptic_inputs['total_presynaptic'], alfa_t) # no noise for the best fit
             
             #calculate the cost functions
             total_cost_tminus=total_cost_tminus + np.power(response_third_order-observed_response,2)
    print 'total cost at start', total_cost_tminus
    
    for i in range(10000000): 
        
        #reset the cost functions
        total_cost=0.
        
               
        derivated_cost_weight_uv=0.
        derivated_cost_weight_blue=0.
        derivated_cost_weight_green=0.
        derivated_cost_alfa=0.
        
        for observation in par.empirical_curves_edited_analytical[empirical_index]:
            
             # the input and observed output for that observation
             receptor_responses=observation[1:4]
             observed_response=observation[4]
             
             # calculate the model predictions for that observation with the current parameters
             presynaptic_inputs=GetPresynapticInput(receptor_responses, weights_t, baseline_UV, baseline_blue, baseline_green, weight_UV, weight_blue, weight_green)
             response_third_order= ActivationFunction(presynaptic_inputs['total_presynaptic'], alfa_t) # no noise for the best fit
             derivated_response_third_order= DerivatedActivationFunction(presynaptic_inputs['total_presynaptic'], alfa_t)
             derivated_response_third_order_to_alfa= Derivated2AlfaActivationFunction(presynaptic_inputs['total_presynaptic'], alfa_t)
             
             #calculate the cost functions
             total_cost=total_cost + np.power(response_third_order-observed_response,2)
                        
             derivated_cost_weight_uv= derivated_cost_weight_uv + GetOneDerivatedCostWeight(weights_t[0],presynaptic_inputs['firing_rate_tm_uv'],response_third_order,derivated_response_third_order,observed_response)
             derivated_cost_weight_blue= derivated_cost_weight_blue + GetOneDerivatedCostWeight(weights_t[1],presynaptic_inputs['firing_rate_tm_blue'],response_third_order,derivated_response_third_order,observed_response)
             derivated_cost_weight_green= derivated_cost_weight_green + GetOneDerivatedCostWeight(weights_t[2],presynaptic_inputs['firing_rate_tm_green'],response_third_order,derivated_response_third_order,observed_response)
             derivated_cost_alfa= derivated_cost_alfa + GetOneDerivatedCostAlfa(response_third_order,derivated_response_third_order_to_alfa,observed_response)
       
        print i, 'cost', total_cost, 'weights', weights_t, 'alfa', alfa_t, 'derivated_cost_alfa', derivated_cost_alfa
 
        learning_rate_weights_i=learning_rate_weights
        learning_rate_alfa_i=learning_rate_alfa    
            
        # get t plus one parameters
        weights_tplus=np.empty_like(weights_t)
       
        weights_tplus[0] = weights_t[0] - learning_rate_weights_i * derivated_cost_weight_uv
        weights_tplus[1] = weights_t[1] - learning_rate_weights_i * derivated_cost_weight_blue
        weights_tplus[2] = weights_t[2] - learning_rate_weights_i * derivated_cost_weight_green
        alfa_tplus = max(alfa_t - learning_rate_alfa_i * derivated_cost_alfa,min_alfa) # don't let alfa be too low, otherwise it will respond without stimulation
        
        #assess the increment
        parameters_tplus= np.append(weights_tplus,alfa_tplus)
        parameters_t=np.append(weights_t,alfa_t)
        increment = np.linalg.norm(np.subtract(parameters_tplus,parameters_t)) #
        if increment < increment_threshold: # terminate the run if the change in the parameters is negligable
            break
        
        
        # update the parameters
        weights_t=weights_tplus
        alfa_t=alfa_tplus
        total_cost_tminus=total_cost
        
    print 'weights', weights_t, 'alfa', alfa_t
    print 'cost', total_cost
    
    if outfile_csv_path!=None:
        with open(outfile_csv_path, 'ab') as out_file:
            writer=csv.writer(out_file, delimiter=',')
            header_row=['learning_rate_weights','learning_rate_alfa','increment_threshold',
                        'empirical_index','baseline_UV', 'baseline_blue', 'baseline_green',
                        'weight_to_UV_tm','weight_to_blue_tm','weight_to_green_tm',
                        'alfa','weight_UV','weight_blue','weight_green','cost','sq_explained','no of iterations']
            writer.writerow((header_row))  
            test_result=[]
            test_result.extend([learning_rate_weights,learning_rate_alfa,increment_threshold])
            test_result.append(empirical_index)
            test_result.extend([baseline_UV,baseline_blue,baseline_green])
            test_result.extend([weight_UV,weight_blue,weight_green])
            test_result.append(alfa_t)
            test_result.extend(weights_t)
            test_result.append(total_cost)
            test_result.append('N')
            test_result.append(i)
            writer.writerow((test_result))
       
    return{'weights':weights_t,'alfa':alfa_t}

## additional functions
    
def GetSpectralProfile(par, weights, alfa, outfile_csv_path=None):
    "Calculate the responses for monochromatic inputs with the specified parameters"
    
    #calculate the model responses with the specified parameters
    spectral_profile=np.empty(np.size(par.wavelength))
    for wavelength_index in np.arange(np.size(par.wavelength)):
        receptor_responses=[par.receptor_responses_UV[wavelength_index], par.receptor_responses_blue[wavelength_index],par.receptor_responses_green[wavelength_index]]
        presynaptic_inputs = GetPresynapticInput(receptor_responses, weights) #this is with fixed baseline and receptor to TM cell parameters! needs to be changed if you want anything than the default
        response_third_order = ActivationFunction(presynaptic_inputs['total_presynaptic'], alfa)
        spectral_profile[wavelength_index]=response_third_order
    print 'model spectral response profile', spectral_profile
    
    if outfile_csv_path!=None:
        with open(outfile_csv_path, 'ab') as out_file:
            writer=csv.writer(out_file, delimiter=',')
            writer.writerow(par.wavelength.tolist())
            writer.writerow(spectral_profile)
    
    return (spectral_profile)
    
    
def Compare2Empirical(par, empirical_index,weights,alfa,outfile_csv_path=None,outfile_plot_library=None):
    "Compare the proposed fit against the original data"
    #plot the response profile and the empirical data
    spectral_profile = GetSpectralProfile(par, weights, alfa,outfile_csv_path)
    plt.close()
    plt.plot(par.wavelength, spectral_profile)
    plt.plot(par.empirical_curves_edited_analytical[empirical_index][:,0], par.empirical_curves_edited_analytical[empirical_index][:,4])
    plt.show()
    
    if outfile_plot_library!=None:
      figure_path=outfile_plot_library+'figure_'+str(empirical_index)+'.png'
      plt.savefig(figure_path)
      