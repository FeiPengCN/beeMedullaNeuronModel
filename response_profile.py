####
# code to accompany the paper: 'Randomly weighted receptor inputs can explain the large diversity of colour-coding
# neurons in the bee visual system', by Vera Vasas, Fei Peng, HaDi MaBouDi, Lars Chittka, Scientific Reports, 2019
# for questions, contact Vera Vasas at v.vasas@qmul.co.uk or vvasas@gmail.com
####

##### CALCULATING THE RESPONSE PROFILE OF COLOUR SENSITIVE NEURONS IN THE MODEL


from scipy.spatial.distance import cdist
import numpy as np
import detect_peaks

## helper functions

def PiecewiseLinearTransformation(x,threshold1,threshold2):
    "0 response below threshold 1, 1 response above threshold2, linear in between, symmetric to negative and positive input"
    if x<=threshold2*-1.:
        y=-1
    elif x>=threshold2:
        y=1.
    elif (x<=threshold1 and x>=threshold1*-1.):
        y=0.
    elif x<0.:
        y= ((x*-1.-threshold1)/ (threshold2-threshold1))*-1.
    else:
        y= (x-threshold1)/ (threshold2-threshold1)
    return(y)
 
def SigmoidTransformationWithThresholds(x,threshold1, threshold2):
    "0 response below threshold 1, 1 response above threshold2, sigmoid in between, symmetric to negative and positive input"
    saturation=1.
    inflection_point=(threshold2-threshold1)/2.
    steepness= saturation/inflection_point*2.
    
    if x==0.:
        y=0.
    elif x>0.:
       y = saturation/2. * (1 +np.tanh (steepness*(x-threshold1-inflection_point)))
    else:
       y = (saturation/2. * (1 +np.tanh (steepness*(x*-1.-threshold1-inflection_point))))*-1.
    return(y)

def smooth(a,WSZ):  # copy and paste from https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

  
## response profile calculations for one neuron

class ResponseProfile:
    "Class of calculations for theoretical spectral sensitivity curves of colour coding neurons."
    
    def __init__(self,activationtype,activationparameters,stimulus_category, presynaptic_weights):
        "Initialize the instance of a response profile"
        self.activationtype= activationtype # activation function for the postsynaptic neuron
        self.activationparameters= activationparameters # parameters of the activation funtion
                                                         # [min threshold,max threshold] for linear piecewise and sigmoidal
               
        self.stimulus_category=stimulus_category # wavelenght of monochromatic light for example
        self.presynaptic_weights= presynaptic_weights # weights of inputs from presynaptic neurons
        
        
    def ActivationCalculation(self,stimulus_list):
        
        activation_list=[] # initialize the list for responses to list of presynaptic inputs
               
        for presynaptic_all in stimulus_list: # go through each stimulus input
                                              
            # go through each presynaptic neuron's input and sum them up
            total_presynaptic = sum([a*b for a,b in zip(presynaptic_all,self.presynaptic_weights)])
            
            if self.activationtype=='piecewise_linear': # linear function between two threshold values
                post_neuron_response = PiecewiseLinearTransformation(total_presynaptic,self.activationparameters[0],self.activationparameters[1])
            
            if self.activationtype=='sigmoid_threshold': # sigmoid function defined by two threshold values
                post_neuron_response = SigmoidTransformationWithThresholds(total_presynaptic,self.activationparameters[0],self.activationparameters[1])
                    
                                        
            activation_list.append(post_neuron_response)
                    
            self.activation_list=activation_list
         
    def GetPeaks(self,windowsize,peak_or_valley=False):
    
             if max(abs(np.array(self.activation_list)))==0:  # no peaks if no response
                 self.peaks=np.zeros(len(self.stimulus_category))
            
             else:
                normalized_activation_list=self.activation_list/ max(abs(np.array(self.activation_list))) # normalize the curve to max at 1, then the threshold for a peak is above o.5
                smoothed_activation_list=smooth(normalized_activation_list,windowsize)
            
                peak_locations =np.zeros(len(self.stimulus_category)) # output is an array where peaks are indicated by ones, index corresponds to wavelength
            
                peak_indices = detect_peaks.detect_peaks(smoothed_activation_list,mph =0.5, edge=None, valley=peak_or_valley)
                for index in peak_indices:
                  peak_locations[index]=1
            
                peak_indices_negative=detect_peaks.detect_peaks(smoothed_activation_list*-1.,mph =0.5, edge=None, valley=peak_or_valley) #look for peaks in inhibition
                for index in peak_indices_negative:
                   peak_locations[index]=1
                self.peaks=peak_locations
                
    
    
    def SquaredError(self, empirical_wavelength, empirical):
        "Calculates the squared difference and the R squared value between an empirical and a model response profile"
              
        activation_list_interpolated=np.interp(empirical_wavelength,self.stimulus_category,self.activation_list) # get the model values corresponding to emprically known data points
        squared_error= cdist(np.reshape(activation_list_interpolated,(1,-1)),np.reshape(empirical,(1,-1)),'sqeuclidean')[0][0] # calculate the squared difference between the two arrays
        zeros_empirical_array= np.zeros(len(empirical))
        total_error=cdist(np.reshape(zeros_empirical_array,(1,-1)),np.reshape(empirical,(1,-1)),'sqeuclidean')[0][0]

        self.squared_error=squared_error
        self.explained_squared=1. - squared_error/total_error


## response profile calculation for the network; fixed structure, based on morphology

class ResponseProfile2ndOrder:
    "Class of calculations for theoretical spectral sensitivity curves of a second order colour coding neuron, including the first order ones with fixed morphology."
    
    def __init__(self,par,activationparameters_1st,presynaptic_weights_1st_UV,presynaptic_weights_1st_blue,presynaptic_weights_1st_green,
                presynaptic_weights_1st_broad):
        "Initialize the instance of a network of five nodes" 
        
        self.first_narrow_UV=ResponseProfile(par.gain_function_1['gain_type'],activationparameters_1st,par.wavelength, presynaptic_weights_1st_UV)
        self.first_narrow_UV.ActivationCalculation(par.receptor_responses_list)
        
        self.first_narrow_blue=ResponseProfile(par.gain_function_1['gain_type'],activationparameters_1st,par.wavelength, presynaptic_weights_1st_blue)
        self.first_narrow_blue.ActivationCalculation(par.receptor_responses_list)
        
        self.first_narrow_green=ResponseProfile(par.gain_function_1['gain_type'],activationparameters_1st,par.wavelength, presynaptic_weights_1st_green)
        self.first_narrow_green.ActivationCalculation(par.receptor_responses_list)
        
        self.first_broad=ResponseProfile(par.gain_function_1['gain_type'],activationparameters_1st,par.wavelength, presynaptic_weights_1st_broad)
        self.first_broad.ActivationCalculation(par.receptor_responses_list)
        
        self.first_responses= zip(self.first_narrow_UV.activation_list,self.first_narrow_blue.activation_list,self.first_narrow_green.activation_list,self.first_broad.activation_list)
        
    def ActivationCalculation(self,par,activationparameters_2st,presynaptic_weights_2nd):   
        
        self.second=ResponseProfile(par.gain_function_2['gain_type'],activationparameters_2st,par.wavelength, presynaptic_weights_2nd)
        self.second.ActivationCalculation(self.first_responses)
  

                       