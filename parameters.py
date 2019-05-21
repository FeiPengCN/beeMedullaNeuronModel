####
# code to accompany the paper: 'Randomly weighted receptor inputs can explain the large diversity of colour-coding
# neurons in the bee visual system', by Vera Vasas, Fei Peng, HaDi MaBouDi, Lars Chittka, Scientific Reports, 2019
# for questions, contact Vera Vasas at v.vasas@qmul.co.uk or vvasas@gmail.com
####

##### SETTING AND UPLOADING THE PARAMETERS FOR THE MODEL


import numpy as np

wavelength=np.arange(300.,701.,5.) #wavelength resolution for the model
empirical_path='' # PATH TO THE EMPIRICAL DATA;CSV


## bee receptor sensitivity curves, smoothed and normalized to have their maximum at 1., 
receptor_sensitivity_UV=np.array([0.334728033, 0.428870293, 0.529288703, 0.629707113, 0.730125523, 0.824267782, 0.907949791, 0.968619247, 1.,          0.99790795,
                                  0.964435146, 0.90167364,  0.822175732, 0.732217573, 0.640167364, 0.550209205, 0.462343096, 0.380753138, 0.307531381, 0.242677824,
                                  0.190376569, 0.150627615, 0.119246862, 0.094142259, 0.075313808, 0.060669456, 0.048117155, 0.039748954, 0.033472803, 0.029288703,
                                  0.025104603, 0.023012552, 0.018828452, 0.016736402, 0.014644351, 0.012552301, 0.008368201, 0.006276151, 0.0041841  , 0.00209205,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0.])                                 

receptor_sensitivity_blue = np.array([0.081967213, 0.109289617, 0.137295082, 0.163934426, 0.19057377 , 0.219262295, 0.24795082 , 0.274590164, 0.301229508, 0.325819672,
                                      0.348360656, 0.366803279, 0.385245902, 0.405737705, 0.428278689, 0.456967213, 0.495901639, 0.545081967, 0.600409836, 0.663934426,
                                      0.727459016, 0.790983607, 0.848360656, 0.901639344, 0.944672131, 0.979508197, 0.99795082 , 1.         , 0.975409836, 0.930327869,
                                      0.866803279, 0.790983607, 0.709016393, 0.62704918 , 0.545081967, 0.463114754, 0.379098361, 0.297131148, 0.225409836, 0.163934426,
                                      0.116803279, 0.086065574, 0.06557377 , 0.051229508, 0.040983607, 0.030737705, 0.020491803, 0.012295082, 0.006147541, 0.00204918,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0.])

receptor_sensitivity_green=np.array([0.12195122 , 0.138888889, 0.150406504, 0.160569106, 0.168699187, 0.176829268, 0.182926829, 0.18699187 , 0.191056911, 0.195121951,
                                     0.197154472, 0.199186992, 0.201219512, 0.201219512, 0.201219512, 0.201219512, 0.199186992, 0.195121951, 0.195121951, 0.193089431,
                                     0.195121951, 0.203252033, 0.215447154, 0.227642276, 0.243902439, 0.262195122, 0.280487805, 0.304878049, 0.333333333, 0.365853659,
                                     0.402439024, 0.443089431, 0.485772358, 0.526422764, 0.56504065 , 0.599593496, 0.632113821, 0.662601626, 0.697154472, 0.735772358,
                                     0.776422764, 0.819105691, 0.861788618, 0.900406504, 0.932926829, 0.961382114, 0.983739837, 0.99796748 , 1.         , 0.993902439,
                                     0.967479675, 0.922764228, 0.863821138, 0.794715447, 0.711382114, 0.62398374 , 0.534552846, 0.445121951, 0.363821138, 0.288617886,
                                     0.227642276, 0.178861789, 0.138211382, 0.103658537, 0.079268293, 0.058943089, 0.042682927, 0.030487805, 0.022357724, 0.014227642,
                                     0.008130081, 0.004065041, 0.00203252,  0., 0., 0., 0., 0., 0., 0.,
                                     0.])


## the adaptation value that sets the receptor's sensitivities    - applicable for monochromatic responses                    
receptor_adaptation_UV=6.
receptor_adaptation_blue=6.
receptor_adaptation_green=6.

## assuming constant illumination, the receptor responses to light are calculated as the following; stored as list
receptor_responses_UV=(receptor_sensitivity_UV*receptor_adaptation_UV)/(1.+receptor_sensitivity_UV*receptor_adaptation_UV)
receptor_responses_blue=(receptor_sensitivity_blue*receptor_adaptation_blue)/(1.+receptor_sensitivity_blue*receptor_adaptation_blue)
receptor_responses_green=(receptor_sensitivity_green*receptor_adaptation_green)/(1.+receptor_sensitivity_green*receptor_adaptation_green)
receptor_responses_list=np.column_stack((receptor_responses_UV, receptor_responses_blue,receptor_responses_green)).tolist()

## read in the empirical data on spectral tuning curves of second order neurons from Kien Menzel 1977a, 1977b
empirical_library=empirical_path
empirical_path=(
 empirical_library+'01_unit14_s.csv',
 empirical_library+'02_unit16_s.csv',
 empirical_library+'03_unit26_on.csv',
 empirical_library+'04_unit21_on.csv',
 empirical_library+'05_unit21_s.csv',
 empirical_library+'06_unit32_s.csv',
 empirical_library+'07_unit58_on.csv',
 empirical_library+'08_unit58_s.csv',
 empirical_library+'11_unit38_on.csv',
 empirical_library+'12_unit38_s.csv',
 empirical_library+'13_unit27_on.csv',
 empirical_library+'14_unit27_s.csv',
 empirical_library+'17_unit50_on.csv',
 empirical_library+'20_unit74_s.csv',
 empirical_library+'21_unit69_s.csv',
 empirical_library+'22_unit29_onoff.csv',
 empirical_library+'23_unit29_s.csv',
 empirical_library+'24_unit11_s.csv',
 empirical_library+'25_unit22_on.csv',
 empirical_library+'26_unit22_s.csv',
 empirical_library+'27_unit13_s.csv',
 empirical_library+'28_unit13a_on.csv')

empirical_curves=[]
for empirical_each_path in empirical_path:
    empirical_curves.append( np.genfromtxt(empirical_each_path, delimiter=',')) # the empirical curves as stored as a list of numpy arrays

    ## read in the edited empirical curves    

empirical_path_edited=(
 [empirical_library+'01_unit14_s_edited.csv',
 empirical_library+'02_unit16_s_edited.csv',
 empirical_library+'03_unit26_on_edited.csv',
 empirical_library+'04_unit21_on_edited.csv',
 empirical_library+'05_unit21_s_edited.csv',
 empirical_library+'06_unit32_s_edited.csv',
 empirical_library+'07_unit58_on_edited.csv',
 empirical_library+'08_unit58_s_edited.csv',
 empirical_library+'11_unit38_on_edited.csv',
 empirical_library+'12_unit38_s_edited.csv',
 empirical_library+'13_unit27_on_edited.csv',
 empirical_library+'14_unit27_s_edited.csv',
 empirical_library+'17_unit50_on_edited.csv',
 empirical_library+'20_unit74_s_edited.csv',
 empirical_library+'21_unit69_s_edited.csv',
 empirical_library+'22_unit29_onoff_edited.csv',
 empirical_library+'23_unit29_s_edited.csv',
 empirical_library+'24_unit11_s_edited.csv',
 empirical_library+'25_unit22_on_edited.csv',
 empirical_library+'26_unit22_s_edited.csv',
 empirical_library+'27_unit13_s_edited.csv',
 empirical_library+'28_unit13a_on_edited.csv'])

empirical_curves_edited=[]
for empirical_each_path in empirical_path_edited:
    empirical_curves_edited.append( np.genfromtxt(empirical_each_path, delimiter=',')) # the empirical curves as stored as a list of numpy arrays

    ## read in the edited empirical curves that have the receptor responses in them (use this format in future)

empirical_path_edited_analytical=(
 [empirical_library+'01_unit14_s_edited_analytical.csv',
  empirical_library+'02_unit16_s_edited_analytical.csv',
  empirical_library+'03_unit26_on_edited_analytical.csv',
  empirical_library+'04_unit21_on_edited_analytical.csv',
  empirical_library+'05_unit21_s_edited_analytical.csv',
  empirical_library+'06_unit32_s_edited_analytical.csv',
  empirical_library+'07_unit58_on_edited_analytical.csv',
  empirical_library+'08_unit58_s_edited_analytical.csv',
  empirical_library+'11_unit38_on_edited_analytical.csv',
  empirical_library+'12_unit38_s_edited_analytical.csv',
  empirical_library+'13_unit27_on_edited_analytical.csv',
  empirical_library+'14_unit27_s_edited_analytical.csv',
  empirical_library+'17_unit50_on_edited_analytical.csv',
  empirical_library+'20_unit74_s_edited_analytical.csv',
  empirical_library+'21_unit69_s_edited_analytical.csv',
  empirical_library+'22_unit29_onoff_edited_analytical.csv',
  empirical_library+'23_unit29_s_edited_analytical.csv',
  empirical_library+'24_unit11_s_edited_analytical.csv',
  empirical_library+'25_unit22_on_edited_analytical.csv',
  empirical_library+'26_unit22_s_edited_analytical.csv',
  empirical_library+'27_unit13_s_edited_analytical.csv',
  empirical_library+'28_unit13a_on_edited_analytical.csv'])
 
empirical_curves_edited_analytical=[]
for empirical_each_path in empirical_path_edited_analytical:
    empirical_curves_edited_analytical.append( np.genfromtxt(empirical_each_path, delimiter=',')) # the empirical curves and receptor inputs are stored as a list of numpy arrays
 


# the total presynaptic input ranges from -1 to 1
presynaptic_total_1={'pre_min': -1., 'pre_max':0.}  
presynaptic_total_2={'pre_min': -1., 'pre_max':1.}

## set the parameters of the gain functions as a range that should be explored
gain_function_1={'gain_type':'piecewise_linear', 'threshold_1':np.arange(0.,0.01,0.1), 'threshold_2':np.arange(0.5,0.51,0.1)}
gain_function_2={'gain_type':'piecewise_linear', 'threshold_1':np.arange(0.3,0.31,0.1), 'threshold_2':np.arange(0.6,0.61,0.1)}
 