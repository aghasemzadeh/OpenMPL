import yaml
import os
import shutil
from lib.core.config import config
from lib.core.config import update_config
from tqdm import tqdm 
import copy
import pandas as pd

from itertools import combinations

def generate_combinations(input_list, n):
    return list(combinations(input_list, n))

''' 
                ######### Default params #########
    name:   cmu_0_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0
    
    
    'NETWORK.TRANSFORMER_ADD_CONFIDENCE_INPUT': False
    'NETWORK.TRANSFORMER_MULT_CONFIDENCE_EMB': False
    'NETWORK.TRANSFORMER_CONCAT_CONFIDENCE_EMB': False
    'NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD': True
    'NETWORK.TRANSFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT': False
    'NETWORK.TRANSFORMER_INPUT_RAYS_AS_TOKEN': True
    'NETWORK.POSE_3D_EMB_LEARNABLE': True
    'NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_TO_RAYS': True
    
    
    'NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS': True
    'NETWORK.TRANSFORMER_NO_SPT': False
    'NETWORK.TRANSFORMER_NO_FPT': False
    
    'NETWORK.TRANSFORMER_CONFIDENCE_IN_FPT': False
    
    'NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP': False
    '''

def make_different_parameters(configs_ro_run):
    different_params = []
    for i in range(len(configs_ro_run)):
        different_params.append({
            'name': [],
            'modifications': {}
        })
        ####### FPT
        if configs_ro_run.loc[i]['FPT'] == 'No':
            different_params[i]['name'].append(('_FPT_', '_NoFPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_NO_FPT'] = True
        elif configs_ro_run.loc[i]['FPT'] == 'Yes':
            different_params[i]['name'].append(('_FPT_', '_FPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_NO_FPT'] = False
        else:
            raise
        ####### SPT
        if configs_ro_run.loc[i]['SPT'] == 'No':
            different_params[i]['name'].append(('_MultiSPT_', '_NoSPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_NO_SPT'] = True
            different_params[i]['modifications']['NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS'] = False
        elif configs_ro_run.loc[i]['SPT'] == 'Single':
            different_params[i]['name'].append(('_MultiSPT_', '_SingleSPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS'] = False
        elif configs_ro_run.loc[i]['SPT'] == 'Multi':
            different_params[i]['name'].append(('_MultiSPT_', '_MultiSPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_MULTIPLE_SPATIAL_BLOCKS'] = True
        else:
            raise
        ####### Raytoken
        if configs_ro_run.loc[i]['Raytoken'] == False:
            different_params[i]['name'].append(('_Raytoken_', '_RaytokenNo_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_INPUT_RAYS_AS_TOKEN'] = False
        elif configs_ro_run.loc[i]['Raytoken'] == True:
            different_params[i]['name'].append(('_Raytoken_', '_Raytoken_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_INPUT_RAYS_AS_TOKEN'] = True
        else:
            raise
        ####### Learn3dEnc
        if configs_ro_run.loc[i]['Learn3dEnc'] == False:
            different_params[i]['name'].append(('_Learn3dEnc_', '_Learn3dEncNo_'))
            different_params[i]['modifications']['NETWORK.POSE_3D_EMB_LEARNABLE'] = False   
        elif configs_ro_run.loc[i]['Learn3dEnc'] == True:
            different_params[i]['name'].append(('_Learn3dEnc_', '_Learn3dEnc_'))
            different_params[i]['modifications']['NETWORK.POSE_3D_EMB_LEARNABLE'] = True
        else:
            raise
        ####### Add3dEncRays
        if configs_ro_run.loc[i]['Add3dEncRays'] == False:
            different_params[i]['name'].append(('_Add3dEncRays_', '_Add3dEncRaysNo_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_TO_RAYS'] = False
        elif configs_ro_run.loc[i]['Add3dEncRays'] == True:
            different_params[i]['name'].append(('_Add3dEncRays_', '_Add3dEncRays_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_ADD_3D_POS_ENCODING_TO_RAYS'] = True
        else:
            raise
        ####### Conf
        if configs_ro_run.loc[i]['Conf'] == 'No':
            different_params[i]['name'].append(('_Conf3rd_', '_Confno_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
        elif configs_ro_run.loc[i]['Conf'] == 'Add':
            different_params[i]['name'].append(('_Conf3rd_', '_ConfAdd_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_ADD_CONFIDENCE_INPUT'] = True
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
        elif configs_ro_run.loc[i]['Conf'] == 'Mult':
            different_params[i]['name'].append(('_Conf3rd_', '_ConfMult_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_MULT_CONFIDENCE_EMB'] = True
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
        elif configs_ro_run.loc[i]['Conf'] == 'Concat':
            different_params[i]['name'].append(('_Conf3rd_', '_ConfConcat_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONCAT_CONFIDENCE_EMB'] = True
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
        elif configs_ro_run.loc[i]['Conf'] == 'Weight':
            different_params[i]['name'].append(('_Conf3rd_', '_ConfWeight_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONF_ATTENTION_UNCERTAINTY_WEIGHT'] = True
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = False
        elif configs_ro_run.loc[i]['Conf'] == '3rd':
            different_params[i]['name'].append(('_Conf3rd_', '_Conf3rd_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_INPUT_AS_THIRD'] = True
        else:
            raise
        ####### ConfFPT
        if configs_ro_run.loc[i]['ConfFPT'] == False:
            different_params[i]['name'].append(('_ConfFPTNo_', '_ConfFPTNo_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_IN_FPT'] = False
        elif configs_ro_run.loc[i]['ConfFPT'] == True:
            different_params[i]['name'].append(('_ConfFPTNo_', '_ConfFPT_'))
            different_params[i]['modifications']['NETWORK.TRANSFORMER_CONFIDENCE_IN_FPT'] = True
        else:
            raise
        ###### 
        try:
            if configs_ro_run.loc[i]['RegHead'] == 'Shallow':
                different_params[i]['name'].append(('_RHShallow', '_RHShallow'))
                different_params[i]['modifications']['NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP'] = False
            elif configs_ro_run.loc[i]['RegHead'] == 'Deep':
                different_params[i]['name'].append(('_RHShallow', '_RHDeep'))
                different_params[i]['modifications']['NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP'] = True
            elif configs_ro_run.loc[i]['RegHead'] == 'Kadkhod':
                different_params[i]['name'].append(('_RHShallow', '_RHKadkhod'))
                different_params[i]['modifications']['NETWORK.TRANSFORMER_OUTPUT_HEAD_DEEP'] = False
                different_params[i]['modifications']['NETWORK.TRANSFORMER_OUTPUT_HEAD_KADKHOD'] = True
                different_params[i]['modifications']['LOSS.TYPE'] = 'MPJPE_KADKHODA'
            else:
                raise
        except KeyError:
            pass
    return different_params
        
############## to change ########################################

config_example = '/home/ucl/elen/abolfazl/OpenMPL/MPL/configs/cmu_panoptic/mpl_amass/cmu_0_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0.yaml'

config_example = '/home/ucl/elen/abolfazl/OpenMPL/MPL/configs/h36m/mpl_amass/hm_0_amass_mmpose_hrnet_MultiSPT_FPT_Conf3rd_ConfFPTNo_Raytoken_Learn3dEnc_Add3dEncRays_2views_Seed0.yaml'


list_configurations_to_run = [
    # ['Yes',     'Multi',    True,   True,   True,   'Add',      False],
    # ['Yes',     'Multi',    True,   True,   True,   'Weight',   False],
    # ['Yes',     'Multi',    True,   True,   True,   'Mult',     False],
    # ['Yes',     'Multi',    True,   True,   True,   'Concat',   False],
    # ['Yes',     'Multi',    True,   True,   True,   'No',       False],
    # ['Yes',     'Single',   True,   True,   True,   '3rd',      False],
    # ['Yes',     'No',       True,   True,   True,   '3rd',      False],
    # ['Yes',     'Multi',    True,   True,   False,  '3rd',      False],
    # ['Yes',     'Multi',    False,  True,   False,  '3rd',      False],
    # ['Yes',     'Multi',    True,   False,  True,   '3rd',      False],
    # ['Yes',     'Multi',    True,   True,   True,   '3rd',      False],
    # ['Yes',     'Multi',    True,   True,   True,   '3rd',      True ],
    # ['Yes',     'Multi',    True,   True,   True,   'No',       True ],
    # ['No',      'Multi',    True,   True,   True,   '3rd',      False],
    # ['No',      'Single',   True,   True,   True,   '3rd',      False],
    # ['No',      'No',       True,   True,   True,   '3rd',      False],
    
    # ['No',      'No',       False,  True,   False,  'No',       False],   # no conf input, no ray token
    # ['No',      'No',       True,   True,   True,   'No',       False],    # no conf input, with ray token
    
    ########## impact of confidence input
    # ['No',      'No',       False,  True,   False,  '3rd',      False],   
    # ['No',      'Single',   False,  True,   False,  'No',       False],
    # ['No',      'Single',   False,  True,   False,  '3rd',      False],
    ['Yes',     'Single',   False,  True,   False,  'No',       False],         ## chosen config
    # ['Yes',     'Single',   False,  True,   False,  '3rd',      False],         ## 2nd chosen config
       
    ########## impact of ray token
    # ['No',      'Single',   True,   True,   True,   'No',       False],
    # ['Yes',     'Single',   True,   True,   True,   'No',       False],    
    
    ######### with RegHead
    # ['Yes',     'Single',   False,  True,   False,  'No',       False,  'Deep'],         
    # ['No',      'Single',   False,  True,   False,  'No',       False,  'Deep'],         
    # ['No',      'No',       False,  True,   False,  'No',       False,  'Deep'],         
    
    ######### with Kadkhod RegHead
    # ['No',      'No',       False,  True,   False,  'No',       False,  'Kadkhod'],         
    # ['No',      'No',       False,  True,   False,  '3rd',       False,  'Kadkhod'],         
    # ['No',      'Single',       False,  True,   False,  'No',       False,  'Kadkhod'],         
    # ['Yes',      'Single',       False,  True,   False,  'No',       False,  'Kadkhod'],         
    
]

views = [3, 6, 12, 13, 23]
# views = [1, 2, 3, 4]
n_views_list = list(range(1, len(views)+1))

# views = list(range(31))
# views.remove(20)
# views.remove(21)
# n_views_list = [2]

# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# seeds = [0, 1, 2]
seeds = [0]
##############################################################

columns = ['FPT', 'SPT', 'Raytoken', 'Learn3dEnc', 'Add3dEncRays', 'Conf', 'ConfFPT']
if '_RH' in config_example:
    columns.append('RegHead')
configurations_to_run = pd.DataFrame(columns=columns)

for i in range(len(list_configurations_to_run)):
    configurations_to_run.loc[len(configurations_to_run)] = list_configurations_to_run[i]

different_params = make_different_parameters(configurations_to_run)

bash_script = [
    '#!/bin/bash',
    '#SBATCH --job-name=ppt',
    '#SBATCH -c 4',
    '#SBATCH -p gpu',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --time=2-00:00:00',
    '#SBATCH --mem=150G',
    '#SBATCH --qos=preemptible',
]

dir_path = config_example.split('/')[:-1]
dir_path = '/'.join(dir_path)


conf_id_main = int(config_example.split('/')[-1].split('_')[1])
list_configs = os.listdir(dir_path)
list_configs = [x for x in list_configs if x.endswith('.yaml')]
conf_id_highest = 0
for conf in list_configs:
    try:
        conf_id = int(conf.split('_')[1])
    except:
        pass
    if conf_id > conf_id_highest:
        conf_id_highest = conf_id

# find the working directory
folds = os.listdir('/home/ucl/elen/abolfazl/PPT/multi-view-PPT/sbatch_runs')
folds = [x for x in folds if 'sbatch_runs' in x]
sbatch_folder_id = 1
if len(folds) > 0:
    folds = [int(x.split('_')[-1]) for x in folds]
    sbatch_folder_id = max(folds) + 1
    
os.makedirs('sbatch_runs/sbatch_runs_{}'.format(sbatch_folder_id), exist_ok=True)

# open the config example
with open(config_example, 'r') as stream:
    config_dict_org = yaml.safe_load(stream)

to_run = 'python run/pose2d/train_pose_former.py --cfg {} --gpus 0 &'.format(config_example)

counter = 1
sbatch_run_counter = 1
start_id = conf_id_highest
# start_id = 1172

for j in tqdm(range(len(seeds))):
    for i in n_views_list:
        # if i > 1:
        #     break
        combinations_list = generate_combinations(views, i)
        for views_comb in combinations_list:
            for ch in different_params:
                config_dict = copy.deepcopy(config_dict_org)
                new_config_id = conf_id_main + counter + start_id
                
                prefix = config_example.split('/')[-1].split('_')[0]
                new_config_path = config_example.replace('{}_{}'.format(prefix, conf_id_main), '{}_{}'.format(prefix, new_config_id))
                # new_config_path = config_example.replace('hm_{}'.format(conf_id), 'hm_{}'.format(new_config_id))
                views_comb_str = ''
                for v in views_comb:
                    views_comb_str += 'V{}'.format(v)
                new_config_path = new_config_path.replace('2views', '{}views{}'.format(i, views_comb_str))
                new_config_path = new_config_path.replace('Seed0', 'Seed{}'.format(seeds[j]))
                config_dict['DATASET']['TRAIN_VIEWS'] = list(views_comb)
                config_dict['DATASET']['TEST_VIEWS'] = list(views_comb)
                config_dict['SEED'] = seeds[j]
                for name_changes in ch['name']:
                    new_config_path = new_config_path.replace(name_changes[0], name_changes[1])
                for key, value in ch['modifications'].items():
                    key_list = key.split('.')
                    if len(key_list) == 1:
                        config_dict[key_list[0]] = value
                    elif len(key_list) == 2:
                        config_dict[key_list[0]][key_list[1]] = value
                    else:
                        raise
                    
                # save the new config
                with open(new_config_path, 'w') as file:
                    documents = yaml.dump(config_dict, file)
                
                to_run = 'python run/pose2d/train_pose_former.py --cfg {} --gpus 0 &'.format(new_config_path)
                sbatched = False
                
                # sbatch_txt = copy.deepcopy(bash_script)
                # if i == 5:
                #     sbatch_txt[6] = '#SBATCH --mem=30G'
                # elif i == 4:
                #     sbatch_txt[6] = '#SBATCH --mem=15G'
                # elif i == 3:
                #     sbatch_txt[6] = '#SBATCH --mem=10G'
                # else:
                #     sbatch_txt[6] = '#SBATCH --mem=8G'
                    
                # sbatch_txt.append(to_run)
                # sbatch_txt.append('wait')
                # with open('sbatch_runs/sbatch_{}.sh'.format(counter), 'w') as file:
                #     for line in sbatch_txt:
                #         file.write(line + '\n')
                
                # if counter % 2 == 1:
                #     sbatch_txt = copy.deepcopy(bash_script)
                #     sbatch_txt[6] = '#SBATCH --mem=150G'
                    
                #     # if i == 5:
                #     #     sbatch_txt[6] = '#SBATCH --mem=30G'
                #     # elif i == 4:
                #     #     sbatch_txt[6] = '#SBATCH --mem=25G'
                #     # elif i == 3:
                #     #     sbatch_txt[6] = '#SBATCH --mem=20G'
                #     # elif i == 2:
                #     #     sbatch_txt[6] = '#SBATCH --mem=15G'
                #     # else:
                #     #     sbatch_txt[6] = '#SBATCH --mem=10G'
                #     sbatch_txt.append(to_run)
                # else:
                #     sbatch_txt.append(to_run)
                #     sbatch_txt.append('wait')
                #     with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, counter//2), 'w') as file:
                #         for line in sbatch_txt:
                #             file.write(line + '\n')
                #     sbatch_txt = []
                        
                
                
                if counter % 3 == 1:
                    sbatch_txt = copy.deepcopy(bash_script)
                    # sbatch_txt[6] = '#SBATCH --mem=150G'
                    if i == 5:
                        sbatch_txt[6] = '#SBATCH --mem=45G'
                        # sbatch_txt[6] = '#SBATCH --mem=100G'
                    elif i == 4:
                        sbatch_txt[6] = '#SBATCH --mem=40G'
                        # sbatch_txt[6] = '#SBATCH --mem=80G'
                    elif i == 3:
                        sbatch_txt[6] = '#SBATCH --mem=35G'
                        # sbatch_txt[6] = '#SBATCH --mem=80G'
                    elif i == 2:
                        sbatch_txt[6] = '#SBATCH --mem=30G'
                        # sbatch_txt[6] = '#SBATCH --mem=70G'
                    else:
                        sbatch_txt[6] = '#SBATCH --mem=25G'
                        # sbatch_txt[6] = '#SBATCH --mem=60G'
                    sbatch_txt.append(to_run)
                elif counter % 3 == 2:
                    sbatch_txt.append(to_run)
                else:
                    sbatch_txt.append(to_run)
                    sbatch_txt.append('wait')
                    # with open('sbatch_runs/sbatch_{}.sh'.format(counter//3), 'w') as file:
                    with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:   
                        for line in sbatch_txt:
                            file.write(line + '\n')
                    sbatch_txt = []
                    sbatched = True
                    sbatch_run_counter += 1
                    
                
                if not sbatched and j == len(seeds)-1 and i == n_views_list[-1] and views_comb == combinations_list[-1] and ch == different_params[-1]:
                    with open('sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, sbatch_run_counter), 'w') as file:
                        for line in sbatch_txt:
                            file.write(line + '\n')
                            
                    sbatch_run_counter += 1
                    
                # print(to_run)
                counter += 1
            
# run sbatch files
# for i in range(1, counter//3):
#     os.system('sbatch sbatch_runs/sbatch_{}.sh'.format(i))

for i in range(1, sbatch_run_counter+1):
    # if i == 500:
    #     break
    print('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))
    os.system('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))

# for i in range(1, counter//2):
#     # if i == 500:
#     #     break
#     print('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))
#     os.system('sbatch sbatch_runs/sbatch_runs_{}/sbatch_{}.sh'.format(sbatch_folder_id, i))

# for i in range(1, counter):
#     os.system('sbatch sbatch_runs/sbatch_{}.sh'.format(i))


# for i in range(1, 136):
#     print('sbatch sbatch_runs/sbatch_runs_35/sbatch_{}.sh'.format(i))
#     os.system('sbatch sbatch_runs/sbatch_runs_35/sbatch_{}.sh'.format(i))
    