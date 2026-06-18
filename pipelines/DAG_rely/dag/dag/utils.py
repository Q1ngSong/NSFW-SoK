import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .hooker import forward

def get_top_mask(map, thres_percentage):
    """
    map: [TC, B, D, H, W]
    return: [TC, B, D] 
    """
    
    if map.dtype == torch.float32:
        # om: [tbdhw]. thres for [h,w]
        # [tbdhw] -> [tbd(h*w)]
        thres = torch.quantile(
            torch.abs(map).flatten(start_dim=-2),
            thres_percentage,
            dim=-1,
            keepdim=False,
        )
    else:
        thres = torch.quantile(
            torch.abs(map).flatten(start_dim=-2).to(torch.float32),
            thres_percentage,
            dim=-1,
            keepdim=False,
        ).to(map.dtype)
    
    return thres


def ca_hook_args(hooker, hook_map='unconditional'):
    
    for _cah in hooker.cross_attention_hookers:
        if hook_map == 'unconditional':
            _cah.store_unconditional_hidden_states = True
            _cah.store_conditional_hidden_states = False
            _cah.store_safety_hidden_states = False
        elif hook_map == 'conditional':
            _cah.store_unconditional_hidden_states = False
            _cah.store_conditional_hidden_states = True
            _cah.store_safety_hidden_states = False
        elif hook_map =='safety':
            _cah.store_unconditional_hidden_states = False
            _cah.store_conditional_hidden_states = False
            _cah.store_safety_hidden_states = True
            
            
    for _cah in hooker.cross_attention_hookers:
        print('uncon:',_cah.store_unconditional_hidden_states,
                'cond:', _cah.store_conditional_hidden_states,
                'safety:', _cah.store_safety_hidden_states,
                )
        break

def sa_replace_args(hooker, enable_safety_guidance,
                    rules = {}):
    '''
    features candidates: [320, 640, 1280]
    name candidates: [up, mid, down]
    '''
    if not hooker.self_attention_hookers:
        print('no sa hookers')
    for sah in hooker.self_attention_hookers:
       
        rules_tuple = [(place_, rules[place_]) for place_ in rules]
        
        if any([ _[0] in sah.name and sah.module.to_k.in_features in _[1] for _ in rules_tuple]): 
            # print('SA replace: ',sah.name)
            if 'mid' in sah.name:
                print('SA-mid hooked', sah.name, 'feature',sah.module.to_k.in_features)
            else: 
                print('SA hooked', sah.name,  'feature',sah.module.to_k.in_features)
                
                
            sah.save_sam = False 
            sah.replace_sa = True
            
            if enable_safety_guidance:
                sah.chunk_num=3
            else:
                sah.chunk_num=2

        else:
            sah.save_sam = False

def temp_store_hidden_states(ca_hook, clear=False,dim_range=1024, time_interval=1) -> None:
    self = ca_hook
    """Stores the hidden states in the parent trace"""
    if not self._current_hidden_state:
        # print("no hooked result")
        return
    # print(ca_hook.name, '#map:', len(self._current_hidden_state))

    queries = []  # This loop can be vectorized, but has a small impact
    # Thus it is not executed during the optimization process
    for c in self._current_hidden_state:
        query = self.module.to_q(c.unsqueeze(0))
        query = self.module.head_to_batch_dim(query)
        queries.append(query)

    # n_epochs (timestep) x heads x inner_dim x (latent_size = 64)
    current_hidden_states = torch.stack(queries)
    
    if current_hidden_states.shape[2] <= dim_range:
        
        _cur_hs = current_hidden_states[-time_interval:]
        
        # if 'attentions-1-transformer-1' in ca_hook.name:
        #     print(ca_hook.name, 'cur hidden state:',_cur_hs.shape)
        self.hidden_states.store(_cur_hs)
    else:
        pass
    
    ### self.hidden_states:  base.attention_storage.OnlineAttentionStorage
    # -> store: append to the list

    # DO not clear.
    if clear:
        self._current_hidden_state = []  # Clear the current hidden states
    

def empty_store_hidden_states(ca_hook) -> None:
    
    self = ca_hook
    """Stores the hidden states in the parent trace"""
    if not self._current_hidden_state:
        return

    queries = []  # This loop can be vectorized, but has a small impact
    # Thus it is not executed during the optimization process
    for c in self._current_hidden_state:
        query = self.module.to_q(c.unsqueeze(0))
        query = self.module.head_to_batch_dim(query)
        
        ### empty!
        queries.append(torch.zeros_like(query))
        break

    # n_epochs x heads x inner_dim x (latent_size = 64)
    current_hidden_states = torch.stack(queries)
    self.hidden_states.store(current_hidden_states)
    
    ### self.hidden_states:  base.attention_storage.OnlineAttentionStorage
    # -> store: append to the list
    

    # DO not clear.
    # self._current_hidden_state = []  # Clear the current hidden states
    
def clear_stored_hidden_state(ca_hook):
    
    self = ca_hook
    
    ### self.hidden_states:  base.attention_storage.OnlineAttentionStorage
    # -> store: append to the list
    # -> clear: clear list
    self.hidden_states.clear()
    

def decode_image_from_l(pipe, l):
    # print(l.shape)
    l = 1 / 0.18215 * l
    image = pipe.vae.decode(l).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = pipe.numpy_to_pil(image)
    return image

def check_ca_hook_state(hooker):
    # debug
    for _cah in hooker.cross_attention_hookers:
        # drop unexpected
        for hs_ in _cah.hidden_states:
            print(_cah, hs_.shape)
    
    
def get_ratio_with_upper_lower(cam_c, UPPER_THRES, LOW_THRES):
    """
    cam_c: (bs, tc, h, w )
        return  (bs, tc)
    cam_c: (n, h, w)
        return (n,)
    """
    
    concept_region = (cam_c / cam_c.max() > UPPER_THRES).int() 
    mask_region = (cam_c / cam_c.max() > LOW_THRES).int()

    # TBHW -> TB
    concept_region_area = concept_region.flatten(start_dim=-2).sum(-1) 
    mask_area = mask_region.flatten(start_dim=-2).sum(-1) 
    
    # Ratio of area
    return concept_region_area.float()/mask_area.float() 


def set_cam(pipe, hooker, safety_embedding,scale, CAM_TIME_INTERVAL=1):
    
    for ca_hok in hooker.cross_attention_hookers:
        temp_store_hidden_states(ca_hok, 
                                 dim_range=1024,
                                 clear=False, time_interval=CAM_TIME_INTERVAL) # 1 
        
            
    data_type=pipe.unet.dtype
    ovam_evaluator = hooker.get_ovam_callable(expand_size=scale)
    # optimized_map = ovam_evaluator(embedding.to(data_type)).squeeze().cpu().numpy()[1] # (512, 512)
    
    # (bs, Tok, 512, 512)
    if type(safety_embedding) is not str:
        safety_embedding = safety_embedding.to(data_type)
        optimized_map = forward(ovam_evaluator, safety_embedding)[:,1:].mean(1) # bs, h, w
    else:
        optimized_map = forward(ovam_evaluator, safety_embedding)[:,1:-1].mean(1) # bs, h, w
        
    for ca_hok in hooker.cross_attention_hookers:
        clear_stored_hidden_state(ca_hok)
        
    return optimized_map 
            
            
# @torch.no_grad()
# def set_info_dict_tbcw(pipe, hooker, curr_info, safety_embeddings, expand_size=64, r_thres_list= [1e-5, 0.01, 0.05, 0.1, 0.3, 0.5],
#                   cam_c = None, CAM_TIME_INTERVAL = 1, MODE_APEAR_UP=0.8, CON_APEAR_LOW = 0.1, CC_DELAY=1):
#     if curr_info is None:
#         curr_info = {
#             'i': [],
#             'mode_dynamic (0.01)': [],
#             'concept_dynamic (0.5)': [],
            
#             'mode_appear': [],  # metric1 = dynamic 0.01 < 85%/80%
#             'concept_appear': [], # metric2 = metric1 and (dynamic 0.5 > 1%) 
#             'concept_consistency': [],# metric3 = $is_consistent(metric2) > 0.5
            
#             # cnters for metric3
#             '_concept_1_consistency_cnter': {'max_length': None, 'curr_length': None },  # appear consistency
#             '_concept_0_consistency_cnter': {'max_length': None, 'curr_length': None },  # disappear consistency
#         }
#     if cam_c is None:
#         cam_c = torch.stack(
#                 [set_cam(pipe, hooker, safety_embedding, 
#                         CAM_TIME_INTERVAL=CAM_TIME_INTERVAL,
#                         scale=(expand_size, expand_size))
#                 for safety_embedding in safety_embeddings])
        
#         cam_c = cam_c.flatten(start_dim=-2)
#         cam_c = cam_c/cam_c.max(-1)[0] 
#         # cam_c = cam_c.view(bs_n * concept_num, expand_size, expand_size)
#         cam_c = cam_c.view(-1, len(safety_embeddings), expand_size, expand_size)
    
#     ratio_info  = {}
#     for idx, r_thres_ in enumerate(r_thres_list):

#         ratio_ = get_ratio_with_upper_lower(cam_c,
#                                     UPPER_THRES=r_thres_, 
#                                     LOW_THRES=-1,#all
#                                     )
        
#         ratio_info['{:.2f}'.format(r_thres_)] = ratio_
#     # raw data    
#     curr_i = len(curr_info["i"]) # if i is None
#     curr_info["i"].append(curr_i) # 0,1,...49
    
#     ratio_001 = ratio_info['0.01']
#     ratio_05 = ratio_info['0.50']
#     curr_info["mode_dynamic (0.01)"].append(ratio_001) # (bs, tc)
#     curr_info["concept_dynamic (0.5)"].append(ratio_05) # (bs, tc)
    
#     # metrics1 : using value-percentage threshold to test
#     # when majority (at least 80%) lies in range larger than value 0.01
#     #  (mask is not nearly all-zero),
#     #   represents the appearance of a distinguishable object compared to the background.
#     mode_appear = ratio_001 < MODE_APEAR_UP
#     curr_info["mode_appear"].append(mode_appear.int()) # (bs, tc)
        
    
#     # metrics2 : using value-percentage threshold to test
#     # when a certain part (no less than 1%) lies in range larger than 0.5 
#     #  (mask gives a high-activated region that large enough),
#     concept_appear = mode_appear.int() * (ratio_05 > CON_APEAR_LOW).int()
#     curr_info["concept_appear"].append(concept_appear) # (bs, tc)
    
#     # metric2: consistency
#     # use zero tensor to initilize cnter
#     if curr_info['_concept_1_consistency_cnter']['max_length'] is None:
#         for _ in curr_info['_concept_1_consistency_cnter']:
#             curr_info['_concept_1_consistency_cnter'][_] = torch.zeros_like(concept_appear).int() # bs, tc
#         for _ in curr_info['_concept_0_consistency_cnter']:
#             curr_info['_concept_0_consistency_cnter'][_] = torch.zeros_like(concept_appear).int() # bs, tc
    
#     # if concept_appear == 1:
#     #     curr_info['_concept_1_consistency_cnter']['curr_length'] += 1
#     #     curr_info['_concept_1_consistency_cnter']['max_length'] = max(curr_info['_concept_1_consistency_cnter']['max_length'],
#     #                                                                   curr_info['_concept_1_consistency_cnter']['curr_length'] )
#     # else:
#     #     curr_info['_concept_1_consistency_cnter']['curr_length'] = 0
#     for b_i in range(concept_appear.shape[0]):
#         for c_i in range(concept_appear.shape[1]):
#             if concept_appear[b_i][c_i] == 1:
#                 curr_info['_concept_1_consistency_cnter']['curr_length'][b_i][c_i] += 1
#                 curr_info['_concept_1_consistency_cnter']['max_length'][b_i][c_i] = max(
#                     curr_info['_concept_1_consistency_cnter']['max_length'][b_i][c_i],
#                     curr_info['_concept_1_consistency_cnter']['curr_length'][b_i][c_i])
                
#                 curr_info['_concept_0_consistency_cnter']['curr_length'][b_i][c_i] = 0
                
                
#             elif curr_info['_concept_1_consistency_cnter']['max_length'][b_i][c_i] > 0: # only after appear
                
#                 curr_info['_concept_1_consistency_cnter']['curr_length'][b_i][c_i] = 0
                
                
#                 curr_info['_concept_0_consistency_cnter']['curr_length'][b_i][c_i] += 1
#                 curr_info['_concept_0_consistency_cnter']['max_length'][b_i][c_i] = max(
#                     curr_info['_concept_0_consistency_cnter']['max_length'][b_i][c_i],
#                     curr_info['_concept_0_consistency_cnter']['curr_length'][b_i][c_i])
                
        
#     all_sequence = torch.stack(curr_info["concept_appear"]).int() # i, bs, tc
    
    
#     concept_consistency_1 = torch.zeros((all_sequence.shape[-2],all_sequence.shape[-1]))# (bs,tc)
#     concept_consistency_0 = torch.ones((all_sequence.shape[-2],all_sequence.shape[-1])) # (bs,tc)
    
    
#     curr_i = curr_i + 1
#     for b_i in range(concept_appear.shape[0]):
#         for c_i in range(concept_appear.shape[1]):
#             sample_i_list = all_sequence[:, b_i, c_i].flatten().tolist()
#             # print(sample_i_list, curr_info['_concept_1_consistency_cnter'])
#             assert len(sample_i_list) == curr_i # iter1, [0]; iter3: [0,0,1]; iter5 [0,0,1,0,0,1]
            
#             if 1 in sample_i_list:
#                 first_1_at = sample_i_list.index(1) # e.g. [0,0,0,...  until iter10=1, .... , curr]
                
#                 # curri = 3, first @ 2. [1]
#                 # curri = 2, first @ 2. [1,0,0,1]
                
#                 sample_i_list = sample_i_list[first_1_at:] # [1,,....,curr] concept appear sequence
#                 assert len(sample_i_list) >0, sample_i_list # at least [1]
                
#                 # effecive = 3-2=1
#                 effective_len = len(sample_i_list)
#                 # print("first 1", first_1_at, 'effective_len', effective_len)
                
#                 if effective_len<=CC_DELAY: 
#                     # cases when [1], meaning concept appear at the first time
#                     # .. with a conservative strategy, we need to look next several steps to determine its consistency.
#                     continue
#                 else:
#                     # effective_len >= 3,  e.g., [1,0,0,1]
                    
#                     ##############################
#                     ### appear consistent test ###
#                     ##############################
#                     max_cons_len_1 = curr_info['_concept_1_consistency_cnter']['max_length'][b_i][c_i]
#                     # max_cons_len: [1,1] -> 2; [1,0] -> 1
#                     test_value_1 = 1.*max_cons_len_1/effective_len
#                     # print("max con 1", max_cons_len_1, 'value_con_1', test_value_1)
#                     # test value: [1,1] -> 1; [1,0] -> 0.5
#                     # we need test value > 0.5 , to be a consistent concept of interest.
#                     concept_consistency_1[b_i][c_i] = test_value_1
                    
                    
#                     ##############################
#                     ## disappear consistent test #
#                     ##############################
#                     max_cons_len_0 = curr_info['_concept_0_consistency_cnter']['max_length'][b_i][c_i]
#                     # max_cons_len: [1,1] -> 2; [1,0] -> 1
#                     test_value_0 = (effective_len-max_cons_len_0)/effective_len
#                     # print("max con 0", max_cons_len_0, 'value_con_0', test_value_0)
#                     # test value: [1,1] -> 1; [1,0] -> 0.5
#                     # we need test value > 0.5 , to be a consistent concept of interest.
#                     concept_consistency_0[b_i][c_i] = test_value_0
                                    
#             else:
#                 # high noise level . No mode or concepts detected.
#                 continue
            
        
#     # metric3 = $is_consistent(metric2) > 0.5
#     #  - testA: is appear consistent? : concept_consistency_1
#     #  - testB: disappear consistent? : concept_consistency_0
#     curr_info["concept_consistency"].append(concept_consistency_0 * concept_consistency_1) # (bs, tc)*(bs, tc)->(bs, tc)
    
#     return curr_info
            
@torch.no_grad()
def set_info_dict(pipe, hooker, curr_info, safety_embeddings, expand_size=64, r_thres_list= [1e-5, 0.01, 0.05, 0.1, 0.3, 0.5],
                  cam_c = None, CAM_TIME_INTERVAL = 1, MODE_APEAR_UP=0.8, CON_APEAR_LOW = 0.01, CC_DELAY=1):
    if curr_info is None:
        curr_info = {
            'i': [],
            'mode_dynamic (0.01)': [],
            'concept_dynamic (0.5)': [],
            
            'mode_appear': [],  # metric1 = dynamic 0.01 < 85%/80%
            'concept_appear': [], # metric2 = metric1 and (dynamic 0.5 > 1%) 
            'concept_consistency': [],# metric3 = $is_consistent(metric2) > 0.5
            
            # cnters for metric3
            '_concept_1_consistency_cnter': {'max_length': None, 'curr_length': None },  # appear consistency
            '_concept_0_consistency_cnter': {'max_length': None, 'curr_length': None },  # disappear consistency
        }
    if cam_c is None:
        cam_c = torch.stack(
                [set_cam(pipe, hooker, safety_embedding, 
                        CAM_TIME_INTERVAL=CAM_TIME_INTERVAL,
                        scale=(expand_size, expand_size))
                for safety_embedding in safety_embeddings])
        
        cam_c = cam_c.flatten(start_dim=-2)
        cam_c = cam_c/cam_c.max(-1)[0] 
        # cam_c = cam_c.view(bs_n * concept_num, expand_size, expand_size)
        cam_c = cam_c.view(-1, len(safety_embeddings), expand_size, expand_size)
    
    ratio_info  = {}
    for idx, r_thres_ in enumerate(r_thres_list):

        ratio_ = get_ratio_with_upper_lower(cam_c,
                                    UPPER_THRES=r_thres_, 
                                    LOW_THRES=-1,#all
                                    )
        
        ratio_info['{:.2f}'.format(r_thres_)] = ratio_
    # raw data    
    curr_i = len(curr_info["i"]) # if i is None
    curr_info["i"].append(curr_i) # 0,1,...49
    
    ratio_001 = ratio_info['0.01']
    ratio_05 = ratio_info['0.50']
    curr_info["mode_dynamic (0.01)"].append(ratio_001) # (bs, tc)
    curr_info["concept_dynamic (0.5)"].append(ratio_05) # (bs, tc)
    
    # metrics1 : using value-percentage threshold to test
    # when majority (at least 80%) lies in range larger than value 0.01
    #  (mask is not nearly all-zero),
    #   represents the appearance of a distinguishable object compared to the background.
    mode_appear = ratio_001 < MODE_APEAR_UP
    curr_info["mode_appear"].append(mode_appear.int()) # (bs, tc)
        
    
    # metrics2 : using value-percentage threshold to test
    # when a certain part (no less than 1%) lies in range larger than 0.5 
    #  (mask gives a high-activated region that large enough),
    concept_appear = mode_appear.int() * (ratio_05 > CON_APEAR_LOW).int()
    curr_info["concept_appear"].append(concept_appear) # (bs, tc)
    
    # metric2: consistency
    # use zero tensor to initilize cnter
    if curr_info['_concept_1_consistency_cnter']['max_length'] is None:
        for _ in curr_info['_concept_1_consistency_cnter']:
            curr_info['_concept_1_consistency_cnter'][_] = torch.zeros_like(concept_appear).int() # bs, tc
        for _ in curr_info['_concept_0_consistency_cnter']:
            curr_info['_concept_0_consistency_cnter'][_] = torch.zeros_like(concept_appear).int() # bs, tc
    
    # if concept_appear == 1:
    #     curr_info['_concept_1_consistency_cnter']['curr_length'] += 1
    #     curr_info['_concept_1_consistency_cnter']['max_length'] = max(curr_info['_concept_1_consistency_cnter']['max_length'],
    #                                                                   curr_info['_concept_1_consistency_cnter']['curr_length'] )
    # else:
    #     curr_info['_concept_1_consistency_cnter']['curr_length'] = 0
    for b_i in range(concept_appear.shape[0]):
        if concept_appear[b_i] == 1:
            curr_info['_concept_1_consistency_cnter']['curr_length'][b_i] += 1
            curr_info['_concept_1_consistency_cnter']['max_length'][b_i] = max(
                curr_info['_concept_1_consistency_cnter']['max_length'][b_i],
                curr_info['_concept_1_consistency_cnter']['curr_length'][b_i])
            
            curr_info['_concept_0_consistency_cnter']['curr_length'][b_i] = 0
            
            
        elif curr_info['_concept_1_consistency_cnter']['max_length'][b_i] > 0: # only after appear
            
            curr_info['_concept_1_consistency_cnter']['curr_length'][b_i] = 0
            
            
            curr_info['_concept_0_consistency_cnter']['curr_length'][b_i] += 1
            curr_info['_concept_0_consistency_cnter']['max_length'][b_i] = max(
                curr_info['_concept_0_consistency_cnter']['max_length'][b_i],
                curr_info['_concept_0_consistency_cnter']['curr_length'][b_i])
            
    
    all_sequence = torch.stack(curr_info["concept_appear"]).int() # i, bs
    
    
    concept_consistency_1 = torch.zeros((all_sequence.shape[-1]))# (bs,)
    concept_consistency_0 = torch.ones((all_sequence.shape[-1])) # (bs,)
    
    
    curr_i = curr_i + 1
    for b_i in range(concept_appear.shape[0]):
        # for c_i in range(concept_appear.shape[1]):
            sample_i_list = all_sequence[:, b_i].flatten().tolist()
            # print(sample_i_list, curr_info['_concept_1_consistency_cnter'])
            assert len(sample_i_list) == curr_i # iter1, [0]; iter3: [0,0,1]; iter5 [0,0,1,0,0,1]
            
            if 1 in sample_i_list:
                first_1_at = sample_i_list.index(1) # e.g. [0,0,0,...  until iter10=1, .... , curr]
                
                # curri = 3, first @ 2. [1]
                # curri = 2, first @ 2. [1,0,0,1]
                
                sample_i_list = sample_i_list[first_1_at:] # [1,,....,curr] concept appear sequence
                assert len(sample_i_list) >0, sample_i_list # at least [1]
                
                # effecive = 3-2=1
                effective_len = len(sample_i_list)
                # print("first 1", first_1_at, 'effective_len', effective_len)
                
                if effective_len<=CC_DELAY: 
                    # cases when [1], meaning concept appear at the first time
                    # .. with a conservative strategy, we need to look next several steps to determine its consistency.
                    continue
                else:
                    # effective_len >= 3,  e.g., [1,0,0,1]
                    
                    ##############################
                    ### appear consistent test ###
                    ##############################
                    max_cons_len_1 = curr_info['_concept_1_consistency_cnter']['max_length'][b_i]
                    # max_cons_len: [1,1] -> 2; [1,0] -> 1
                    test_value_1 = 1.*max_cons_len_1/effective_len
                    # print("max con 1", max_cons_len_1, 'value_con_1', test_value_1)
                    # test value: [1,1] -> 1; [1,0] -> 0.5
                    # we need test value > 0.5 , to be a consistent concept of interest.
                    concept_consistency_1[b_i] = test_value_1
                    
                    
                    ##############################
                    ## disappear consistent test #
                    ##############################
                    max_cons_len_0 = curr_info['_concept_0_consistency_cnter']['max_length'][b_i]
                    # max_cons_len: [1,1] -> 2; [1,0] -> 1
                    test_value_0 = (effective_len-max_cons_len_0)/effective_len
                    # print("max con 0", max_cons_len_0, 'value_con_0', test_value_0)
                    # test value: [1,1] -> 1; [1,0] -> 0.5
                    # we need test value > 0.5 , to be a consistent concept of interest.
                    concept_consistency_0[b_i] = test_value_0
                                    
            else:
                # high noise level . No mode or concepts detected.
                continue
            
        
    # metric3 = $is_consistent(metric2) > 0.5
    #  - testA: is appear consistent? : concept_consistency_1
    #  - testB: disappear consistent? : concept_consistency_0
    curr_info["concept_consistency"].append(concept_consistency_0 * concept_consistency_1) # (bs, tc)*(bs, tc)->(bs, tc)
    
    return curr_info