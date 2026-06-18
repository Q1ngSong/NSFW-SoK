
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
import numpy as np
from .utils import *
from ..stable_diffusion.safe_utils import _encode_prompt

def get_embds(self, prompt, device , num_images_per_prompt,
                do_classifier_free_guidance, negative_prompt, 
                prompt_embeds,negative_prompt_embeds,
                text_encoder_lora_scale, safety_concept ):
        
    return _encode_prompt(
        self, prompt,device,num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt, 
        enable_safety_guidance=safety_concept is not None,
        safety_text_concept = safety_concept
    ) # [uncond, cond, safety] or [uncond, cond]
    
    
def detect_and_guide(
    pipe, i, noise_pred,
    guideline_token_embedding, 
    batch_size,
    num_images_per_prompt,
    ITER_INTERVAL=(5,50), LAMBDA = 0.99,
    
    CAM_SCALER = 5, # default 5 from SEGA
    UP_THRES=0.5, LOW_THRES=0.01,
    
    hooker=None, SLD_GUIDANCE_SCALE=2000,
    CAM_TIME_INTERVAL=1,
    **kargs
):
    if len(kargs)>0:
        print('warning: parameters not used:',kargs)
    
    begin_iter , end_iter = ITER_INTERVAL[0],ITER_INTERVAL[1]
    
    noise_pred_uncond, noise_pred_text, safety_cond_noise_pred = noise_pred.chunk(3)
    
    cond_guidance_step = noise_pred_text - noise_pred_uncond
    safety_cond_guidance_step = safety_cond_noise_pred - noise_pred_uncond
    
    bs_n = batch_size * num_images_per_prompt
    size_dim = safety_cond_guidance_step.shape[-1]
    
    # CAM: detection
    
    if hooker is not None:
        # bs_n, H, W
        cam_c = set_cam(
            pipe, hooker, guideline_token_embedding,
            CAM_TIME_INTERVAL=CAM_TIME_INTERVAL,
            scale=(size_dim,size_dim)
            )

        assert cam_c.shape == (bs_n, size_dim, size_dim)
        cam_c = cam_c.flatten(start_dim=-2)
        cam_c = cam_c/(cam_c.max(-1)[0])
        cam_c = cam_c.view(bs_n, size_dim, size_dim)
        
        
        # cam_momentum_scale, cam_momentum_beta = 0.5, 0.7
        # cam_momentum = getattr(pipe,'cam_momentum', torch.zeros_like(cam_c))
        # cam_c = cam_c + cam_momentum_scale * cam_momentum 
        # cam_momentum = (1 - cam_momentum_beta) * cam_c + cam_momentum_beta * cam_momentum 

        
        curr_info =  getattr(pipe, 'curr_info', None)
        pipe.curr_info = set_info_dict(pipe,  hooker,
                                safety_embeddings = guideline_token_embedding,
                                curr_info=curr_info, cam_c = cam_c,
                                CAM_TIME_INTERVAL = CAM_TIME_INTERVAL, 
                                CC_DELAY=0
                                ) 
        
        # print('adaptive interval',i,(pipe.curr_info["concept_consistency"]))
        # concept_adaptive_interval_scale = pipe.curr_info["concept_consistency"][-1] # 1.0 or 0.0
        
        if i < begin_iter or i >= end_iter:
            concept_adaptive_interval_scale = pipe.curr_info["concept_consistency"][-1]
        else: 
            concept_adaptive_interval_scale = torch.tensor([1.0], device="cuda")# 1.0 or 0.0
        concept_adaptive_interval_scale = concept_adaptive_interval_scale.to(cam_c.device)
        print('adaptive interval scale:',i, concept_adaptive_interval_scale)
        
        
        # b, size, size
        concept_region = (cam_c / cam_c.max() > UP_THRES).int() 
        mask_region = (cam_c / cam_c.max() > LOW_THRES).int()
        
    ## SEGA/SLD
    is_sega = LAMBDA>=0.1
    is_sld = LAMBDA<0.1
    
    if is_sega:
        om = safety_cond_guidance_step # b, c, h, w
        # print('sega: origial map (bchw)', om.shape)
        thres = get_top_mask(om, LAMBDA)[:, :, None, None]
        # larger safety editing amount is chosen.
        scale_om = torch.where(
            torch.abs(om) >= thres, # b, c, 1, 1
            torch.ones_like(om) , 
            torch.zeros_like(om),
        )
        # print('sega: scale map (bchw)', scale_om.shape)
        

    ## SLD
    if is_sld:
        # difference between epsilon^p and epsilon^{c_s}
        om = noise_pred_text - safety_cond_noise_pred
        print('sld: origial map (bchw)', om.shape)

        clamped_om = torch.clamp(om.abs() * SLD_GUIDANCE_SCALE, max=1.0) # ~CAM. sld_guidance_scale: 200, 1000, 2000 -- control the percentage of (1.0)
        
        scale_om = torch.where(
            om >= LAMBDA , 
            torch.zeros_like(clamped_om), 
            clamped_om, 
            # max om is 1.
        ) # mask by zeros
        print('sld: scaled map (bchw)', scale_om.shape)

        pass
        
    # * I. CAM intensity scaler
    # ** only mask region (>= LOWER THRESHOLD) apply safety guidance
    # ** rescale the magnitude of CAM from [0.01,1] to [1, 5]
    
    cam_scale = (cam_c * mask_region).flatten(start_dim=-2) # b, -1
    cam_scale = cam_scale / LOW_THRES # [min=1, max=1/0.01]
    cam_scale = torch.clamp(cam_scale, max=5.) # [1, 5]
    cam_scale = cam_scale.reshape(cam_c.shape)
    
    # * 2. CAM area scaler
    # ** intuition: in the same image, a larger object needs larger scaling.
    # ** area cluculation is based on [concept region].
    
    pixel_wise_base_scale = CAM_SCALER
    if pixel_wise_base_scale < 0.01: 
        area_total = concept_region.flatten(start_dim=-2).int().sum(-1)
        area_total_lower = mask_region.flatten(start_dim=-2).int().sum(-1)
        
        # ** (deprecated) area adaptive (non-spatial)
        # adaptive_scaler = CAM_SCALER * 1.0 * area_total # t,b
        # mask_region_ = mask_region.to(scale_om.dtype) * adaptive_scaler[:,:,None,None] # tbhc*tb
        # mask_region_ = mask_region_ * cam_scale 
        
        # ** following: area adaptive (object-aware)
        adaptive_scaler = torch.zeros_like(concept_region).float() # bhw
        
        for b_s in range(concept_region.shape[0]):
            labeled_mask, _ = label(concept_region[b_s].detach().cpu(), structure=np.ones((3, 3)))
            area_avg = area_total[b_s] / _
            labeled_mask = torch.tensor(labeled_mask)
            scales = torch.zeros((_)) # num_obj
            
            for l_idx, l in enumerate((range(1, labeled_mask.max()+1))):
                obj_l_mask = (labeled_mask == l).to(dtype=adaptive_scaler.dtype, device=adaptive_scaler.device)
                area_div = obj_l_mask.sum() - area_avg + area_total[b_s]# scaler
                adaptive_scaler[b_s] += obj_l_mask * area_div
                scales[l_idx] = area_div

            weights_to_each_cluster = np.stack([
                1/(distance_transform_edt(labeled_mask != l) + 1e-10)
                for l in range(1, labeled_mask.max()+1)
            ],0) 
            weights_to_each_cluster = torch.from_numpy(weights_to_each_cluster)
            # num_obj, H, W
            total_weights = weights_to_each_cluster.sum(0) 
            # H,W
            interpolated_scaler = torch.stack([
                scales[l_idx] * weights_to_each_cluster[l_idx][labeled_mask == 0] # hw
                for l_idx in range(_)
            ],0).sum(0) / total_weights[labeled_mask == 0]
            # H,W
            interpolated_scaler = interpolated_scaler.to(dtype=adaptive_scaler.dtype, device=adaptive_scaler.device)
            # if a large mask region, than need to rescale the scale for larger editing
            interpolated_scaler = interpolated_scaler * torch.clamp((area_total_lower-area_total)[b_s]/area_total[b_s], max=1)
            adaptive_scaler[b_s][labeled_mask == 0] = interpolated_scaler
                # H,W
        adaptive_scaler = CAM_SCALER * adaptive_scaler # s_0 * (bhw)

        # adaptive_scaler = CAM_SCALER * 1.0 * mask_region.flatten(start_dim=-2).int().sum(-1) # t,b
        
        mask_region_ = mask_region.to(scale_om.dtype) * adaptive_scaler # tbhc*tbhc
        mask_region_ = mask_region_ * cam_scale 
    else:
        # e.g.,  Scaler=1
        mask_region_ = mask_region.to(scale_om.dtype) * CAM_SCALER 
        
    scale = torch.einsum('bhw,bchw->bchw', mask_region_, scale_om) 
        
    if  LOW_THRES < 0 or hooker is None:
        # no mask region is applied.
        # SLD/SEGA without mometum
        safety_guidance_final = safety_cond_guidance_step * scale_om
    else:
        # apply CAM mask & 2 scalers
        safety_guidance_final = safety_cond_guidance_step * (scale * concept_adaptive_interval_scale)
    
    # print('cond_guidance_step', cond_guidance_step.shape)
    # print('safety_guidance_final', safety_guidance_final.shape)
    
    safety_momentum_scale, safety_momentum_beta = 0.5, 0.7
    safety_momentum =  getattr(pipe, 'safety_momentum', torch.zeros_like(safety_guidance_final))
    safety_guidance_final = safety_guidance_final + safety_momentum_scale * safety_momentum
    pipe.safety_momentum = safety_momentum_beta * safety_momentum + (1 - safety_momentum_beta) * safety_guidance_final
    

    if i < begin_iter or i >= end_iter:
        return noise_pred_uncond, cond_guidance_step
    return noise_pred_uncond, cond_guidance_step - safety_guidance_final
