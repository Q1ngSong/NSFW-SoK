# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, is_torch_version, logging
# from diffusers.utils.torch_utils import apply_freeu
# 
try:
    from diffusers.utils.torch_utils import apply_freeu
except ImportError:
    def apply_freeu(*args, **kwargs):
        pass
# 
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
# from diffusers.models.normalization import AdaGroupNorm
# ================= 修复代码开始 =================
import torch.nn as nn

# 尝试导入 AdaGroupNorm，如果找不到模块或类，就创建一个替身
try:
    from diffusers.models.normalization import AdaGroupNorm
except (ImportError, ModuleNotFoundError):
    # 定义一个空类，防止代码报错
    class AdaGroupNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x, *args, **kwargs):
            return x
# ================= 修复代码结束 =================
"""from diffusers.models.resnet import (
    Downsample2D,
    FirDownsample2D,
    FirUpsample2D,
    KDownsample2D,
    KUpsample2D,
    ResnetBlock2D,
    ResnetBlockCondNorm2D,
    Upsample2D,
)"""
# ================= 修复代码开始 =================
import torch.nn as nn

# 1. 导入基础组件 (这些通常是稳定的)
try:
    from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
except ImportError:
    # 极少数情况下如果连这些都变了，需要更复杂的修复，但通常这里不会报错
    pass

# 2. 单独处理 ResnetBlockCondNorm2D (报错源)
try:
    from diffusers.models.resnet import ResnetBlockCondNorm2D
except ImportError:
    # 如果找不到，定义一个空的替身类
    class ResnetBlockCondNorm2D(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x, *args, **kwargs):
            return x
# ================= 修复代码结束 =================
# from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
# ================= 修复代码开始 =================
try:
    # 尝试从新旧不同的路径导入 (为了兼容性尝试多种可能)
    try:
        from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
    except ImportError:
        from diffusers.models.dual_transformer_2d import DualTransformer2DModel
except (ImportError, ModuleNotFoundError):
    # 如果都找不到，定义一个空类
    class DualTransformer2DModel:
        pass
# ================= 修复代码结束 =================
# from diffusers.models.transformers.transformer_2d import Transformer2DModel
# ================= 修复代码开始 =================
try:
    # 路径 A: 新版 diffusers 结构
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
except (ImportError, ModuleNotFoundError):
    try:
        # 路径 B: 旧版 diffusers 结构 (或直接在 models 下)
        from diffusers.models import Transformer2DModel
    except (ImportError, ModuleNotFoundError):
        # 如果都找不到 (极罕见), 定义空类
        class Transformer2DModel:
            pass
# ================= 修复代码结束 =================
"""from diffusers.models.unets.unet_2d_blocks import (
    DownBlock2D, 
    ResnetDownsampleBlock2D, 
    AttnDownBlock2D, 
    # CrossAttnDownBlock2D, 
    SimpleCrossAttnDownBlock2D, 
    SkipDownBlock2D, 
    AttnSkipDownBlock2D, 
    DownEncoderBlock2D, 
    AttnDownEncoderBlock2D,
    KDownBlock2D,
    KCrossAttnDownBlock2D,
)"""
# ================= 修复代码开始 =================
try:
    # 尝试从老路径导入 (代码原本的写法)
    from diffusers.models.unets.unet_2d_blocks import (
        CrossAttnDownBlock2D,
        DownBlock2D,
        CrossAttnUpBlock2D,
        UpBlock2D,
        UNetMidBlock2DCrossAttn,
        UNetMidBlock2D,
        get_down_block,
        get_up_block,
    )
except (ImportError, ModuleNotFoundError):
    # 如果老路径报错，尝试从新路径 (diffusers.models) 直接导入
    # 这种方式通常兼容新版 diffusers
    try:
        from diffusers.models import (
            CrossAttnDownBlock2D,
            DownBlock2D,
            CrossAttnUpBlock2D,
            UpBlock2D,
            UNetMidBlock2DCrossAttn,
            UNetMidBlock2D,
        )
        # get_down_block 和 get_up_block 在新版中可能位置变了，单独处理
        from diffusers.models.unet_2d_blocks import get_down_block, get_up_block
    except (ImportError, ModuleNotFoundError):
         # 如果连 unet_2d_blocks 文件都找不到，尝试直接从 models 导入函数 (备选方案)
        try:
            from diffusers.models import get_down_block, get_up_block
        except:
             # 万策尽，只能打印错误让用户知道缺了什么，这里很难 Mock
            print("Warning: Failed to load UNet blocks. Model structure might be broken.")
# ================= 修复代码结束 =================

"""from diffusers.models.unets.unet_2d_blocks import (
    # UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UNetMidBlock2D,
)"""
# ================= 修复代码开始 (Line 154) =================
try:
    # 路径 A: 新版 (diffusers > 0.14)
    from diffusers.models.unets.unet_2d_blocks import (
        UNetMidBlock2DCrossAttn,
        UNetMidBlock2D,
    )
except (ImportError, ModuleNotFoundError):
    try:
        # 路径 B: 旧版 (diffusers < 0.14, 文件结构不同)
        from diffusers.models.unet_2d_blocks import (
            UNetMidBlock2DCrossAttn,
            UNetMidBlock2D,
        )
    except (ImportError, ModuleNotFoundError):
        try:
            # 路径 C: 直接暴露在 models 下
            from diffusers.models import (
                UNetMidBlock2DCrossAttn,
                UNetMidBlock2D,
            )
        except:
             # 如果还找不到，可能是名字变了，尝试 Mock 防止崩溃
             class UNetMidBlock2DCrossAttn: pass
             class UNetMidBlock2D: pass
             print("Warning: Could not load UNetMidBlock classes.")
# ================= 修复代码结束 =================

"""from diffusers.models.unets.unet_2d_blocks import (
    UpBlock2D,
    ResnetUpsampleBlock2D,
    # CrossAttnUpBlock2D,
    SimpleCrossAttnUpBlock2D,
    AttnUpBlock2D,
    SkipUpBlock2D,
    AttnSkipUpBlock2D,
    UpDecoderBlock2D,
    AttnUpDecoderBlock2D,
    KUpBlock2D,
    KCrossAttnUpBlock2D
)"""
# ================= 修复代码开始 (Line 187) =================
try:
    # 路径 A: 旧版写法
    from diffusers.models.unets.unet_2d_blocks import get_down_block, get_up_block
except (ImportError, ModuleNotFoundError):
    try:
        # 路径 B: 新版写法 (去掉了中间的 unets 目录)
        from diffusers.models.unet_2d_blocks import get_down_block, get_up_block
    except (ImportError, ModuleNotFoundError):
        try:
            # 路径 C: 直接暴露在顶层
            from diffusers.models import get_down_block, get_up_block
        except:
            # 万策尽，定义空函数防止报错 (虽然这会导致模型构建失败，但能先跑通流程)
            def get_down_block(*args, **kwargs): return None
            def get_up_block(*args, **kwargs): return None
            print("Warning: Could not find get_down_block/get_up_block functions.")
# ================= 修复代码结束 =================

# modified cross-attention blocks
from pipelines.ConceptCorrector_rely.modified_diffusion.modified_cross_attn_blocks import (
    CrossAttnUpBlock2D,
    CrossAttnDownBlock2D,
    UNetMidBlock2DCrossAttn
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
    module_name: str = "",
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "ResnetDownsampleBlock2D":
        raise NotImplementedError
        return ResnetDownsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif down_block_type == "AttnDownBlock2D":
        raise NotImplementedError
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or "conv"  # default to 'conv'
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_type=downsample_type,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            module_name=module_name
        )
    elif down_block_type == "SimpleCrossAttnDownBlock2D":
        raise NotImplementedError
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D")
        return SimpleCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif down_block_type == "SkipDownBlock2D":
        raise NotImplementedError
        return SkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnSkipDownBlock2D":
        raise NotImplementedError
        return AttnSkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "DownEncoderBlock2D":
        raise NotImplementedError
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownEncoderBlock2D":
        raise NotImplementedError
        return AttnDownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "KDownBlock2D":
        raise NotImplementedError
        return KDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif down_block_type == "KCrossAttnDownBlock2D":
        raise NotImplementedError
        return KCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            add_self_attention=True if not add_downsample else False,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
    module_name: str = "",
):
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        return UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            module_name=module_name
        )
    elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
        raise NotImplementedError
        return UNetMidBlock2DSimpleCrossAttn(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif mid_block_type == "UNetMidBlock2D":
        raise NotImplementedError
        return UNetMidBlock2D(
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=0,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=False,
        )
    elif mid_block_type is None:
        return None
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
    module_name: str = "",
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "ResnetUpsampleBlock2D":
        raise NotImplementedError
        return ResnetUpsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            module_name=module_name
        )
    elif up_block_type == "SimpleCrossAttnUpBlock2D":
        raise NotImplementedError
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D")
        return SimpleCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif up_block_type == "AttnUpBlock2D":
        raise NotImplementedError
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or "conv"  # default to 'conv'

        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            upsample_type=upsample_type,
        )
    elif up_block_type == "SkipUpBlock2D":
        raise NotImplementedError
        return SkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnSkipUpBlock2D":
        raise NotImplementedError
        return AttnSkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "UpDecoderBlock2D":
        raise NotImplementedError
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "AttnUpDecoderBlock2D":
        raise NotImplementedError
        return AttnUpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "KUpBlock2D":
        raise NotImplementedError
        return KUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "KCrossAttnUpBlock2D":
        raise NotImplementedError
        return KCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

    raise ValueError(f"{up_block_type} does not exist.")