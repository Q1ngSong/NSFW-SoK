import sys
import os

# Register this llava folder as top-level package alias "llava"
this_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

sys.modules["llava"] = sys.modules[__name__]

from .model import LlavaLlamaForCausalLM
