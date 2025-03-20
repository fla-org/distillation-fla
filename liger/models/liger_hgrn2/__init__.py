from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_hgrn2.configuration_liger_hgrn2 import LigerHGRN2Config
from liger.models.liger_hgrn2.modeling_liger_hgrn2 import LigerHGRN2ForCausalLM, LigerHGRN2Model

AutoConfig.register(LigerHGRN2Config.model_type, LigerHGRN2Config)
AutoModel.register(LigerHGRN2Config, LigerHGRN2Model)
AutoModelForCausalLM.register(LigerHGRN2Config, LigerHGRN2ForCausalLM)


__all__ = ['LigerHGRN2Config', 'LigerHGRN2ForCausalLM', 'LigerHGRN2Model']