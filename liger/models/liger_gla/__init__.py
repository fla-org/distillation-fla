from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_gla.configuration_liger_gla import LigerGLAConfig
from liger.models.liger_gla.modeling_liger_gla import LigerGLAForCausalLM, LigerGLAModel

AutoConfig.register(LigerGLAConfig.model_type, LigerGLAConfig)
AutoModel.register(LigerGLAConfig, LigerGLAModel)
AutoModelForCausalLM.register(LigerGLAConfig, LigerGLAForCausalLM)


__all__ = ['LigerGLAConfig', 'LigerGLAForCausalLM', 'LigerGLAModel']