from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_mistral_gla.configuration_liger_mistral_gla import LigerMistralGLAConfig
from liger.models.liger_mistral_gla.modeling_liger_mistral_gla import LigerMistralGLAForCausalLM, LigerMistralGLAModel

AutoConfig.register(LigerMistralGLAConfig.model_type, LigerMistralGLAConfig)
AutoModel.register(LigerMistralGLAConfig, LigerMistralGLAModel)
AutoModelForCausalLM.register(LigerMistralGLAConfig, LigerMistralGLAForCausalLM)


__all__ = ['LigerMistralGLAConfig', 'LigerMistralGLAForCausalLM', 'LigerMistralGLAModel']