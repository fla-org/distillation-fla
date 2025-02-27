from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from liger.models.liger_gsa.configuration_liger_gsa import LigerGSAConfig
from liger.models.liger_gsa.modeling_liger_gsa import LigerGSAForCausalLM, LigerGSAModel

AutoConfig.register(LigerGSAConfig.model_type, LigerGSAConfig)
AutoModel.register(LigerGSAConfig, LigerGSAModel)
AutoModelForCausalLM.register(LigerGSAConfig, LigerGSAForCausalLM)


__all__ = ['LigerGSAConfig', 'LigerGSAForCausalLM', 'LigerGSAModel']