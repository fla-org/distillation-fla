from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from lolcats.models.liger_llama3_8b_gla.configuration_liger_llama3_8b_gla import \
    LigerLlama3GLAConfig
from lolcats.models.liger_llama3_8b_gla.modeling_liger_llama3_8b_gla import (
    LigerLlama3GLAForCausalLM, LigerLlama3GLAModel)

AutoConfig.register(LigerLlama3GLAConfig.model_type, LigerLlama3GLAConfig)
AutoModel.register(LigerLlama3GLAConfig, LigerLlama3GLAModel)
AutoModelForCausalLM.register(LigerLlama3GLAConfig, LigerLlama3GLAForCausalLM)

__all__ = ['LolcatsConfig', 'LolcatsModelForCausalLM', 'LolcatsModel']
