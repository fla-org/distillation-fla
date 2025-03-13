from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from lolcats.models.lolcats.configuration_lolcats import LolcatsConfig
from lolcats.models.lolcats.modeling_lolcats import LolcatsModel, LolcatsModelForCausalLM

AutoConfig.register(LolcatsConfig.model_type, LolcatsConfig)
AutoModel.register(LolcatsConfig, LolcatsModel)
AutoModelForCausalLM.register(LolcatsConfig, LolcatsModelForCausalLM)


__all__ = ['LolcatsConfig', 'LolcatsModelForCausalLM', 'LolcatsModel']