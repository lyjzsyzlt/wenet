from __future__ import print_function

import torch
import yaml

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import save_checkpoint

if __name__ == '__main__':
    with open('../examples/aishell/s1/conf/train_unified_conformer.yaml', 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs['cmvn_file'] = '../examples/aishell/s1/fbank/train_sp/global_cmvn'
    configs['is_json_cmvn'] = False
    input_dim = 80
    vocab_size = 4234
    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size

    # Init asr model from configs
    model = init_asr_model(configs)
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print('============参数量=============：', num_parameters / 1000000)
    save_checkpoint(model, 'x.pt')