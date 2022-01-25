import torch
from ptflops import get_model_complexity_info

def model_summary(model: torch.nn.Module):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=True, verbose=True
        )
        
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay
