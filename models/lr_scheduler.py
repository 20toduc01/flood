import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_scheduler_by_step(optimizer, total_steps):
    """
    As described in the paper for in section 2.2. Note that this decays the learning rate after each step.
    For the ImageNet experiments, the authors used a linear schedule with a warmup period of 5 epochs.
    """
    scheduler = LambdaLR(
        optimizer,
        lambda k: math.cos(7 * math.pi * k / (16 * total_steps))
    )
    return scheduler