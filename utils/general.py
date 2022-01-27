import yaml

def load_yaml(file_path):
    file = open(file_path, 'r')
    options = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return options

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._last_val = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._last_val = val
        self._sum += val * n
        self._count += n
    
    @property
    def last_val(self):
        return self._last_val

    @property
    def avg(self):
        return self._sum / self._count