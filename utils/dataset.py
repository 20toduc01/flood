import os
import random
import pandas as pd

from typing import Optional, Sequence, Callable
from PIL import Image
from torch.utils.data import Dataset

# TODO: Once I decided on metadata format, which is only needed when
# I want to take metadata info into acount, I should implement 
# return metadata if exists in __getitem__
class LabeledDataset(Dataset):
    def __init__(
        self,
        folder_path: str = './data/dev/images',
        transforms: Sequence[Callable] = None,
        num_classes: int = 2,
        meta_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        # Variable initialization
        self.folder_path = folder_path
        self.transform = transforms
        self.meta_path = meta_path
        if meta_path is not None:
            self.meta_df = pd.read_csv(meta_path)

        # Keep a list of image names and their class (subfolder)
        self.num_classes = num_classes
        im_list = []

        for class_idx in range(num_classes):
            im_names = os.listdir(os.path.join(folder_path, str(class_idx)))
            im_list.extend([(class_idx, im_name) for im_name in im_names])
        
        if seed is not None:
            rd = random.Random(seed)
            rd.shuffle(im_list)
        else:
            random.shuffle(im_list)

        self.im_list = im_list

    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx: int):
        # Get image path
        class_idx, im_name = self.im_list[idx]
        im_path = os.path.join(
            self.folder_path, str(class_idx), im_name
        )

        # Get image
        im = Image.open(im_path).convert('RGB')

        # Transform image
        if self.transform is not None:
            im = self.transform(im)

        return im, class_idx


class UnlabaledDataset(Dataset):
    def __init__(
        self,
        folder_path: str = './data/test/images',
        transforms: Sequence[Callable] = None,
        meta_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        # Variable initialization
        self.folder_path = folder_path
        self.transforms = transforms
        if isinstance(transforms, list) and len(transforms) > 1:
            self.multi_transforms = True
        self.meta_path = meta_path
        if meta_path is not None:
            self.meta_df = pd.read_csv(meta_path)

        # Keep a list of image names and their class (subfolder)
        im_list = []
        sub_dirs = os.listdir(folder_path)
        
        for dir in sub_dirs:
            im_names = os.listdir(os.path.join(folder_path, dir))
            im_list.extend([(dir, im_name) for im_name in im_names])

        if seed is not None:
            rd = random.Random(seed)
            rd.shuffle(im_list)
        else:
            random.shuffle(im_list)

        self.im_list = im_list

    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx: int):
        # Get image path
        sub_dir, im_name = self.im_list[idx]
        im_path = os.path.join(
            self.folder_path, str(sub_dir), im_name
        )

        # Get image
        im = Image.open(im_path)

        # Transform image
        if self.transforms is None:
            return

        if self.multi_transforms:
            imgs = []
            for t in self.transforms:
                imgs.append(t(im))
            return imgs
        else:
            return self.transforms(im)


def _test():
    import random
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    gd = 3
    
    S = LabeledDataset(
        folder_path='./data/dev/images',
        transforms=None,
        num_classes=2,
    )

    size = len(S)
    fig = plt.figure(figsize=(gd*2, gd*2))
    grid = ImageGrid(fig, 111,
        nrows_ncols=(gd, gd),
        axes_pad=0.4,  # pad between axes in inch.
    )

    for ax in grid:
        idx = random.randint(0, size - 1)
        im, gt = S[idx]
        ax.imshow(im)
        ax.set_title("Flood" if gt == 1 else "No Flood")
    plt.show()

    U = UnlabaledDataset(
        folder_path='./data/U/images'
    )

    size = len(U)
    fig = plt.figure(figsize=(gd*2, gd*2))
    grid = ImageGrid(fig, 111,
        nrows_ncols=(gd, gd),
        axes_pad=0.4,  # pad between axes in inch.
    )

    for ax in grid:
        idx = random.randint(0, size - 1)
        im = U[idx]
        ax.imshow(im)
    plt.show()


if __name__ == "__main__":
    _test()
