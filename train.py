import wandb

from data_utils.augment import TrivialAugmentWide

WANDB = True

if WANDB:
    wandb.init(project="my-test-project", entity="toduck15hl")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm

from models import EMA, create_efficientnet_b1, create_efficientnet_b2, create_resnet18
from utils import AverageMeter, BinaryFocalLossWithLogits
from data_utils import LabeledDataset, UnlabaledDataset, \
    RandAugment, WeakAugment, Cutout
from models.lr_scheduler import cosine_scheduler_by_step
from models.utils import split_parameters


def train(device='cuda'):
    # Model
    model = create_resnet18(num_class=1, pretrained=True)
    # model = EMA(model, decay=0.9999)

    # Train data and loader
    augment_strong = T.Compose([
        TrivialAugmentWide(),
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Cutout(n_holes=1, length=64)
    ])

    s_set = LabeledDataset("./data/dev/images", transforms=augment_strong)
    # u_set = UnlabaledDataset("./U/images", transforms=[augment_weak, augment_strong]) # Manually apply transform later
    val_set = LabeledDataset("./data/val/images", transforms=T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))
    s_loader = torch.utils.data.DataLoader(
        s_set,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )
    # u_loader = torch.utils.data.DataLoader(
    #     u_set,
    #     batch_size=56,
    #     shuffle=True,
    #     num_workers=2
    # )

    epochs = 120

    # Optimizer
    decay_params, no_decay_params = split_parameters(model)
    optimizer = torch.optim.SGD(
        [
            {'params': decay_params, 'weight_decay': 0.0001},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=0.001, momentum=0.9, nesterov=True
    )

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14) # Prevents early overflow

    num_steps = len(s_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # criterion = torch.nn.CrossEntropyLoss(torch.Tensor([0.3, 0.7]).to(device))
    # criterion = BinaryFocalLossWithLogits(alpha=0.7, gamma=2)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.]).to(device))
    # criterion2 = torch.nn.CrossEntropyLoss(torch.Tensor([0.36, 0.64]), reduction='sum').to(device)
    model = model.to(device)
    model.train()

    cur_step = 0

    for ep in tqdm(range(1, epochs + 1)):
        epoch_loss = AverageMeter()
        for batch in tqdm(s_loader):
            cur_step += 1
            # Draw a batch of labeled samples
            x_S, y_S = batch

            with torch.cuda.amp.autocast():
                # Labeled loss
                x_S = x_S.to(device)
                y_S = y_S.to(device)
                pred_S = model(x_S)
                loss_S = criterion(pred_S, y_S.unsqueeze(1).float())
            
            scaler.scale(loss_S).backward()
            scaler.step(optimizer)
            scaler.update()
            
            model.zero_grad(set_to_none=True)
            
            epoch_loss.update(loss_S.item())

        scheduler.step()
        
        log_info = {"loss": epoch_loss.avg}

        if ep % 5 == 0 or ep == 1:
            print(f"Saving checkpoint at ep {ep}")
            val_loss = AverageMeter()
            torch.save(model.state_dict(), f"./weights/model_{ep}.pth")
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    x_val, y_val = batch
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    with torch.cuda.amp.autocast():
                        pred = model(x_val)
                        loss = criterion(pred, y_val.unsqueeze(1).float())
                    val_loss.update(loss.item())
                log_info["val_loss"] = val_loss.avg
            model.train()

        if WANDB:
                wandb.log(log_info)

    return model

        
if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), './weights/fin.pth')