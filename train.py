import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_img = Path('/root/chennuo/ISIC2018_Task1-2_Training_Input/')
dir_mask = Path('/root/chennuo/ISIC2018_Task1_Training_GroundTruth/')
dir_checkpoint = Path('./checkpoints/')
# 新增筛选后数据保存目录
dir_best_data_img = Path('./best_data/imgs/')
dir_best_data_mask = Path('./best_data/masks/')
# 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha    # Focal Loss 正类权重
        self.gamma = gamma    # 难易样本调节因子
        self.smooth = smooth  # 防止除零

    def forward(self, inputs, targets, epoch=None, total_epochs=None):
        """
        :param inputs: 模型预测的 logits
        :param targets: 真实标签
        :param epoch: 当前训练的 epoch（用于动态权重调整）
        :param total_epochs: 总训练的 epoch 数
        """
        # 确保 targets 是整数类型
        targets = targets.long()

        # 确定任务类型
        if inputs.shape[1] > 1:  # 多分类任务
            inputs = torch.softmax(inputs, dim=1)  # 使用 softmax
            targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        else:  # 二分类任务
            inputs = torch.sigmoid(inputs)  # 使用 sigmoid
            targets_one_hot = targets.unsqueeze(1).float()

        # 动态权重调整
        focal_weight, dice_weight, boundary_weight = self._dynamic_weights(epoch, total_epochs)

        # 动态 Focal Loss
        bce_loss = F.binary_cross_entropy(inputs, targets_one_hot, reduction='none') if inputs.shape[1] == 1 else F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # 正确分类概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # 广义 Dice Loss
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()

        # 边界敏感损失
        boundary_mask = self._get_boundary_mask(targets_one_hot)
        boundary_loss = F.mse_loss(inputs * boundary_mask, targets_one_hot * boundary_mask)

        # 总损失
        total_loss = focal_weight * focal_loss + dice_weight * dice_loss + boundary_weight * boundary_loss
        return total_loss

    def _dynamic_weights(self, epoch, total_epochs):
        """
        动态调整损失函数权重
        :param epoch: 当前训练的 epoch
        :param total_epochs: 总训练的 epoch 数
        :return: focal_weight, dice_weight, boundary_weight
        """
        if epoch is None or total_epochs is None:
            # 默认权重
            return 0.4, 0.4, 0.2

        # 动态调整权重
        progress = epoch / total_epochs
        focal_weight = max(0.2, 0.4 * (1 - progress**2))  # 非线性减少 Focal Loss 权重
        dice_weight = min(0.6, 0.4 + 0.2 * progress**2)   # 非线性增加 Dice Loss 权重
        boundary_weight = 0.2  # 边界损失权重保持不变
        return focal_weight, dice_weight, boundary_weight

    def _get_boundary_mask(self, targets, dilation_radius=2):
        """
        生成边界掩码
        :param targets: 真实标签 (one-hot 编码, [batch_size, n_classes, height, width])
        :param dilation_radius: 边界扩展半径
        :return: 边界掩码
        """
        kernel = torch.ones((1, 1, 3, 3), device=targets.device)
        boundary_masks = []

        # 对每个通道单独计算边界
        for c in range(targets.shape[1]):  # 遍历每个类别
            channel = targets[:, c:c+1, :, :]  # 提取单个通道
            eroded = F.max_pool2d(channel, kernel_size=3, stride=1, padding=1)
            dilated = -F.max_pool2d(-channel, kernel_size=3, stride=1, padding=1)
            boundary = (dilated - eroded).abs()
            boundary_mask = F.conv2d(boundary, kernel, padding=dilation_radius) > 0
            boundary_masks.append(boundary_mask)

        # 合并所有通道的边界掩码
        boundary_mask = torch.stack(boundary_masks, dim=1).float().squeeze(2)

        # 调整边界掩码的尺寸与 targets 对齐
        boundary_mask = F.interpolate(boundary_mask, size=targets.shape[2:], mode='nearest')

        return boundary_mask


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders  修改过num_workers  
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = HybridLoss()
    global_step = 0

    # 5. Begin training   增加print 语句
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    # 计算复合损失函数
                    loss = criterion(masks_pred, true_masks, epoch=epoch, total_epochs=epochs)

                    if torch.isnan(masks_pred).any() or torch.isinf(masks_pred).any():
                        print("Model output contains NaN or Inf!")
                        break

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print("Loss contains NaN or Inf!")
                        break

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    return model, dataset

# 新增筛选函数
def select_best_data(model, dataset, device, top_percent=0.2, amp=False, epoch=None, total_epochs=None):
    if epoch is not None and total_epochs is not None:
        if epoch < total_epochs * 0.5:
            top_percent = 0.3  # 前 50% 训练阶段，选择更多样本
        else:
            top_percent = 0.1  # 后 50% 训练阶段，选择更少但高质量的样本

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    scores = []
    indices = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating all data")):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
            # 计算每个样本的 Dice 分数
            if model.n_classes == 1:
                score = 1 - dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
            else:
                score = 1 - dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
            scores.append(score.item())
            indices.append(i)

    # 按分数排序
    sorted_indices = [x for _, x in sorted(zip(scores, indices), reverse=True)]
    num_best = int(len(dataset) * top_percent)
    best_indices = sorted_indices[:num_best]

    # 创建保存目录
    dir_best_data_img.mkdir(parents=True, exist_ok=True)
    dir_best_data_mask.mkdir(parents=True, exist_ok=True)

    # 保存筛选后的数据
    import shutil
    for idx in best_indices:
        img_path = dataset.images[idx]
        mask_path = dataset.masks[idx]
        shutil.copy(img_path, dir_best_data_img / img_path.name)
        shutil.copy(mask_path, dir_best_data_mask / mask_path.name)

    logging.info(f"Selected top {top_percent * 100}% data saved to {dir_best_data_img} and {dir_best_data_mask}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--top-percent', type=float, default=0.2, help='Percentage of best data to select')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        # 初步训练模型
        trained_model, dataset = train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

        # 筛选高 Dice 数据
        select_best_data(trained_model, dataset, device, top_percent=args.top_percent, amp=args.amp, epoch=args.epochs, total_epochs=args.epochs)

        # 用筛选后的数据重新训练
        best_dataset = BasicDataset(dir_best_data_img, dir_best_data_mask, img_scale=args.scale)
        trained_model, _ = train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        # 初步训练模型
        trained_model, dataset = train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

        # 筛选高 Dice 数据
        select_best_data(trained_model, dataset, device, top_percent=args.top_percent, amp=args.amp, epoch=args.epochs, total_epochs=args.epochs)

        # 用筛选后的数据重新训练
        best_dataset = BasicDataset(dir_best_data_img, dir_best_data_mask, img_scale=args.scale)
        trained_model, _ = train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )