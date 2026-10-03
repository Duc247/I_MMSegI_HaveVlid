import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_datasets_dir = os.path.join(_cur_dir, "datasets")
if _datasets_dir not in sys.path:
    sys.path.insert(0, _datasets_dir)
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs): pass
            def add_scalar(self, *args, **kwargs): pass
            def add_image(self, *args, **kwargs): pass
            def close(self): pass
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, SoftmaxWeightedLoss
from utils import ContrastiveLoss
try:
    from torchvision import transforms
except ImportError:
    class DummyTransforms:
        @staticmethod
        def Compose(transforms_list):
            def apply_transforms(sample):
                for t in transforms_list:
                    sample = t(sample)
                return sample
            return apply_transforms
    transforms = DummyTransforms

import csv
import shutil

def validate(model, valloader, ce_loss, dice_loss, num_classes=4):
    model.eval()
    val_loss = 0.0
    dice_sums = {1: 0.0, 2: 0.0, 3: 0.0}
    dice_counts = {1: 0, 2: 0, 3: 0}
    with torch.no_grad():
        for val_batch in valloader:
            image_batch = val_batch['image'].cuda()
            image1_batch = val_batch['image1'].cuda()
            image2_batch = val_batch['image2'].cuda()
            label_batch = val_batch['label'].cuda()
            
            # ViT_seg eval returns out_seg
            out_seg = model(image_batch, image1_batch, image2_batch)
            l_ce = ce_loss(out_seg, label_batch)
            l_dice = dice_loss(out_seg, label_batch, softmax=True)
            batch_loss = (0.2 * l_ce + 0.8 * l_dice).item()
            val_loss += batch_loss

            preds = torch.argmax(torch.softmax(out_seg, dim=1), dim=1)
            for c in (1, 2, 3):
                p_c = (preds == c).float()
                l_c = (label_batch == c).float()
                for i_b in range(preds.size(0)):
                    p_i = p_c[i_b]
                    l_i = l_c[i_b]
                    intersect = 2.0 * (p_i * l_i).sum().item()
                    union = p_i.sum().item() + l_i.sum().item()
                    if union > 0:
                        dice_sums[c] += intersect / union
                        dice_counts[c] += 1
                    else:
                        dice_sums[c] += 1.0
                        dice_counts[c] += 1

    model.train()
    avg_loss = val_loss / max(1, len(valloader))
    d_myo = dice_sums[1] / max(1, dice_counts[1])
    d_scar = dice_sums[2] / max(1, dice_counts[2])
    d_edema = dice_sums[3] / max(1, dice_counts[3])
    d_mean = (d_myo + d_scar + d_edema) / 3.0
    d_pathology = (d_scar + d_edema) / 2.0
    return {
        "val_loss": avg_loss,
        "dice_myo": d_myo,
        "dice_scar": d_scar,
        "dice_edema": d_edema,
        "dice_mean": d_mean,
        "dice_pathology": d_pathology
    }


def trainer_Myops(args, model, snapshot_path):
    try:
        from datasets.dataset_Myops import Myops_dataset, RandomGenerator, ValGenerator
    except (ModuleNotFoundError, ImportError):
        from dataset_Myops import Myops_dataset, RandomGenerator, ValGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * max(1, args.n_gpu)

    # 1. Dataset Train (243 bệnh nhân / 1578 lát cắt chuẩn SCAR)
    db_train = Myops_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    num_workers = getattr(args, 'num_workers', 2)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # 2. Dataset Val (61 bệnh nhân / 394 lát cắt chuẩn SCAR)
    val_file = os.path.join(args.list_dir, "val.txt")
    has_val = os.path.exists(val_file)
    valloader = None
    if has_val:
        db_val = Myops_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="val",
                               transform=transforms.Compose([ValGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        print("The length of validation set is: {} slices (61 patients)".format(len(db_val)))
    else:
        print("Warning: val.txt not found in list_dir, running without validation.")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    con_loss = ContrastiveLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
    val_interval = getattr(args, 'val_interval', 1)
    backup_dir = getattr(args, 'backup_dir', None)

    metrics_csv_path = os.path.join(snapshot_path, "metrics.csv")
    csv_header = ["epoch", "train_loss", "val_loss", "dice_myo", "dice_scar", "dice_edema", "dice_mean", "dice_pathology", "lr", "epoch_time_s"]
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_header)

    best_score = -1.0
    best_epoch = -1
    best_val_metrics = {}

    start_epoch = 0
    if getattr(args, 'resume', None) and os.path.exists(args.resume):
        import re
        match = re.search(r'epoch_(\d+)\.pth', os.path.basename(args.resume))
        if match:
            start_epoch = int(match.group(1))
            iter_num = start_epoch * len(trainloader)
            print(f"🔄 Tiếp tục huấn luyện từ Epoch {start_epoch + 1}/{max_epoch} (Checkpoint: {args.resume})...")

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    for epoch_num in range(start_epoch, max_epoch):
        t_epoch_start = time.time()
        do_contrast = epoch_num > args.start_contrast_epoch
        epoch_train_loss = 0.0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch_num + 1:03d}/{max_epoch}", ncols=95, leave=False)
        for i_batch, sampled_batch in enumerate(pbar):
            image_batch, image1_batch, image2_batch, label_batch = sampled_batch['image'], sampled_batch['image1'], sampled_batch['image2'], sampled_batch['label']
            image_batch, image1_batch, image2_batch, label_batch = image_batch.cuda(), image1_batch.cuda(), image2_batch.cuda(), label_batch.cuda()
            out_pre, dec_seg, features_embedding_list, text_embedding_list= model(image_batch, image1_batch, image2_batch, do_contrast)
            ignores = ([2,3],[0],[0])
            loss_all = 0
            if do_contrast:
                for i in range(len(features_embedding_list)):
                    feature_list = features_embedding_list[i]
                    ignore = ignores[i]
                    loss_con = con_loss(feature_list,
                                        label_batch,
                                        text_embedding_list,
                                        ignore,
                                        sample_num = args.contrast_sample_num,
                                        )
                    loss_all += loss_con
                loss_all = loss_all /len(features_embedding_list)
            else:
                loss_all = 0
            out_cross_loss = ce_loss(out_pre, label_batch)
            out_dice_loss = dice_loss(out_pre, label_batch, softmax=True)
            out_loss = 0.2* out_cross_loss + 0.8* out_dice_loss
            dec_cross_loss = torch.zeros(1).cuda().float()
            dec_dice_loss = torch.zeros(1).cuda().float()
            for dec_pred in dec_seg:
                dec_cross_loss += ce_loss(dec_pred, label_batch)
                dec_dice_loss += dice_loss(dec_pred, label_batch, softmax=True)
            dec_loss = 0.2* dec_cross_loss + 0.8* dec_dice_loss
            
            if epoch_num < args.region_fusion_start_epoch:
                loss = out_loss * 0.0 + dec_loss+ loss_all * args.contrast_w
            else:
                loss = out_loss + 0.5 * dec_loss+ 0.5 * loss_all * args.contrast_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            epoch_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_:.5f}"})

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', out_dice_loss, iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d : loss : %f, loss_fuse: %f' % (iter_num, loss.item(), out_dice_loss.item()))

            if iter_num % 100 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image('train/Image', image, iter_num)
                out_pre_vis = torch.argmax(torch.softmax(out_pre, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', out_pre_vis[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        epoch_train_loss /= max(1, len(trainloader))
        epoch_time = time.time() - t_epoch_start

        # Validation Loop
        val_metrics = {"val_loss": 0.0, "dice_myo": 0.0, "dice_scar": 0.0, "dice_edema": 0.0, "dice_mean": 0.0, "dice_pathology": 0.0}
        if valloader is not None and (epoch_num + 1) % val_interval == 0:
            val_metrics = validate(model, valloader, ce_loss, dice_loss, num_classes)
            writer.add_scalar('val/loss', val_metrics["val_loss"], epoch_num + 1)
            writer.add_scalar('val/dice_mean', val_metrics["dice_mean"], epoch_num + 1)
            writer.add_scalar('val/dice_myo', val_metrics["dice_myo"], epoch_num + 1)
            writer.add_scalar('val/dice_scar', val_metrics["dice_scar"], epoch_num + 1)
            writer.add_scalar('val/dice_edema', val_metrics["dice_edema"], epoch_num + 1)

            summary_msg = (
                f"✅ [Epoch {epoch_num + 1:03d}/{max_epoch}] ({epoch_time:.1f}s) | "
                f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Mean Dice: {val_metrics['dice_mean']:.4f} [Myo: {val_metrics['dice_myo']:.4f}, Scar: {val_metrics['dice_scar']:.4f}, Edema: {val_metrics['dice_edema']:.4f}]"
            )
            print(summary_msg, flush=True)
            logging.info(summary_msg)

            # Check and save best model
            if val_metrics["dice_mean"] > best_score:
                best_score = val_metrics["dice_mean"]
                best_epoch = epoch_num + 1
                best_val_metrics = val_metrics
                best_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), best_path)
                best_msg = f"🏆 [NEW BEST] Epoch {best_epoch}: Val Mean Dice = {best_score:.4f} -> Saved best.pth"
                print(best_msg, flush=True)
                logging.info(best_msg)

                if backup_dir:
                    try:
                        os.makedirs(backup_dir, exist_ok=True)
                        shutil.copy2(best_path, os.path.join(backup_dir, 'best.pth'))
                        logging.info(f"Backed up best.pth to {backup_dir}")
                    except Exception as e:
                        logging.warning(f"Backup best.pth failed: {e}")

        # Always save latest checkpoint
        latest_path = os.path.join(snapshot_path, 'latest.pth')
        torch.save(model.state_dict(), latest_path)

        # Periodic checkpoint
        save_interval = 20  
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved periodic checkpoint: {save_mode_path}")

        # Append metrics to CSV
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch_num + 1,
                f"{epoch_train_loss:.5f}",
                f"{val_metrics['val_loss']:.5f}",
                f"{val_metrics['dice_myo']:.5f}",
                f"{val_metrics['dice_scar']:.5f}",
                f"{val_metrics['dice_edema']:.5f}",
                f"{val_metrics['dice_mean']:.5f}",
                f"{val_metrics['dice_pathology']:.5f}",
                f"{optimizer.param_groups[0]['lr']:.7f}",
                f"{epoch_time:.2f}"
            ])

        if backup_dir and (epoch_num + 1) % 5 == 0:
            try:
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(latest_path, os.path.join(backup_dir, 'latest.pth'))
                shutil.copy2(metrics_csv_path, os.path.join(backup_dir, 'metrics.csv'))
            except Exception as e:
                pass

    writer.close()
    logging.info(f"Training Finished! Best Epoch: {best_epoch} with Val Mean Dice: {best_score:.4f}")
    return {"best_epoch": best_epoch, "best_score": best_score, "metrics": best_val_metrics}
