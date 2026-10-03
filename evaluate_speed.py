"""Comprehensive evaluation script for I_MMSeg on 3D held-out test cases.

Measures:
1. Per-slice inference latency (ms/slice)
2. Per-case inference latency and segmentation metrics (s/case, Dice, HD95)
3. Total test set inference time and overall throughput (FPS / slices per second)
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
import logging
import os
from pathlib import Path
import sys
import time

import h5py
try:
    from medpy import metric
except ImportError:
    metric = None
import numpy as np
from scipy.ndimage import zoom
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
datasets_path = PROJECT_ROOT / "datasets"
if str(datasets_path) not in sys.path:
    sys.path.insert(0, str(datasets_path))

try:
    from datasets.dataset_Myops import Myops_dataset
except (ModuleNotFoundError, ImportError):
    from dataset_Myops import Myops_dataset
from networks.vit_seg_configs import get_r50_b16_config
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def calculate_metric_percase(pred, gt):
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)
    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        if metric is not None:
            dice = float(metric.binary.dc(pred_bin, gt_bin))
            try:
                hd95 = float(metric.binary.hd95(pred_bin, gt_bin))
            except Exception:
                hd95 = 0.0
        else:
            dice = float(2.0 * np.sum(pred_bin * gt_bin) / (np.sum(pred_bin) + np.sum(gt_bin)))
            hd95 = 0.0
        return dice, hd95
    elif pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, 0.0


def benchmark_single_volume(
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    label: np.ndarray,
    model: torch.nn.Module,
    patch_size: list[int] = [128, 128],
    device: torch.device = torch.device("cuda"),
    warmup: bool = False
) -> tuple[np.ndarray, list[float], float]:
    """Infers a 3D volume slice by slice, measuring latency per slice and for the whole case."""
    img_bssfp, img_lge, img_t2w = images
    num_slices = img_bssfp.shape[2] if len(img_bssfp.shape) == 3 else 1
    slice_latencies_ms = []

    if len(img_bssfp.shape) == 3:
        prediction = np.zeros_like(label, dtype=np.uint8)
        
        # Synchronize before starting case timer
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_case_start = time.perf_counter()

        for ind in range(num_slices):
            slice_b = img_bssfp[:, :, ind]
            slice_l = img_lge[:, :, ind]
            slice_t = img_t2w[:, :, ind]
            x, y = slice_b.shape[0], slice_b.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice_b = zoom(slice_b, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice_l = zoom(slice_l, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice_t = zoom(slice_t, (patch_size[0] / x, patch_size[1] / y), order=3)

            input_b = torch.from_numpy(slice_b).unsqueeze(0).unsqueeze(0).float().to(device)
            input_l = torch.from_numpy(slice_l).unsqueeze(0).unsqueeze(0).float().to(device)
            input_t = torch.from_numpy(slice_t).unsqueeze(0).unsqueeze(0).float().to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_slice_start = time.perf_counter()

            with torch.no_grad():
                out = model(input_b, input_l, input_t)
                pred_slice = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_slice_end = time.perf_counter()
            slice_ms = (t_slice_end - t_slice_start) * 1000.0
            slice_latencies_ms.append(slice_ms)

            pred_slice_np = pred_slice.cpu().detach().numpy().astype(np.uint8)
            if x != patch_size[0] or y != patch_size[1]:
                pred_res = zoom(pred_slice_np, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred_res = pred_slice_np
            prediction[:, :, ind] = pred_res

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_case_total = time.perf_counter() - t_case_start

    else:
        # Single 2D image
        input_b = torch.from_numpy(img_bssfp).unsqueeze(0).unsqueeze(0).float().to(device)
        input_l = torch.from_numpy(img_lge).unsqueeze(0).unsqueeze(0).float().to(device)
        input_t = torch.from_numpy(img_t2w).unsqueeze(0).unsqueeze(0).float().to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        with torch.no_grad():
            out = model(input_b, input_l, input_t)
            pred_slice = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_case_total = time.perf_counter() - t_start
        slice_latencies_ms.append(t_case_total * 1000.0)
        prediction = pred_slice.cpu().detach().numpy().astype(np.uint8)

    return prediction, slice_latencies_ms, t_case_total


def evaluate_dataset(
    model: torch.nn.Module,
    data_root: str | Path,
    list_dir: str | Path,
    output_dir: str | Path,
    img_size: int = 128,
    save_predictions: bool = True,
    warmup_cases: int = 2,
    device: torch.device = torch.device("cuda")
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_root)
    b_path = str(data_root / "bSSFP" / "test_vol_h5")
    l_path = str(data_root / "LGE" / "test_vol_h5")
    t_path = str(data_root / "T2w" / "test_vol_h5")

    db_test = Myops_dataset(base_dir=b_path, base_dir1=l_path, base_dir2=t_path, split="test_vol", list_dir=str(list_dir))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"\n========================================================")
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ & ĐO TỐC ĐỘ INFERENCE (I_MMSeg)")
    print(f"========================================================")
    print(f"Tổng số ca bệnh kiểm thử (3D Volumes): {len(db_test)}")
    print(f"Thiết bị thực thi: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Độ phân giải patch: {img_size}x{img_size}")
    print(f"Thư mục lưu kết quả: {output_dir}\n")

    model.eval()

    # Warmup GPU
    if warmup_cases > 0 and len(db_test) > 0:
        print(f"🔄 Đang chạy warmup GPU trên {warmup_cases} ca đầu tiên...")
        for i, sample in enumerate(testloader):
            if i >= warmup_cases:
                break
            im0 = sample["image"].squeeze(0).numpy()
            im1 = sample["image1"].squeeze(0).numpy()
            im2 = sample["image2"].squeeze(0).numpy()
            lbl = sample["label"].squeeze(0).numpy()
            benchmark_single_volume((im0, im1, im2), lbl, model, patch_size=[img_size, img_size], device=device, warmup=True)
        print("✅ Hoàn tất warmup GPU!\n")

    per_case_records = []
    all_slice_latencies = []
    total_test_inference_time = 0.0
    total_slices_count = 0

    dice_myo_all = []
    dice_scar_all = []
    dice_edema_all = []
    dice_aar_all = []
    hd95_myo_all = []
    hd95_scar_all = []
    hd95_edema_all = []
    hd95_aar_all = []

    progress_bar = tqdm(testloader, desc="Testing 3D cases", unit="case")
    for sample in progress_bar:
        case_name = sample["case_name"][0]
        im_b = sample["image"].squeeze(0).numpy()
        im_l = sample["image1"].squeeze(0).numpy()
        im_t = sample["image2"].squeeze(0).numpy()
        label = sample["label"].squeeze(0).numpy()
        num_slices = im_b.shape[2] if len(im_b.shape) == 3 else 1

        pred, slice_times, case_time = benchmark_single_volume(
            (im_b, im_l, im_t),
            label,
            model,
            patch_size=[img_size, img_size],
            device=device
        )

        all_slice_latencies.extend(slice_times)
        total_test_inference_time += case_time
        total_slices_count += num_slices
        mean_slice_ms = float(np.mean(slice_times)) if slice_times else 0.0

        # Metrics: 1: Myocardium, 2: Scar, 3: Edema, AAR: 2 + 3
        d_myo, h_myo = calculate_metric_percase(pred == 1, label == 1)
        d_scar, h_scar = calculate_metric_percase(pred == 2, label == 2)
        d_edema, h_edema = calculate_metric_percase(pred == 3, label == 3)
        d_aar, h_aar = calculate_metric_percase((pred == 2) | (pred == 3), (label == 2) | (label == 3))

        dice_myo_all.append(d_myo)
        dice_scar_all.append(d_scar)
        dice_edema_all.append(d_edema)
        dice_aar_all.append(d_aar)

        hd95_myo_all.append(h_myo)
        hd95_scar_all.append(h_scar)
        hd95_edema_all.append(h_edema)
        hd95_aar_all.append(h_aar)

        record = {
            "case_name": case_name,
            "num_slices": num_slices,
            "case_time_sec": round(case_time, 4),
            "mean_slice_ms": round(mean_slice_ms, 2),
            "min_slice_ms": round(float(np.min(slice_times)), 2),
            "max_slice_ms": round(float(np.max(slice_times)), 2),
            "fps_case": round(num_slices / max(1e-5, case_time), 2),
            "dice_myocardium": round(d_myo, 4),
            "dice_scar": round(d_scar, 4),
            "dice_edema": round(d_edema, 4),
            "dice_aar": round(d_aar, 4),
            "hd95_myocardium": round(h_myo, 2),
            "hd95_scar": round(h_scar, 2),
            "hd95_edema": round(h_edema, 2),
            "hd95_aar": round(h_aar, 2),
        }
        per_case_records.append(record)

        # Save predictions if required
        if save_predictions:
            np.savez_compressed(pred_dir / f"{case_name}_pred.npz", prediction=pred, ground_truth=label)
            try:
                pred_itk = sitk.GetImageFromArray(pred.transpose(2, 0, 1).astype(np.float32))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, str(pred_dir / f"{case_name}_pred.nii.gz"))
            except Exception:
                pass

        progress_bar.set_postfix({
            "case_s": f"{case_time:.2f}s",
            "ms/slice": f"{mean_slice_ms:.1f}ms",
            "Scar Dice": f"{d_scar:.3f}"
        })

    # Summary Statistics
    slice_arr = np.array(all_slice_latencies)
    case_times_arr = np.array([r["case_time_sec"] for r in per_case_records])

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(per_case_records),
        "total_test_slices": int(total_slices_count),
        "total_inference_time_sec": round(float(total_test_inference_time), 3),
        "total_inference_time_min": round(float(total_test_inference_time / 60.0), 3),
        "overall_throughput_fps": round(float(total_slices_count / max(1e-5, total_test_inference_time)), 2),
        
        # Per slice latency stats (ms)
        "slice_latency_ms": {
            "mean": round(float(np.mean(slice_arr)), 2),
            "median": round(float(np.median(slice_arr)), 2),
            "std": round(float(np.std(slice_arr)), 2),
            "min": round(float(np.min(slice_arr)), 2),
            "max": round(float(np.max(slice_arr)), 2),
            "p95": round(float(np.percentile(slice_arr, 95)), 2),
            "p99": round(float(np.percentile(slice_arr, 99)), 2),
        },

        # Per case latency stats (seconds)
        "case_latency_sec": {
            "mean": round(float(np.mean(case_times_arr)), 3),
            "median": round(float(np.median(case_times_arr)), 3),
            "std": round(float(np.std(case_times_arr)), 3),
            "min": round(float(np.min(case_times_arr)), 3),
            "max": round(float(np.max(case_times_arr)), 3),
        },

        # Mean Segmentation Metrics
        "mean_metrics": {
            "dice_myocardium": round(float(np.mean(dice_myo_all)), 4),
            "dice_scar": round(float(np.mean(dice_scar_all)), 4),
            "dice_edema": round(float(np.mean(dice_edema_all)), 4),
            "dice_aar": round(float(np.mean(dice_aar_all)), 4),
            "mean_pathology_dice": round(float((np.mean(dice_scar_all) + np.mean(dice_edema_all)) / 2.0), 4),
            "overall_mean_dice": round(float((np.mean(dice_myo_all) + np.mean(dice_scar_all) + np.mean(dice_edema_all)) / 3.0), 4),
            "hd95_myocardium": round(float(np.mean(hd95_myo_all)), 2),
            "hd95_scar": round(float(np.mean(hd95_scar_all)), 2),
            "hd95_edema": round(float(np.mean(hd95_edema_all)), 2),
            "hd95_aar": round(float(np.mean(hd95_aar_all)), 2),
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    # Save to CSV and JSON
    csv_file = output_dir / "per_case_speed_and_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_records)

    json_file = output_dir / "test_speed_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Formatted Report
    print("\n" + "="*70)
    print("📊 BÁO CÁO TỔNG KẾT TỐC ĐỘ INFERENCE & CHỈ SỐ TEST TẬP MYOPS-380 (I_MMSeg)")
    print("="*70)
    print(f"Tổng số ca bệnh kiểm thử:          {summary['total_test_cases']} ca (volumes)")
    print(f"Tổng số lát cắt (2D slices):       {summary['total_test_slices']} lát cắt")
    print(f"⏱️ Tổng thời gian test toàn tập:   {summary['total_inference_time_sec']:.2f} giây ({summary['total_inference_time_min']:.2f} phút)")
    print(f"⚡ Tốc độ trung bình từng lát cắt:  {summary['slice_latency_ms']['mean']:.2f} ms/slice (Median: {summary['slice_latency_ms']['median']:.2f} ms)")
    print(f"⚡ Tốc độ trung bình từng ca bệnh:  {summary['case_latency_sec']['mean']:.3f} giây/ca (Min: {summary['case_latency_sec']['min']:.3f}s, Max: {summary['case_latency_sec']['max']:.3f}s)")
    print(f"🚀 Throughput xử lý:               {summary['overall_throughput_fps']:.1f} slices/giây (FPS)")
    print("-" * 70)
    print("🎯 CHỈ SỐ ĐỘ CHÍNH XÁC PHÂN ĐOẠN (DICE SCORE & HD95):")
    print(f"  • Myocardium (Cơ tim):           Dice = {summary['mean_metrics']['dice_myocardium']:.4f} | HD95 = {summary['mean_metrics']['hd95_myocardium']:.2f} voxels")
    print(f"  • Scar (Sẹo cơ tim):             Dice = {summary['mean_metrics']['dice_scar']:.4f} | HD95 = {summary['mean_metrics']['hd95_scar']:.2f} voxels")
    print(f"  • Edema (Phù nề):                Dice = {summary['mean_metrics']['dice_edema']:.4f} | HD95 = {summary['mean_metrics']['hd95_edema']:.2f} voxels")
    print(f"  • AAR (Vùng nguy cơ / Scar+Edema):Dice = {summary['mean_metrics']['dice_aar']:.4f} | HD95 = {summary['mean_metrics']['hd95_aar']:.2f} voxels")
    print(f"  ⭐ Mean Pathology Dice (Scar+Edema): {summary['mean_metrics']['mean_pathology_dice']:.4f}")
    print(f"  ⭐ Overall Mean Dice:              {summary['mean_metrics']['overall_mean_dice']:.4f}")
    print("="*70)
    print(f"📁 Chi tiết từng ca lưu tại:   {csv_file}")
    print(f"📁 Tóm tắt thống kê lưu tại:   {json_file}")
    print("="*70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate I_MMSeg model speed and accuracy on test_vol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default=f"{PROJECT_ROOT}/MyoPS380_dataset/Processed_data", help="Data root path")
    parser.add_argument("--list_dir", type=str, default=f"{PROJECT_ROOT}/list", help="List directory containing test_vol.txt")
    parser.add_argument("--output_dir", type=str, default=f"{PROJECT_ROOT}/runs/evaluation_speed", help="Output directory")
    parser.add_argument("--img_size", type=int, default=128, help="Patch size")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16", help="ViT config name")
    parser.add_argument("--vit_patches_size", type=int, default=16)
    parser.add_argument("--n_skip", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--save_predictions", action="store_true", default=True, help="Save .npz and .nii.gz predictions")
    parser.add_argument("--warmup_cases", type=int, default=2, help="Number of warmup cases before timing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    evaluate_dataset(
        model=net,
        data_root=args.data_root,
        list_dir=args.list_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        save_predictions=args.save_predictions,
        warmup_cases=args.warmup_cases,
        device=device
    )


if __name__ == "__main__":
    main()
