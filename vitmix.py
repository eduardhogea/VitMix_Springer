#!/usr/bin/env python
# coding: utf-8
"""
Script for:
  - Training ViT-based models on the PH2 dataset (medical mode) or using ImageNet-pretrained weights (imagenet mode)
    OR evaluating on the VOC segmentation validation set (voc mode).
  - Generating explanation maps (LRP, Saliency, Rollout, CAM)
  - Visualizing 1-way, 2-way, and 3-way combinations, computing metrics (IoU, F1, Pixel Accuracy, and Deletion AUC)
  - Saving CSV results and printing the top 2 methods (by F1)
  - Also evaluating on student-provided masks (from the "masks" folder) for medical and imagenet modes.
Usage: python script.py [medical|imagenet|voc]
"""

import os, sys, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from skimage.io import imread
from torchvision.datasets import VOCSegmentation

repo_path = "Transformer-Explainability"
if repo_path not in sys.path:
    sys.path.append(repo_path)
from baselines.ViT.ViT_explanation_generator import LRP, Baselines
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_LRP_new

plt.rcParams['figure.figsize'] = (15, 4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1].lower() in ["medical", "imagenet", "voc"]:
    eval_mode = sys.argv[1].lower()
else:
    eval_mode = "medical"
print(f"Evaluation mode: {eval_mode}")

# -----------------------------------------------------------------------------
# Helper Functions (common)
# -----------------------------------------------------------------------------
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap((255 * mask).astype(np.uint8), cv2.COLORMAP_JET) / 255.0
    cam = heatmap + np.array(img)
    return (cam / np.max(cam) * 255).astype(np.uint8)

def combine_and_visualize(attr, input_image, use_thresh=True):
    attr = F.interpolate(attr, scale_factor=16, mode='bilinear').reshape(224,224).cpu().detach().numpy()
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    if use_thresh:
        attr = cv2.threshold((attr * 255).astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        attr[attr == 255] = 1
    input_image_np = input_image.permute(1,2,0).cpu().detach().numpy()
    min_val = input_image.min().item()
    max_val = input_image.max().item()
    img_vis = (input_image_np - min_val) / (max_val - min_val + 1e-8)
    return show_cam_on_image(img_vis, attr), attr

def deletion_metric(model, image, attribution_map, class_index=None, steps=100):
    model.eval()
    if isinstance(attribution_map, np.ndarray):
        attribution_map = torch.from_numpy(attribution_map).unsqueeze(0).unsqueeze(0).float()
    importance_order = np.argsort(-attribution_map.detach().cpu().flatten().numpy())
    image_np = image.permute(1,2,0).cpu().detach().numpy()
    modified_image = image_np.copy()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        if class_index is None:
            class_index = output.argmax().item()
        initial_confidence = torch.softmax(output, dim=1)[0, class_index].item()
    confidence_drop = [initial_confidence]
    total_pixels = image_np.shape[0] * image_np.shape[1]
    pixels_per_step = total_pixels // steps
    for step in range(1, steps+1):
        pixels_to_mask = importance_order[(step-1)*pixels_per_step: step*pixels_per_step]
        for idx in pixels_to_mask:
            h, w = divmod(int(idx), image_np.shape[1])
            modified_image[h, w, :] = 0
        modified_image_tensor = torch.from_numpy(modified_image).permute(2,0,1).float().to(device)
        with torch.no_grad():
            output = model(modified_image_tensor.unsqueeze(0))
            confidence = torch.softmax(output, dim=1)[0, class_index].item()
        confidence_drop.append(confidence)
    x_axis = np.linspace(0, 1, len(confidence_drop))
    auc_score = auc(x_axis, confidence_drop)
    return auc_score, confidence_drop

def compute_metrics(mask, gt):
    inter = np.logical_and(gt, mask).sum()
    union = np.logical_or(gt, mask).sum()
    jaccard = inter / union if union else 0
    tp = inter; fp = mask.sum() - tp; fn = gt.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    pix_acc = inter / gt.sum() if gt.sum() > 0 else 0
    return jaccard, f1, pix_acc

def scale_and_print_stats(df):
    for col in ["Jaccard Index (IoU)", "F1 Score", "Pixel Accuracy"]:
        df[col] = (df[col] * 100).round(2)
    if "Deletion" in df.columns:
        df["Deletion"] = df["Deletion"].round(2)
    print(df)

def save_and_display_results(filename, results):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    if df.empty:
        print("No results to display statistics.")
    else:
        print("Statistics by Method:")
        stats = df.groupby("Method")[["Jaccard Index (IoU)", "F1 Score", "Pixel Accuracy", "Deletion"]].mean()
        scale_and_print_stats(stats)

def print_top_two(csv_file, label=""):
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"No results for {label}.")
        return
    grouped = df.groupby("Method")[["F1 Score", "Jaccard Index (IoU)", "Pixel Accuracy", "Deletion"]].mean()
    grouped["F1 Score"] = (grouped["F1 Score"] * 100).round(2)
    grouped["Jaccard Index (IoU)"] = (grouped["Jaccard Index (IoU)"] * 100).round(2)
    grouped["Pixel Accuracy"] = (grouped["Pixel Accuracy"] * 100).round(2)
    grouped["Deletion"] = grouped["Deletion"].round(2)
    print(f"\nTop 2 methods for {label} (by F1 Score):")
    print(grouped.sort_values("F1 Score", ascending=False).head(2))

# -----------------------------------------------------------------------------
# Explanation Generation Functions
# -----------------------------------------------------------------------------
def get_explanation_functions(model_A, model_B):
    b = Baselines(model_A)
    attr_gen = LRP(model_B)
    def generate_LRP_expl(img, cls_idx):
        return attr_gen.generate_LRP(img.unsqueeze(0), method="transformer_attribution", index=cls_idx).detach()
    def generate_saliency_expl(img, cls_idx):
        img.requires_grad_()
        out = model_B(img.unsqueeze(0))
        cls_idx = cls_idx if cls_idx is not None else out.argmax().item()
        loss = out[0, cls_idx]
        model_B.zero_grad(); loss.backward()
        sal = img.grad.data.abs().max(dim=0, keepdim=True)[0]
        return F.interpolate(sal.unsqueeze(0), size=(14,14), mode='bilinear')
    def generate_rollout_expl(img, start_layer=3):
        return b.generate_rollout(img.unsqueeze(0), start_layer=start_layer)
    def generate_CAM_expl(img, cls_idx):
        return b.generate_cam_attn(img.unsqueeze(0), index=cls_idx).reshape(1,1,14,14)
    return generate_LRP_expl, generate_saliency_expl, generate_rollout_expl, generate_CAM_expl

# -----------------------------------------------------------------------------
# Evaluation & Visualization Functions for Ground Truth Masks
# -----------------------------------------------------------------------------
def visualize_methods(input_image, model_A, gen_funcs, use_thresh=True):
    cls_idx = model_A(input_image.unsqueeze(0)).argmax().item()
    gen_LRP, gen_sal, gen_roll, gen_CAM = gen_funcs
    methods = {
        'LRP':      lambda img: gen_LRP(img, cls_idx).reshape(1,1,14,14),
        'saliency': lambda img: gen_sal(img, cls_idx),
        'rollout':  lambda img: gen_roll(img).reshape(1,1,14,14),
        'CAM':      lambda img: gen_CAM(img, cls_idx)
    }
    return [(name, *combine_and_visualize(func(input_image), input_image, use_thresh))
            for name, func in methods.items()]

def visualize_combinations(input_image, model_A, gen_funcs, n=2, combine_methods=('sqrt','multiply'), use_thresh=True):
    cls_idx = model_A(input_image.unsqueeze(0)).argmax().item()
    gen_LRP, gen_sal, gen_roll, gen_CAM = gen_funcs
    methods = {
        'LRP':      lambda img: gen_LRP(img, cls_idx).reshape(1,1,14,14),
        'saliency': lambda img: gen_sal(img, cls_idx),
        'rollout':  lambda img: gen_roll(img).reshape(1,1,14,14),
        'CAM':      lambda img: gen_CAM(img, cls_idx).reshape(1,1,14,14)
    }
    results = []
    for combo in combinations(methods.keys(), n):
        for cm in combine_methods:
            attrs = [methods[m](input_image).to(device) for m in combo]
            if n == 3:
                combined = torch.sqrt(attrs[0] * attrs[1] * attrs[2]) if cm=='sqrt' else torch.prod(torch.stack(attrs), dim=0)
            else:
                combined = torch.sqrt(attrs[0] * attrs[1]) if cm=='sqrt' else attrs[0] * attrs[1]
            vis, mask = combine_and_visualize(combined, input_image, use_thresh)
            method_name = " + ".join(combo) + f" ({cm})"
            results.append((method_name, vis, mask))
    return results

def run_evaluation(evaluator, dataset_images, dataset_masks, base_transform, out_folder, csv_name, model_A, gen_funcs):
    os.makedirs(out_folder, exist_ok=True)
    all_results = []
    for idx, (img_data, mask_data) in enumerate(tqdm(list(zip(dataset_images, dataset_masks)),
                                                       total=len(dataset_images),
                                                       desc="Ground Truth Evaluation")):
        image = Image.fromarray(img_data).convert('RGB')
        gt_mask = (np.array(transforms.Resize((224,224))(Image.fromarray(mask_data).convert('L'))) > 0).astype(np.uint8)
        im_tens = base_transform(np.array(image)).to(device)
        for method_name, vis, mask in evaluator(im_tens, model_A, gen_funcs):
            auc_cam, _ = deletion_metric(model_A, im_tens, mask)
            jaccard, f1, pix_acc = compute_metrics(mask, gt_mask)
            all_results.append({
                "Image Index": idx,
                "Method": method_name,
                "Jaccard Index (IoU)": jaccard,
                "F1 Score": f1,
                "Pixel Accuracy": pix_acc,
                "Deletion": auc_cam
            })
            method_dir = os.path.join(out_folder, f"method_{method_name.replace(' ', '_')}")
            os.makedirs(method_dir, exist_ok=True)
            image.save(os.path.join(method_dir, f"image_{idx}.png"))
            Image.fromarray((gt_mask*255).astype(np.uint8)).save(os.path.join(method_dir, f"gt_mask_{idx}.png"))
            Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(method_dir, f"result_mask_{idx}.png"))
    save_and_display_results(csv_name, all_results)

def run_voc_evaluation(evaluator, voc_dataset, base_transform, out_folder, csv_name, model_A, gen_funcs):
    os.makedirs(out_folder, exist_ok=True)
    all_results = []
    for idx, (img, target) in enumerate(tqdm(voc_dataset, desc="VOC Evaluation")):
        gt_mask = (np.array(transforms.Resize((224,224))(target)) > 0).astype(np.uint8)
        im_tens = base_transform(np.array(img)).to(device)
        with torch.no_grad():
            logits = model_A(im_tens.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            max_prob = probs.max().item()
        if max_prob < 0.8:
            continue
        for method_name, vis, mask in evaluator(im_tens, model_A, gen_funcs):
            auc_cam, _ = deletion_metric(model_A, im_tens, mask)
            jaccard, f1, pix_acc = compute_metrics(mask, gt_mask)
            all_results.append({
                "Image Index": idx,
                "Method": method_name,
                "Jaccard Index (IoU)": jaccard,
                "F1 Score": f1,
                "Pixel Accuracy": pix_acc,
                "Deletion": auc_cam
            })
            method_dir = os.path.join(out_folder, f"method_{method_name.replace(' ', '_')}")
            os.makedirs(method_dir, exist_ok=True)
            img.save(os.path.join(method_dir, f"image_{idx}.png"))
            Image.fromarray((gt_mask*255).astype(np.uint8)).save(os.path.join(method_dir, f"gt_mask_{idx}.png"))
            Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(method_dir, f"result_mask_{idx}.png"))
    save_and_display_results(csv_name, all_results)

def run_1way_student(eval_dict):
    print(f"Running 1-Way student evaluation, mode={eval_mode} ...")
    methods = ['LRP','saliency','rollout','CAM']
    transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    all_results = []
    for img_path, mask_list in tqdm(sorted(eval_dict.items()), desc="1-Way Student Evaluation"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tens_ = transform_(img_rgb).to(device)
        pred = model_A(tens_.unsqueeze(0))
        c_idx = pred.argmax().item()
        for (mk, student_id) in mask_list:
            m_gray = cv2.imread(mk, cv2.IMREAD_GRAYSCALE)
            if m_gray is None:
                continue
            m_gray = cv2.resize(m_gray, (224,224), interpolation=cv2.INTER_NEAREST)
            m_bin = (m_gray > 0).astype(np.uint8)
            for mm in methods:
                if mm == 'LRP':
                    attr = generate_LRP(tens_, c_idx).reshape(1,1,14,14)
                elif mm == 'saliency':
                    attr = generate_saliency(tens_, c_idx)
                elif mm == 'rollout':
                    attr = generate_rollout(tens_).reshape(1,1,14,14)
                elif mm == 'CAM':
                    attr = generate_CAM(tens_, c_idx).reshape(1,1,14,14)
                else:
                    continue
                auc_cam, _ = deletion_metric(model_A, tens_, attr)
                up = F.interpolate(attr, scale_factor=16, mode='bilinear')
                up = up.reshape(224,224).cpu().detach().numpy()
                up = (up - up.min())/(up.max()-up.min()+1e-8)
                up = (up * 255).astype(np.uint8)
                _, up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                up[up==255] = 1
                iou, f1, px = compute_metrics(up, m_bin)
                all_results.append({
                    "ImagePath": img_path,
                    "Student": student_id,
                    "Method": mm,
                    "Jaccard Index (IoU)": iou,
                    "F1 Score": f1,
                    "Pixel Accuracy": px,
                    "Deletion": auc_cam
                })
    df = pd.DataFrame(all_results)
    out_name = f"metrics_results_1WAY_student_{eval_mode}.csv"
    df.to_csv(out_name, index=False)
    print(f"Saved 1-Way student results to {out_name}")
    stats = df.groupby("Method")[["Jaccard Index (IoU)", "F1 Score", "Pixel Accuracy", "Deletion"]].mean()
    scale_and_print_stats(stats)

def run_2way_student(eval_dict):
    print(f"Running 2-Way student evaluation, mode={eval_mode} ...")
    single_methods = ['LRP','saliency','rollout','CAM']
    combos = list(combinations(single_methods,2))
    combine_methods = ['sqrt','multiply']
    transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    all_results = []
    for img_path, mask_list in tqdm(sorted(eval_dict.items()), desc="2-Way Student Evaluation"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tens_ = transform_(img_rgb).to(device)
        out_ = model_A(tens_.unsqueeze(0))
        c_idx = out_.argmax().item()
        def get_attr(mm):
            if mm=='LRP':
                return generate_LRP(tens_, c_idx).reshape(1,1,14,14).to(device)
            elif mm=='saliency':
                return generate_saliency(tens_, c_idx).to(device)
            elif mm=='rollout':
                return generate_rollout(tens_).reshape(1,1,14,14).to(device)
            elif mm=='CAM':
                return generate_CAM(tens_, c_idx).reshape(1,1,14,14).to(device)
            else:
                raise ValueError("Unknown method")
        single_attrs = {sm: get_attr(sm) for sm in single_methods}
        for (mk, st_id) in mask_list:
            m_gray = cv2.imread(mk, cv2.IMREAD_GRAYSCALE)
            if m_gray is None:
                continue
            m_gray = cv2.resize(m_gray, (224,224), interpolation=cv2.INTER_NEAREST)
            m_bin = (m_gray > 0).astype(np.uint8)
            for (m1, m2) in combos:
                for cm in combine_methods:
                    if cm=='sqrt':
                        comb = torch.sqrt(single_attrs[m1]*single_attrs[m2])
                    else:
                        comb = single_attrs[m1]*single_attrs[m2]
                    auc_cam, _ = deletion_metric(model_A, tens_, comb)
                    up = F.interpolate(comb, scale_factor=16, mode='bilinear')
                    up = up.reshape(224,224).cpu().detach().numpy()
                    up = (up - up.min())/(up.max()-up.min()+1e-8)
                    up = (up*255).astype(np.uint8)
                    _, up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    up[up==255]=1
                    iou, f1, px = compute_metrics(up, m_bin)
                    method_name = f"{m1} + {m2} ({cm})"
                    all_results.append({
                        "ImagePath": img_path,
                        "Student": st_id,
                        "Method": method_name,
                        "Jaccard Index (IoU)": iou,
                        "F1 Score": f1,
                        "Pixel Accuracy": px,
                        "Deletion": auc_cam
                    })
    df = pd.DataFrame(all_results)
    out_name = f"metrics_results_2WAY_student_{eval_mode}.csv"
    df.to_csv(out_name, index=False)
    print(f"Saved 2-Way student results to {out_name}")
    stats = df.groupby("Method")[["Jaccard Index (IoU)", "F1 Score", "Pixel Accuracy", "Deletion"]].mean()
    scale_and_print_stats(stats)

def run_3way_student(eval_dict):
    print(f"Running 3-Way student evaluation, mode={eval_mode} ...")
    single_methods = ['LRP','saliency','rollout','CAM']
    combos = list(combinations(single_methods,3))
    combine_methods = ['sqrt','multiply']
    transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    all_results = []
    for img_path, mask_list in tqdm(sorted(eval_dict.items()), desc="3-Way Student Evaluation"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tens_ = transform_(img_rgb).to(device)
        out_ = model_A(tens_.unsqueeze(0))
        c_idx = out_.argmax().item()
        def get_attr(m):
            if m=='LRP':
                return generate_LRP(tens_, c_idx).reshape(1,1,14,14).to(device)
            elif m=='saliency':
                return generate_saliency(tens_, c_idx).to(device)
            elif m=='rollout':
                return generate_rollout(tens_).reshape(1,1,14,14).to(device)
            elif m=='CAM':
                return generate_CAM(tens_, c_idx).reshape(1,1,14,14).to(device)
            else:
                raise ValueError("Unknown method")
        single_attrs = {sm: get_attr(sm) for sm in single_methods}
        for (mk, st_id) in mask_list:
            m_gray = cv2.imread(mk, cv2.IMREAD_GRAYSCALE)
            if m_gray is None:
                continue
            m_gray = cv2.resize(m_gray, (224,224), interpolation=cv2.INTER_NEAREST)
            m_bin = (m_gray > 0).astype(np.uint8)
            for combo in combos:
                for cm in combine_methods:
                    a1 = single_attrs[combo[0]]
                    a2 = single_attrs[combo[1]]
                    a3 = single_attrs[combo[2]]
                    if cm=='sqrt':
                        combined = torch.sqrt(a1*a2*a3)
                    else:
                        combined = a1*a2*a3
                    auc_cam, _ = deletion_metric(model_A, tens_, combined)
                    up = F.interpolate(combined, scale_factor=16, mode='bilinear')
                    up = up.reshape(224,224).cpu().detach().numpy()
                    up = (up - up.min())/(up.max()-up.min()+1e-8)
                    up = (up*255).astype(np.uint8)
                    _, up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    up[up==255]=1
                    iou, f1, px = compute_metrics(up, m_bin)
                    method_name = f"{combo[0]} + {combo[1]} + {combo[2]} ({cm})"
                    all_results.append({
                        "ImagePath": img_path,
                        "Student": st_id,
                        "Method": method_name,
                        "Jaccard Index (IoU)": iou,
                        "F1 Score": f1,
                        "Pixel Accuracy": px,
                        "Deletion": auc_cam
                    })
    df = pd.DataFrame(all_results)
    out_name = f"metrics_results_3WAY_student_{eval_mode}.csv"
    df.to_csv(out_name, index=False)
    print(f"Saved 3-Way student results to {out_name}")
    stats = df.groupby("Method")[["Jaccard Index (IoU)", "F1 Score", "Pixel Accuracy", "Deletion"]].mean()
    scale_and_print_stats(stats)

if eval_mode in ["medical", "imagenet"]:
    if eval_mode == "medical":
        drive_path = r"C:\VitMix\Transformer-Explainability\PH2Dataset"
        root_path = os.path.join(drive_path, "PH2_Dataset")
        metadata_path = os.path.join(drive_path, "PH2_dataset.txt")
        metadata = pd.read_csv(metadata_path, sep="\\|", engine="python", skipinitialspace=True)
        metadata.columns = metadata.columns.str.strip()
        metadata = metadata.dropna(how="all", axis=1)[["Name", "Clinical Diagnosis"]]
        metadata["Name"] = metadata["Name"].str.strip()
        metadata["Clinical Diagnosis"] = pd.to_numeric(metadata["Clinical Diagnosis"], errors="coerce")
        metadata.dropna(subset=["Clinical Diagnosis"], inplace=True)
        metadata["Clinical Diagnosis"] = metadata["Clinical Diagnosis"].astype(int)
        
        images, lesions, image_names = [], [], []
        for dirpath, _, files in os.walk(root_path):
            if files:
                if dirpath.endswith('_Dermoscopic_Image'):
                    image_names.append(os.path.basename(dirpath).replace('_Dermoscopic_Image', ''))
                    images.append(imread(os.path.join(dirpath, files[0])))
                elif dirpath.endswith('_lesion'):
                    lesions.append(imread(os.path.join(dirpath, files[0])))

        matched_labels, matched_images, matched_lesions = [], [], []
        for i, name in enumerate(image_names):
            row = metadata[metadata["Name"] == name]
            if not row.empty:
                matched_labels.append(row["Clinical Diagnosis"].values[0])
                matched_images.append(images[i] if i < len(images) else None)
                matched_lesions.append(lesions[i] if i < len(lesions) else None)

        cleaned = [(img, les, lab) for img, les, lab in zip(matched_images, matched_lesions, matched_labels)
                   if img is not None and les is not None and not np.isnan(lab)]
        if cleaned:
            cleaned_images, cleaned_lesions, cleaned_labels = zip(*cleaned)
            cleaned_images, cleaned_lesions, cleaned_labels = list(cleaned_images), list(cleaned_lesions), list(cleaned_labels)
        else:
            cleaned_images = cleaned_lesions = cleaned_labels = []
        print(f"Cleaned {len(cleaned_images)} PH2 images, {len(cleaned_lesions)} masks, and {len(cleaned_labels)} labels.")
        if cleaned_images:
            print("Nice!")
        else:
            print("No valid PH2 samples to display.")
        X_ph2, y_ph2 = list(cleaned_images), list(cleaned_labels)
        base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print("\n--- MEDICAL mode: Training on PH2 dataset ---")
        model_A = vit_LRP_new(pretrained=True, num_classes=3).to(device)
        model_B = vit_LRP(pretrained=True, num_classes=3).to(device)
        train_counts = Counter(y_ph2)
        total_samples = sum(train_counts.values())
        weights = [total_samples / train_counts[i] for i in range(3)]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.1,0.1)),
            transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        class PH2Dataset(Dataset):
            def __init__(self, images, labels, transform):
                self.images, self.labels, self.transform = images, labels, transform
            def __len__(self):
                return len(self.images)
            def __getitem__(self, idx):
                return self.transform(self.images[idx]), self.labels[idx]
        train_dataset = PH2Dataset(X_ph2, y_ph2, transform=aug_transform)
        val_dataset = PH2Dataset(X_ph2, y_ph2, transform=base_transform)
        batch_size = 16
        sampler = WeightedRandomSampler([1/train_counts[lab] for lab in y_ph2], len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = optim.AdamW(model_B.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        def train_model(model, epochs=30, patience=7):
            best_acc, stop_count = 0, 0
            for epoch in range(epochs):
                model.train()
                losses = []
                for imgs, lbls in train_loader:
                    imgs, lbls = imgs.to(device), lbls.long().to(device)
                    loss = criterion(model(imgs), lbls)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    losses.append(loss.item())
                avg_loss = np.mean(losses)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
                model.eval()
                correct, total = 0, 0
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.long().to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == lbls).sum().item(); total += lbls.size(0)
                acc = 100 * correct / total
                print(f"Validation Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc, stop_count = acc, 0
                    torch.save(model.state_dict(), "best_model.pth")
                else:
                    stop_count += 1
                    if stop_count >= patience:
                        print("Early stopping triggered."); break
                scheduler.step()
            return model
        model_B = train_model(model_B)
        torch.save(model_B.state_dict(), "best_model.pth")
        model_A.load_state_dict(torch.load("best_model.pth"))
        gen_funcs = get_explanation_functions(model_A, model_B)
        run_evaluation(visualize_methods, X_ph2, cleaned_lesions, base_transform,
                       "1way_folder", "metrics_results_1WAY.csv", model_A, gen_funcs)
        run_evaluation(lambda img, mA, gf: visualize_combinations(img, mA, gf, n=2),
                       X_ph2, cleaned_lesions, base_transform,
                       "2way_folder", "metrics_results_2WAY.csv", model_A, gen_funcs)
        run_evaluation(lambda img, mA, gf: visualize_combinations(img, mA, gf, n=3),
                       X_ph2, cleaned_lesions, base_transform,
                       "3way_folder", "metrics_results_3WAY.csv", model_A, gen_funcs)
        print_top_two("metrics_results_1WAY.csv", label="PH2 1-Way")
        print_top_two("metrics_results_2WAY.csv", label="PH2 2-Way")
        print_top_two("metrics_results_3WAY.csv", label="PH2 3-Way")
    else:
        print("\n--- IMAGENET mode: Using student masks only ---")
        model_A = vit_LRP_new(pretrained=True, num_classes=3).to(device)
        model_B = vit_LRP(pretrained=True, num_classes=3).to(device)
        gen_funcs = get_explanation_functions(model_A, model_B)
    generate_LRP, generate_saliency, generate_rollout, generate_CAM = gen_funcs
    search_folder = r"C:\VitMix\Splits"
    mask_folder = r"C:\VitMix\Splits\masks"
    image_extensions = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
    all_images = []
    for rt,_,files in os.walk(search_folder):
        for ff in files:
            if ff.lower().endswith(image_extensions):
                all_images.append(os.path.join(rt,ff))
    image_to_masks = defaultdict(list)
    mask_files = [mf for mf in os.listdir(mask_folder) if '_' in mf.lower() and mf.lower().endswith('.png')]
    for mf in mask_files:
        base_part = mf.rsplit('_',1)[0]
        stud_part = mf.rsplit('_',1)[1].replace(".png","")
        possible_names = [base_part+ext for ext in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']]
        matched = next((im for im in all_images if os.path.basename(im).lower() in [pn.lower() for pn in possible_names]), None)
        if matched:
            image_to_masks[matched].append((os.path.join(mask_folder,mf), stud_part))
        else:
            print(f"No matching image found for mask: {mf}")
    print(f"\nWe have {len(image_to_masks)} images with at least one student mask.")
    filtered_dict = {}
    if eval_mode == 'medical':
        for k, v in image_to_masks.items():
            if os.path.basename(k).startswith('IMD'):
                filtered_dict[k] = v
    else:
        for k, v in image_to_masks.items():
            if os.path.basename(k).startswith('n'):
                filtered_dict[k] = v
    print(f"Filtered dictionary size for mode={eval_mode}: {len(filtered_dict)}")
    run_1way_student(filtered_dict)
    run_2way_student(filtered_dict)
    run_3way_student(filtered_dict)
else:
    print("\n--- VOC mode: Evaluating on VOC Segmentation Validation Data ---")
    model_A = vit_LRP_new(pretrained=True, num_classes=3).to(device)
    model_B = vit_LRP(pretrained=True, num_classes=3).to(device)
    gen_funcs = get_explanation_functions(model_A, model_B)
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    download_voc = not os.path.exists('VOCdevkit')
    voc_dataset = VOCSegmentation(root='VOCdevkit', year='2012', image_set='val', download=download_voc)
    run_voc_evaluation(visualize_methods, voc_dataset, base_transform,
                       "voc_1way_folder", "voc_metrics_results_1WAY.csv", model_A, gen_funcs)
    run_voc_evaluation(lambda img, mA, gf: visualize_combinations(img, mA, gf, n=2),
                       voc_dataset, base_transform,
                       "voc_2way_folder", "voc_metrics_results_2WAY.csv", model_A, gen_funcs)
    run_voc_evaluation(lambda img, mA, gf: visualize_combinations(img, mA, gf, n=3),
                       voc_dataset, base_transform,
                       "voc_3way_folder", "voc_metrics_results_3WAY.csv", model_A, gen_funcs)
    print_top_two("voc_metrics_results_1WAY.csv", label="VOC 1-Way")
    print_top_two("voc_metrics_results_2WAY.csv", label="VOC 2-Way")
    print_top_two("voc_metrics_results_3WAY.csv", label="VOC 3-Way")

print("\nScript finished. Check CSV files and output folders for evaluation details!")
