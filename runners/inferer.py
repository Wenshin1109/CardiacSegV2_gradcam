import os
import time
import importlib
from pathlib import PurePath

import torch

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    AddChannel,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric,get_confusion_matrix, compute_confusion_matrix_metric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def check_channel(inp):
    # check shape is 5
    add_ch = AddChannel()
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 4:
        inp = add_ch(inp)
    if len_inp_shape == 3:
        inp = add_ch(inp)
        inp = add_ch(inp)
    return inp


def eval_label_pred(data, cls_num, device):
    # post transform
    post_label = AsDiscrete(to_onehot=cls_num)
    
    # metric
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )
    
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
        get_not_nans=False
    )
    
    confusion_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name="sensitivity", 
        compute_sample=False, 
        reduction="mean", 
        get_not_nans=False
    )
    
    # batch data
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    
    # check shape is 5
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # deallocate batch data
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    val_output_convert = [
        post_label(val_pred_tensor) for val_pred_tensor in val_pred
    ]
    
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    hd95_metric(y_pred=val_output_convert, y=val_labels_convert)
    confusion_metric(y_pred=val_output_convert, y=val_labels_convert)

    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    hd95_vals = hd95_metric.get_buffer().detach().cpu().numpy().squeeze()
    
    confusion_vals = confusion_metric.get_buffer().detach().cpu().numpy().squeeze()
    tp = confusion_vals[0]
    fp = confusion_vals[1]
    tn = confusion_vals[2]
    fn = confusion_vals[3]
    sensitivity_vals = tp / (tp + fn)
    specificity_vals = tn / (tn + fp)
    
    
    return dc_vals, hd95_vals, sensitivity_vals, specificity_vals


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    
    
    # test
    start_time = time.time()
    data['pred'] = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # post process transform
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])
    
    # eval infer tta
    if 'label' in data.keys():
        tta_dc_vals, tta_hd95_vals, _ , _ = eval_label_pred(data, args.out_channels, args.device)
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('hd95:', tta_hd95_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_hd'] = tta_hd95_vals
        
        # post label transform 
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)
    
    # post transform
    data = post_transform(data)
    
    # eval infer origin
    if 'label' in data.keys():
        # get orginal label
        lbl_dict = {'label': data['label_meta_dict']['filename_or_obj']}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data['label']
        
        ori_dc_vals, ori_hd95_vals, ori_sensitivity_vals, ori_specificity_vals = eval_label_pred(data, args.out_channels, args.device)
        print('infer test original:')
        print('dice:', ori_dc_vals)
        print('hd95:', ori_hd95_vals)
        print('sensitivity:', ori_sensitivity_vals)
        print('specificity:', ori_specificity_vals)
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_hd'] = ori_hd95_vals
        ret_dict['ori_sensitivity'] = ori_sensitivity_vals
        ret_dict['ori_specificity'] = ori_specificity_vals
    
    if args.data_name == 'mmwhs':
        mmwhs_transform = Compose([
            LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
            MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                            target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
            # AddChannel(),
            # Spacing(
            #     pixdim=(args.space_x, args.space_y, args.space_z),
            #     mode=("nearest"),
            # ),
            # SqueezeDim()
        ])
        data['pred'] = mmwhs_transform(data['pred'])
        
    
    if not args.test_mode:
        # save pred result
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)

        save_img(
          data['pred'], 
          data['pred_meta_dict'], 
          infer_img_pth
        )
        
    return ret_dict


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch.nn.functional as F
from torch.cuda.amp import autocast


def align_tensor(tensor, target_shape):
    """
    調整張量的空間大小以符合目標形狀。
    """
    print(f"Aligning tensor from {tensor.shape} to {target_shape}...")

    # 計算需要補零的大小
    padding = [
        0, max(target_shape[4] - tensor.shape[4], 0),  # W
        0, max(target_shape[3] - tensor.shape[3], 0),  # H
        0, max(target_shape[2] - tensor.shape[2], 0),  # D
    ]
    aligned_tensor = F.pad(tensor, padding)
    print(f"Aligned tensor shape: {aligned_tensor.shape}")

    return aligned_tensor


from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

def run_infering_with_gradcam(
        model, data, model_inferer, post_transform, args, target_layers, batch_size=1
    ):
    

    ret_dict = {}

    torch.cuda.empty_cache()

    # 確保模型的所有參數啟用梯度計算
    for param in model.parameters():
        param.requires_grad = True
    for param in model.decoder1.conv_block.cbam.parameters():
      if param.grad is None:
          print(f"[DEBUG] Gradient for CBAM is None. Ensure backward pass is performed correctly.")

    # Grad-CAM visualization
    cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = data['image'].to(args.device).requires_grad_(True)
    # 確保影像形狀符合要求
    target_shape = (input_tensor.shape[0], input_tensor.shape[1], 128, 128, 128)
    if input_tensor.shape != target_shape:
        print(f"Mismatch detected: {input_tensor.shape} vs {target_shape}")
        input_tensor = align_tensor(input_tensor, target_shape)

    num_samples = input_tensor.shape[0]
    print(f"[INFO] Total samples: {num_samples}, Batch size: {batch_size}")

    for i in range(0, num_samples, batch_size):
        batch_input = input_tensor[i:i + batch_size]
        
        # 提供有效的 mask
        mask_shape = batch_input.shape[2:]  # (D, H, W)
        mask = np.ones(mask_shape, dtype=np.float32)  # 全 1 的 mask
        # 設置目標類別為 1，選擇空間位置
        targets = [SemanticSegmentationTarget(category=1, mask=mask)]  # mask 可用於指定特定空間位置

        with autocast():  # 混合精度執行
            grayscale_cam = cam(input_tensor=batch_input, targets=targets)

        for j, cam_result in enumerate(grayscale_cam):
            idx = i + j
            original_image = batch_input[j, 0].cpu().numpy()
            normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            cam_image = show_cam_on_image(normalized_image, cam_result, use_rgb=True)

            save_path = os.path.join(args.infer_dir, f"grad_cam_visualization_{idx}.png")
            cv2.imwrite(save_path, cam_image)
            print(f"[INFO] Grad-CAM visualization saved at {save_path}")

        torch.cuda.empty_cache()


    return ret_dict
