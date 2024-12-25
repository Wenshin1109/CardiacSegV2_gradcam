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
import nibabel as nib

def run_infering_with_gradcam(
        model, data, model_inferer, post_transform, args, target_layers, batch_size=1
    ):
    

    ret_dict = {}

    torch.cuda.empty_cache()

    # 確保模型的所有參數啟用梯度計算
    for param in model.parameters():
        param.requires_grad = True
    for param in model.decoder1.conv_block.conv3.parameters():
      param.requires_grad = True
      if param.grad is None:
          print(f"[DEBUG] Gradient for conv3 is None. Ensure backward pass is performed correctly.")

    # Grad-CAM visualization
    # cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = data['image'].to(args.device).requires_grad_(True)
    # 確保影像形狀符合要求
    target_shape = (input_tensor.shape[0], input_tensor.shape[1], 128, 128, 128)
    if input_tensor.shape != target_shape:
        print(f"Mismatch detected: {input_tensor.shape} vs {target_shape}")
        input_tensor = align_tensor(input_tensor, target_shape)

    num_samples = input_tensor.shape[0]
    print(f"[INFO] Total samples: {num_samples}, Batch size: {batch_size}")

    # model.train()

    activations = []
    gradients = []

    # 前向勾子
    def forward_hook(module, input, output):
        activations.append(output)

    # 新的完整反向勾子
    def backward_hook(module, grad_input, grad_output):
        # 放大梯度
        scaled_grad = grad_output[0] * 1e8  # 將梯度放大 1e8 倍
        gradients.append(scaled_grad)

    # 註冊前向和完整反向勾子
    target_layer = model.decoder1.conv_block.conv3
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    for i in range(0, num_samples, batch_size):
        batch_input = input_tensor[i:i + batch_size]

        # 檢查激活與梯度
        output = model(input_tensor)  # 執行正向傳播
        scalar_output = output.mean()
        scalar_output.backward()  # 執行反向傳播

        # 檢查目標層的激活與梯度
        if len(activations) == 0 or len(gradients) == 0:
            raise ValueError("[ERROR] Activations or gradients not captured for the target layer.")
        else:
            print(f"[DEBUG] Activation min: {activations[0].min()}, max: {activations[0].max()}")
            print(f"[DEBUG] Gradient min: {gradients[0].min()}, max: {gradients[0].max()}")

        
        # # 提供有效的 mask
        # mask_shape = batch_input.shape[2:]  # (D, H, W)
        # mask = np.ones(mask_shape, dtype=np.float32)  # 全 1 的 mask
        # # 設置目標類別為 1，選擇空間位置
        # targets = [SemanticSegmentationTarget(category=1, mask=mask)]  # mask 可用於指定特定空間位置

        # 計算 Grad-CAM 權重
        weights = gradients[0].mean(axis=(2, 3, 4))  # 沿空間維度計算均值
        print(f"[DEBUG] Weights shape: {weights.shape}")
        # 擴展權重以匹配激活圖形狀
        weights = weights.view(1, 48, 1, 1, 1)  # 形狀為 (1, 48, 1, 1, 1)
        # 加權激活圖
        weighted_activations = activations[0] * weights

        # with autocast():  # 混合精度執行
            # grayscale_cam = cam(input_tensor=batch_input, targets=targets)
        # grayscale_cam = weighted_activations.sum(axis=1)  # 沿通道維度求和
        grayscale_cam = weighted_activations.sum(axis=1).detach().cpu().numpy()  # 沿通道維度求和
        
        # 檢查輸出結果
        print(f"[INFO] Grayscale CAM shape: {grayscale_cam.shape}")
        print(f"[DEBUG] Grayscale CAM - min: {grayscale_cam.min()}, max: {grayscale_cam.max()}")

        # 儲存 Grayscale CAM 為 3D NIfTI
        save_path_3d = os.path.join(args.infer_dir, f"grad_cam_3d_visualization_{i}.nii.gz")
        nii_img = nib.Nifti1Image(grayscale_cam, affine=np.eye(4))  # 單位仿射矩陣
        nib.save(nii_img, save_path_3d)
        print(f"[INFO] 3D Grad-CAM visualization saved at {save_path_3d}")


        # for j, cam_result in enumerate(grayscale_cam):
        #     idx = i + j
            

        #     # 將 cam_result 從張量轉換為 NumPy 陣列
        #     mask = cam_result.detach().cpu().numpy().astype(np.float32)
        #     print(f"Before normalization - mask type: {mask.dtype}, shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")

        #     # 檢查分母是否為 0 並進行正規化
        #     if mask.max() - mask.min() == 0:
        #         print("[WARNING] mask has no variation, skipping normalization.")
        #         mask = np.zeros_like(mask, dtype=np.uint8)  # 設置為全 0
        #     else:
        #         # 正規化並轉換為 uint8
        #         mask = (mask - mask.min()) / (mask.max() - mask.min())
        #         mask = np.uint8(255 * mask)

        #     # 檢查無效值
        #     if np.isnan(mask).any() or np.isinf(mask).any():
        #         print("[ERROR] Detected invalid values in mask, replacing with zeros.")
        #         mask = np.zeros_like(mask, dtype=np.uint8)

        #     # 確保 mask 是單通道
        #     if len(mask.shape) > 2:
        #         mask = mask[:, :, 0]
            
        #     print(f"After normalization - mask type: {mask.dtype}, shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")

        #     # 使用 OpenCV 應用顏色映射
        #     heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        #     # 確保移除梯度計算關聯
        #     original_image = batch_input[j, 0].detach().cpu().numpy()
        #     normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            
        #     cam_image = show_cam_on_image(normalized_image, heatmap, use_rgb=True)
            
        #     # 儲存結果
        #     save_path = os.path.join(args.infer_dir, f"grad_cam_visualization_{idx}.png")
        #     cv2.imwrite(save_path, cam_image)
        #     print(f"[INFO] Grad-CAM visualization saved at {save_path}")

        torch.cuda.empty_cache()


    return ret_dict
