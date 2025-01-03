import sys
# set package path
sys.path.append("/content/CardiacSegV2_gradcam")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
from functools import partial

import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from data_utils.data_loader_utils import load_data_dict_json
from data_utils.dataset import get_infer_data
from data_utils.io import load_json
from runners.inferer import run_infering
from runners.inferer import run_infering_with_gradcam
from networks.network import network

from expers.args import get_parser

from process_img import process_and_save



def main():
    args = get_parser(sys.argv[1:])
    main_worker(args)
    
    
def is_deep_sup(checkpoint):
    for key in list(checkpoint["state_dict"].keys()):
        if 'ds' in key:
            return True
    return False

def main_worker(args):
    # make dir
    os.makedirs(args.infer_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # model
    model = network(args.model_name, args)



    # check point
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        if is_deep_sup(checkpoint) and args.model_name != 'cotr':
            # load check point epoch and best acc
            print("Tag 'ds (deeply supervised)' found in state dict - fixing!")
            for key in list(checkpoint["state_dict"].keys()):
                if 'ds' in key:
                    checkpoint["state_dict"].pop(key) 
        
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        
        print(
          "=> loaded checkpoint '{}')"\
          .format(args.checkpoint)
        )


    pred_img = []
    # 遍歷所有原始影像，進行處理並更新清單
    for root, dirs, files in os.walk("/content/drive/MyDrive/myo_pred/mwhs/image", topdown=False):
        for name in files:
            original_path = os.path.join(root, name)
            # print(f"[INFO] Found image: {original_path}")

            # 設置處理後影像的輸出路徑
            processed_img_pth = f"/content/drive/MyDrive/myo_pred/mwhs/infer/processed_{name}"

            # 處理影像
            process_and_save(original_path, processed_img_pth)

            # 更新處理後的影像到清單中
            pred_img.append(processed_img_pth)
            # print(f"[INFO] Processed image added to list: {processed_img_pth}")
    # 檢查結果
    # print(f"[DEBUG] All processed images: {pred_img}")

    # 使用處理後的影像進行推論
    for processed_img_pth in pred_img:
        args.img_pth = processed_img_pth
        # print(f"[INFO] Updated args.img_pth to: {args.img_pth}")


    # inferer
    keys = ['pred']
    if args.data_name == 'mmwhs' or args.data_name == 'mmwhs2':
        axcodes = 'LAS'
    else:
        axcodes = 'LPS'
    # axcodes = 'RAS'
    post_transform = Compose([
        Orientationd(keys=keys, axcodes=axcodes),
        ToNumpyd(keys=keys),
        Restored(keys=keys, ref_image="image")
    ])
    
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )



    # prepare data_dict
    if args.data_dicts_json and args.data_name != 'mmwhs':
        data_dicts = load_data_dict_json(args.data_dir, args.data_dicts_json)
    elif args.data_dicts_json and args.data_name == 'mmwhs':
        data_dicts = load_json(args.data_dicts_json)
    else:
        if args.lbl_pth is not None:
            data_dicts = [{
                'image': args.img_pth,
                'label': args.lbl_pth
            }]
        else:
            data_dicts = [{
                'image': args.img_pth,
            }]

    # 列印 data_dicts 確認內容
    # print(f"[DEBUG] Data dicts: {data_dicts}")

    # # run infer
    # for data_dict in data_dicts:
    #     print('infer data:', data_dict)
      
    #     # load infer data
    #     data = get_infer_data(data_dict, args)

    #     # infer
    #     run_infering(
    #         model,
    #         data,
    #         model_inferer,
    #         post_transform,
    #         args
    #     )

    # run infer with gradcam



    target_layers = [model.decoder1.conv_block.conv3]
    for data_dict in data_dicts:

        # print('infer data:', data_dict)
      
        # load infer data
        data = get_infer_data(data_dict, args)
        # print(f"[INFO] Loaded image shape: {data['image'].shape}")

        # infer
        run_infering_with_gradcam(
            model,
            data,
            model_inferer,
            post_transform,
            args,
            target_layers
        )
    


if __name__ == "__main__":
    main()