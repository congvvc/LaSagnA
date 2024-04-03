import argparse
import os
import shutil
import sys
import time
from functools import partial

from datetime import timedelta
from fuzzywuzzy import process

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distributed
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LaSagnA import LaSagnAForCausalLM
from model.llava import conversation as conversation_lib
from utils.miou import IoU

from utils.test_sem_seg_dataset import TestSemSegDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX,
                        INPUT_CAT_START_TOKEN, INPUT_CAT_END_TOKEN, SEM_SEG_TASK_TOKEN, 
                        INS_TOKEN,  NO_INS_TOKEN, OUTPUT_CAT_START_TOKEN, OUTPUT_CAT_END_TOKEN,
                        AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                        intersectionAndUnionGPU)

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'



def parse_args(args):
    parser = argparse.ArgumentParser(description="LaSagnA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="./checkpoint/LLaVA-Lightning-7B-v1-1"
    )
    parser.add_argument("--vis_save_path", default="", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size") 
    parser.add_argument("--test_class_internal", default=2, type=int) 
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="./checkpoint/CLIP/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--use_all_classes", action="store_true", default=True)
    parser.add_argument("--random_test", action="store_true", default=False)

    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str) #  refcoco|unc|val  refcoco+|unc|val  refcocog|umd|val
    parser.add_argument("--val_dataset_list", default='cityscapes,ade20k,cocostuff', type=str) #  cityscapes,ade20k,cocostuff,mapillary
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./output/log", type=str)
    parser.add_argument("--exp_name", default="LaSagnA-7b", type=str)

    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--num_all_classes", default=100, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--final_eval", action="store_true", default=True)
    parser.add_argument("--vision_pretrained", default="./SAM/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # num_added_tokens = tokenizer.add_tokens("[SEG]")
    # args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    
    tokenizer.add_tokens(
            [INPUT_CAT_START_TOKEN, INPUT_CAT_END_TOKEN, SEM_SEG_TASK_TOKEN], special_tokens=True
        )
    

    num_added_tokens_ins = tokenizer.add_tokens(INS_TOKEN)
    num_added_tokens_no_ins = tokenizer.add_tokens(NO_INS_TOKEN)
    # 获取新添加标记的索引
    
    args.ins_token_idx = tokenizer(INS_TOKEN, add_special_tokens=False).input_ids[0]
    args.no_ins_token_idx = tokenizer(NO_INS_TOKEN, add_special_tokens=False).input_ids[0]


    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "ins_token_idx": args.ins_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    if args.local_rank == 0:
        print('----------begin load main model---------')
    model = LaSagnAForCausalLM.from_pretrained(args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    model.get_model().initialize_LaSagnA_modules(model.get_model().config)


    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "box_decoder"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "box_decoder",]
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True


    train_dataset = None
    
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    
    args.val_dataset_list = args.val_dataset_list.split(',')
    val_dataset_list = []
    
    for k in range(len(args.val_dataset_list)):
        val_dataset = TestSemSegDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.image_size,
            100,
            args.test_class_internal,
            args.val_dataset_list[k],
            args.use_all_classes,
            args.random_test,
            )
        val_dataset_list.append(val_dataset)


    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
 


    if args.resume:
        # load_path, client_state = model_engine.load_checkpoint(args.resume)
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(model, args.resume)
        model = model.merge_and_unload()
    timeout = timedelta(days=1)

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=None,
        config=ds_config,
        timeout=timeout,
    )



    # validation dataset
    val_loader_list = []
    if val_dataset is not None:
        assert args.val_batch_size == 1
        for k in range(len(args.val_dataset_list)):
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_list[k], shuffle=False, drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                val_dataset_list[k],
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,),)
            val_loader_list.append(val_loader)
    
    if args.eval_only:
        for k in range(len(args.val_dataset_list)):
            iou, miou = validate(val_loader_list[k], model_engine, args, args.val_dataset_list[k], tokenizer) # ade20k:150, coco:182
        exit()


def get_class_name(text):
    SEG_ANSWER_LIST = [
    "The segmentation results are ",
    "The segmentations are ",
    "Sure, the results are ",
    "Sure, the segmentation results are ",
    "Sure, the segmentations are ",
    "The results are ",
    "Sure, "]

    for seg_ans in SEG_ANSWER_LIST:
        text = text.replace(seg_ans, "")

    text = text.replace("\n", "").replace("  ", " ")
    import re
    match = re.search(r'ASSISTANT:(.+)', text)
    if match:
        a1 = match.group(1).strip()
    else:
        a1 = ''
    a1 = a1.replace("</s>", '').replace(".", '')
    # Extract nouns before each '[INS]' and store in list 'b'
    b = re.findall(r'([^\<\>]+)(?=\s*\<ins\>)', a1)
    if b != []:
        b = [kk.replace(" ", "") for kk in b]
    return b



def semantic_inference(origin_mask_cls, origin_mask_pred, all_classes, num_classes):


    mask_cls, mask_pred = origin_mask_cls, origin_mask_pred
    mask_labels = []                         
    for c in mask_cls:
        best_match = process.extractOne(c, all_classes)
        mask_labels.append(all_classes.index(best_match[0]))
    labels_per_image = mask_labels
    mask_pred[mask_pred<=0] = 0
    # pred_masks = mask_pred > 0
    # K, H, W
    result = torch.zeros((num_classes, mask_pred.size(1), mask_pred.size(2)), device=origin_mask_pred.device)

    for idx, cur_label in enumerate(labels_per_image):
        result[cur_label] = mask_pred[idx]

    # result = result.float()
    # result = result.argmax(dim=0)
    
    return result

def validate(val_loader, model, args, val_name, tokenizer):
    # intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    # union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    # acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    # ade20k||cocostuff||cityscapes||mapillary
    if val_name == 'ade20k':
        num_classes = 150
    elif val_name == 'cocostuff':
        num_classes = 171
    elif val_name == 'cityscapes':
        num_classes = 19
    elif val_name == 'mapillary':
        num_classes = 65
    
    miou_metric = IoU(num_classes, ignore_index=None)
    
    model.eval()
    
    miou_metric.reset()
    unk_id = tokenizer.unk_token_id

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                input_dict['offset'],
                input_dict['images_clip'],
                input_dict['images'],
                input_dict['input_ids'],
                input_dict['resize_list'],
                [k.shape for k in input_dict['label_list']],
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
        gt_classes_index = input_dict['classes_index'][0]
        gt_classes_name = input_dict['classes_name'][0]
        all_classes = input_dict['all_classes'][0]

        pred_class_name = []

        for k in range(output_ids.shape[0]):
            cur_output_ids = output_ids[k][output_ids[k] != IMAGE_TOKEN_INDEX]
            cur_output_ids = cur_output_ids[cur_output_ids != unk_id]
            text_output = tokenizer.decode(cur_output_ids, skip_special_tokens=False)
            cur_pre_class = get_class_name(text_output)
            if cur_pre_class != []:
                pred_class_name.extend(cur_pre_class)
        print(f'pred classes is {pred_class_name}')
        print(f'gt classes is {gt_classes_name}')
        masks_list = input_dict['masks_list'][0]
        masks_list = masks_list.bool()
        output_list = pred_masks[0]


        gt_masks = torch.zeros((num_classes, masks_list.size(1), masks_list.size(2)), device=masks_list.device).bool()
        for idx, cur_label in enumerate(gt_classes_index):
            gt_masks[cur_label] = gt_masks[cur_label] | masks_list[idx]

        gt_masks = gt_masks.float()

        # (N, K, H, W) tensor of integer values between 0 and K-1.
        gt_masks = gt_masks.unsqueeze(0)
        # true_output_list = torch.stack([true_output_list[k] * gt_classes_index[k] for k in range(len(gt_classes_index))], dim=0).unsqueeze(0)

        # (N, K, H, W)
        true_output_list = semantic_inference(pred_class_name, output_list, all_classes, num_classes).unsqueeze(0)

        miou_metric.add(true_output_list.detach(), gt_masks.detach())


    iou, miou = miou_metric.value()

    if args.local_rank == 0:
        print(f'----------dataset {val_name} test result-----------')
        print("miou: {:.4f}".format(miou))

    return iou, miou


if __name__ == "__main__":
    main(sys.argv[1:])
