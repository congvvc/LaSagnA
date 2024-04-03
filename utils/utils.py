from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# about the input 
INPUT_CAT_START_TOKEN = "<cat_start>"
INPUT_CAT_END_TOKEN = "<cat_end>"
SEM_SEG_TASK_TOKEN = "<semantic>"

SEG_INPUT_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Segment this image.' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Please segment this image.' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Can you segment this image?' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please respond with segmentation mask." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please output segmentation mask." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please give segmentation result." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please visualize segmentation mask." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please respond with segmentation result." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please output segmentation result." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Segment all the objects in this image.' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Please segment all the objects in this image.' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + 'Can you segment all the objects in this image?' + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please respond with segmentation mask of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please output segmentation mask of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please give segmentation result of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please visualize segmentation mask of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please respond with segmentation result of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
    DEFAULT_IMAGE_TOKEN + "\n" + SEM_SEG_TASK_TOKEN + "What are in this image? Please output segmentation result of all the objects." + INPUT_CAT_START_TOKEN + " {all_class} " + INPUT_CAT_END_TOKEN,
   
]



# about the output
INS_TOKEN = "<ins>"
NO_INS_TOKEN = "<no_ins>"
OUTPUT_CAT_START_TOKEN = "<cat>"
OUTPUT_CAT_END_TOKEN = "</cat>"

SEG_INS_OUTPUT = "{class_name}" + INS_TOKEN
SEG_NO_INS_OUTPUT = "{class_name}" + NO_INS_TOKEN


SEG_ANSWER_LIST = [
    "The segmentation results are ",
    "The segmentations are ",
    "Sure, ",
    "Sure, the results are ",
    "Sure, the segmentation results are ",
    "Sure, the segmentations are ",
    "The results are ",
    "",
]


DET_ANSWER_LIST = [
    "The detection results are ",
    "The detections are ",
    "Sure, ",
    "Sure, the results are ",
    "Sure, the detection results are ",
    "Sure, the detections are ",
    "The results are ",
    "",
]

SHORT_QUESTION_LIST_DET = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect the {class_name} in image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you locate the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you locate the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you locate the {class_name} in image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you locate the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please locate the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please locate the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please locate the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please locate the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please find the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please find the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please find the {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with detection box.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with detection result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output detection box.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give detection box.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give detection result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output detection result.",
]

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Where is the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Where is the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {class_name} in image?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give segmentation result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation result.",
]

SEG_DET_SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect and segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect and segment the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you visualize the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you visualize the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Where is the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Where is the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you detect and segment the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you visualize the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you find the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect and segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please visualize the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect and segment the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect and segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please visualize the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Detect and segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Visualize the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Detect and segment the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Visualize the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Detect and segment the {class_name} in image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Visualize the {class_name} in image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the box and mask specifically for the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the mask specifically for the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the mask specifically for the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the segmentation and detection specifically for the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the segmentation mask and detection box specifically for the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "I need the visualization specifically for the {class_name} in image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please visualize detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please visualize detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please give detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output detection and segmentation result.",
]


LONG_QUESTION_LIST_DET = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection box.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection box in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection box in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection box in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection box in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with detection box?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with detection result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output detection box?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output detection result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find detection result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find detection box?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you visualize detection box?",

]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find segmentation mask?",

]

SEG_DET_LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection and segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection box and segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection and segmentation result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Visualize detection box and segmentation mask in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection and segmentation result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Find detection box and segmentation mask in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection and segmentation result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output detection box and segmentation mask in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection and segmentation result in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Respond with detection box and segmentation mask in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with detection box and segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with detection and segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output detection box and segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output detection and segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find detection and segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find detection box and segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you visualize detection box and segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please find segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize segmentation result.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please visualize segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you respond with segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you output segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find segmentation result?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you find segmentation mask?",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you visualize segmentation mask?",


]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
    "Please output segmentation mask and describe why.",
    "Please output segmentation mask and give the reason.",
    "Please output segmentation mask and make some explanation.",
    "Please respond with detection box and segmentation mask and explain why.",
    "Please respond with detection and segmentation result and explain why.",
    "Please output detection box and segmentation mask and explain why.",
    "Please output detection and segmentation result and explain why.",
    "Please find detection and segmentation result and explain why.",
    "Please find detection box and segmentation mask and explain why.",
    "Please visualize detection and segmentation result and explain why.",
    "Please visualize detection box and segmentation mask and explain why.",
    "Respond with detection box and segmentation mask and explain why.",
    "Respond with detection and segmentation result and explain why.",
    "Output detection box and segmentation mask and explain why.",
    "Output detection and segmentation result and explain why.",
    "Find detection and segmentation result and explain why.",
    "Find detection box and segmentation mask and explain why.",
    "Visualize detection and segmentation result and explain why.",
    "Visualize detection box and segmentation mask and explain why.",
    "Visualize detection and segmentation result in this image and explain why.",
    "Visualize detection box and segmentation mask in the image and explain why.",
    "Find detection and segmentation result in this image and explain why.",
    "Find detection box and segmentation mask in the image and explain why.",
    "Output detection and segmentation result in this image and explain why.",
    "Output detection box and segmentation mask in the image and explain why.",
    "Respond with detection and segmentation result in this image and explain why.",
    "Respond with detection box and segmentation mask in the image and explain why.",
    "Please respond with detection box and segmentation mask and explain the reason.",
    "Please respond with detection and segmentation result and explain the reason.",
    "Please output detection box and segmentation mask and explain the reason.",
    "Please output detection and segmentation result and explain the reason.",
    "Please find detection and segmentation result and explain the reason.",
    "Please find detection box and segmentation mask and explain the reason.",
    "Please visualize detection and segmentation result and explain the reason.",
    "Please visualize detection box and segmentation mask and explain the reason.",
    "Respond with detection box and segmentation mask and explain the reason.",
    "Respond with detection and segmentation result and explain the reason.",
    "Output detection box and segmentation mask and explain the reason.",
    "Output detection and segmentation result and explain the reason.",
    "Find detection and segmentation result and explain the reason.",
    "Find detection box and segmentation mask and explain the reason.",
    "Visualize detection and segmentation result and explain the reason.",
    "Visualize detection box and segmentation mask and explain the reason.",
    "Visualize detection and segmentation result in this image and explain the reason.",
    "Visualize detection box and segmentation mask in the image and explain the reason.",
    "Find detection and segmentation result in this image and explain the reason.",
    "Find detection box and segmentation mask in the image and explain the reason.",
    "Output detection and segmentation result in this image and explain the reason.",
    "Output detection box and segmentation mask in the image and explain the reason.",
    "Respond with detection and segmentation result in this image and explain the reason.",
    "Respond with detection box and segmentation mask in the image and explain the reason.",
    "Please respond with detection box and segmentation mask give some explanation.",
    "Please respond with detection and segmentation result give some explanation.",
    "Please output detection box and segmentation mask give some explanation.",
    "Please output detection and segmentation result give some explanation.",
    "Please find detection and segmentation result give some explanation.",
    "Please find detection box and segmentation mask give some explanation.",
    "Please visualize detection and segmentation result give some explanation.",
    "Please visualize detection box and segmentation mask give some explanation.",
    "Respond with detection box and segmentation mask give some explanation.",
    "Respond with detection and segmentation result give some explanation.",
    "Output detection box and segmentation mask give some explanation.",
    "Output detection and segmentation result give some explanation.",
    "Find detection and segmentation result give some explanation.",
    "Find detection box and segmentation mask give some explanation.",
    "Visualize detection and segmentation result give some explanation.",
    "Visualize detection box and segmentation mask give some explanation.",
    "Visualize detection and segmentation result in this image give some explanation.",
    "Visualize detection box and segmentation mask in the image give some explanation.",
    "Find detection and segmentation result in this image give some explanation.",
    "Find detection box and segmentation mask in the image give some explanation.",
    "Output detection and segmentation result in this image give some explanation.",
    "Output detection box and segmentation mask in the image give some explanation.",
    "Respond with detection and segmentation result in this image give some explanation.",
    "Respond with detection box and segmentation mask in the image give some explanation.",
    "Please respond with detection box and segmentation mask and give the reason.",
    "Please respond with detection and segmentation result and give the reason.",
    "Please output detection box and segmentation mask and give the reason.",
    "Please output detection and segmentation result and give the reason.",
    "Please find detection and segmentation result and give the reason.",
    "Please find detection box and segmentation mask and give the reason.",
    "Please visualize detection and segmentation result and give the reason.",
    "Please visualize detection box and segmentation mask and give the reason.",
    "Respond with detection box and segmentation mask and give the reason.",
    "Respond with detection and segmentation result and give the reason.",
    "Output detection box and segmentation mask and give the reason.",
    "Output detection and segmentation result and give the reason.",
    "Find detection and segmentation result and give the reason.",
    "Find detection box and segmentation mask and give the reason.",
    "Visualize detection and segmentation result and give the reason.",
    "Visualize detection box and segmentation mask and give the reason.",
    "Visualize detection and segmentation result in this image and give the reason.",
    "Visualize detection box and segmentation mask in the image and give the reason.",
    "Find detection and segmentation result in this image and give the reason.",
    "Find detection box and segmentation mask in the image and give the reason.",
    "Output detection and segmentation result in this image and give the reason.",
    "Output detection box and segmentation mask in the image and give the reason.",
    "Respond with detection and segmentation result in this image and give the reason.",
    "Respond with detection box and segmentation mask in the image and give the reason.",
    "Please respond with detection box and segmentation mask and make some explanation.",
    "Please respond with detection and segmentation result and make some explanation.",
    "Please output detection box and segmentation mask and make some explanation.",
    "Please output detection and segmentation result and make some explanation.",
    "Please find detection and segmentation result and make some explanation.",
    "Please find detection box and segmentation mask and make some explanation.",
    "Please visualize detection and segmentation result and make some explanation.",
    "Please visualize detection box and segmentation mask and make some explanation.",
    "Respond with detection box and segmentation mask and make some explanation.",
    "Respond with detection and segmentation result and make some explanation.",
    "Output detection box and segmentation mask and make some explanation.",
    "Output detection and segmentation result and make some explanation.",
    "Find detection and segmentation result and make some explanation.",
    "Find detection box and segmentation mask and make some explanation.",
    "Visualize detection and segmentation result and make some explanation.",
    "Visualize detection box and segmentation mask and make some explanation.",
    "Visualize detection and segmentation result in this image and make some explanation.",
    "Visualize detection box and segmentation mask in the image and make some explanation.",
    "Find detection and segmentation result in this image and make some explanation.",
    "Find detection box and segmentation mask in the image and make some explanation.",
    "Output detection and segmentation result in this image and make some explanation.",
    "Output detection box and segmentation mask in the image and make some explanation.",
    "Respond with detection and segmentation result in this image and make some explanation.",
    "Respond with detection box and segmentation mask in the image and make some explanation.",

]

ANSWER_LIST = [
    "It is <ins>.",
    "Sure, <ins>.",
    "Sure, it is <ins>.",
    "The result is <ins>.",
    "Sure, the result is <ins>.",
    "The segmentation result is <ins>.",
    "The segmentation result is <ins>.",
    "The segmentation is <ins>.",
    "The result is <ins>.",
    "The segmentation mask is <ins>.",
    "Sure, the segmentation result is <ins>.",
    "Sure, segmentation is <ins>.",
    "Sure, the segmentation is <ins>.",
    "Sure, the segmentation mask is <ins>.",
    "<ins>.",
]

SEG_DET_ANSWER_LIST = [
    "It is <ins>.",
    "Sure, <ins>.",
    "Sure, it is <ins>.",
    "The result is <ins>.",
    "Sure, the result is <ins>.",
    "The detection and segmentation result is <ins>.",
    "The segmentation result is <ins>.",
    "The detection and segmentation is <ins>.",
    "The detection result is <ins>.",
    "The detection box and segmentation mask is <ins>.",
    "Sure, the detection and segmentation result is <ins>.",
    "Sure, the segmentation result is <ins>.",
    "Sure, the detection result is <ins>.",
    "Sure, the detection and segmentation is <ins>.",
    "Sure, the segmentation is <ins>.",
    "Sure, the detection box and segmentation mask is <ins>.",
    "<ins>.",
]

ANSWER_LIST_DET = [
    "It is <ins>.",
    "Sure, <ins>.",
    "Sure, it is <ins>.",
    "The result is <ins>.",
    "Sure, the result is <ins>.",
    "The detection result is <ins>.",
    "The segmentation result is <ins>.",
    "The segmentation is <ins>.",
    "The detection box is <ins>.",
    "Sure, the detection result is <ins>.",
    "Sure, the segmentation result is <ins>.",
    "Sure, the detection result is <ins>.",
    "Sure, the detection is <ins>.",
    "Sure, the segmentation is <ins>.",
    "Sure, the detection box is <ins>.",
    "<ins>.",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
