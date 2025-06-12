from leap_binder import (preprocess_func, input_encoder, gt_encoder, custom_loss,
                         pred_visualizer, gt_visualizer, metadata_image_path)
# import tensorflow as tf
import numpy as np
from utils.general import non_max_suppression
from leap_utils import bb_array_to_object, draw_image_with_boxes
import torch
from code_loader.contract.visualizer_classes import LeapImageWithBBox
import matplotlib
# matplotlib.use('Tkagg')
import onnx
import onnxruntime as ort

from code_loader.default_metrics import flatten_non_batch_dims
from leap_binder import leap_binder

def check_integration():

    image_index = 1 #100
    leap_binder.check()
    nc = 3
    model_path = '/mnt/Data/hanoch/runs/train/yolov71351/weights/epoch_049.onnx'
    res = preprocess_func()
    for train in res:
        for image_index in range(10):
            inpt = input_encoder(image_index, train)[None, ...]
            if 1:
                cuda = [True if torch.cuda.is_available() else False][0]
                if 0:
                    providers = [('TensorrtExecutionProvider', {'device_id': 0}), 'CUDAExecutionProvider'] if cuda else [
                        'CPUExecutionProvider']
                else:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
                # if isinstance(opt.weights, list):
                #     opt.weights = opt.weights[0]

                session = ort.InferenceSession(model_path, providers=providers)
                input_shape = session.get_inputs()[0].shape
                print('Input shape : ', input_shape)
                outname = [i.name for i in session.get_outputs()]
                inname = [i.name for i in session.get_inputs()]
                inp = {inname[0]: inpt}
                prediction = session.run(outname, inp)[0][None,...]
                gt = gt_encoder(image_index, train)[None, ...]

                meta_dat = metadata_image_path(image_index, train)
                loss = custom_loss(prediction, gt)

                gt_img_with_bbox = gt_visualizer(gt, inpt)
                img_with_bbox = pred_visualizer(prediction, inpt)



if __name__ == '__main__':
    check_integration()
