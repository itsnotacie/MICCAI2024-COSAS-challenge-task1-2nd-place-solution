import json
import os

import warnings
import numpy as np
# os.system("pwd")
# os.system("ls /output/")

import torch
import cv2
# from mmseg.apis import inference_model, init_model
from fire import initFire, FireModel, FireRunner, FireData
from fire.runner import write

from config import cfg

print(torch.cuda.current_device())

# def main():
#     input_root = '/input/images/adenocarcinoma-image'
#     output_root = '/output/images/adenocarcinoma-mask'

#     if not os.path.exists(output_root):
#         os.makedirs(output_root)

#     initFire(cfg)
#     cfg['img_size'] = [1376,1376]
#     model = FireModel(cfg)

#     data = FireData(cfg)
#     # data.showTrainData()
#     # b
    
#     test_loader = data.getTestDataloader()


#     runner = FireRunner(cfg, model)

#     #print(model)
#     runner.modelLoad(cfg['model_path'])


#     runner.predict(test_loader,output_root)


### TTA
def main():
    input_root = '/input/images/adenocarcinoma-image'
    output_root = '/output/images/adenocarcinoma-mask'

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    initFire(cfg)

    model = FireModel(cfg)
    data = FireData(cfg)

    
    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    test_loader = data.getTestDataloader()
    res_dict0 = runner.predictRaw(test_loader)

    # test_loader = data.getTestDataloaderTTA1()
    # res_dict1 = runner.predictRaw(test_loader)

    test_loader = data.getTestDataloaderTTA2()
    res_dict2 = runner.predictRaw(test_loader)

    cfg['img_size'] = [1376,1376]
    data = FireData(cfg)
    test_loader = data.getTestDataloader()
    res_dict3 = runner.predictRaw(test_loader)

    res_dict = {}
    for basename,mask in res_dict0.items():

        mask1 = res_dict2[basename]
        mask1 = cv2.flip(mask1,1)

        mask2 = res_dict3[basename]

        mask = (mask+mask1+mask2)/3
        mask = np.where(mask <= 0.5, 0, 1).astype(np.uint8)

        save_path = os.path.join(output_root, basename)
        write(save_path, mask)


if __name__ == '__main__':
    main()