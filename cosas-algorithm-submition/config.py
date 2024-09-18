# @https://github.com/fire717/Fire

cfg = {
    ### Global Set
    "model_name": "unetplusplus",    #unetplusplus  STDC
    "backbone": "efficientnet-b6",  #efficientnet-b3 STDCNet1446  STDCNet813
    'GPU_ID': '0',
    'full_train':False,

    "class_number": 1,
    "random_seed":42,
    "cfg_verbose":True,
    "num_workers":0,

    ### Train Setting
    'train_path':"../COSAS24-TrainingSet/task1",
    'pretrained':"imagenet", # "imagenet" local path or ''

    'try_to_train_items': 0,   # 0 means all, or run part(200 e.g.) for bug test
    'save_best_only': True,  #only save model if better than before
    'save_one_only':True,    #only save one best model (will del model before)
    "save_dir": "/output/",
    'metrics': ['acc'], # default is acc,  can add F1  ...
    "loss": 'CE', # CE, CEV2-0.5, Focalloss-1 ...STDCLoss

    'show_data':False,

    ### Train Hyperparameters
    "img_size": [1280, 1280], # [h, w] 896
    'learning_rate':0.0001,
    'batch_size':6,
    'epochs':200,
    'optimizer':'Adam',  #Adam  SGD AdaBelief Ranger
    'scheduler':'default-0.1-30', #default  SGDR-5-2    step-4-0.8
    'threshold': 0.5,

    'warmup_epoch':0, # 
    'weight_decay' : 0,#0.0001,
    'early_stop_patient':50,

    'label_smooth':0,
    # 'checkpoint':None,
    'class_weight': None,#s[1.4, 0.78], # None [1, 1]
    'clip_gradient': 0,#1,       # 0
    'freeze_nonlinear_epoch':0,
  
    'dropout':0., #before last_linear
    
    ### Test
    'model_path':'/opt/app/unetplusplus_efficientnet-b6_e98_0.95522.pth',#test model

    'test_path':"/input/images/adenocarcinoma-image",#test without label, just show img result
    'test_batch_size':2,
}
