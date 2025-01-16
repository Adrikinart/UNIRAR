import os 
from . import Unisal
from . import TranSalNetDense
from . import TranSalNetRes
from . import TempSal

import torch


def load_model(args , DEFAULT_DEVICE):
    path_ = os.path.dirname(os.path.abspath(__file__))

    if args.model == "Unisal":
        model = Unisal(
            bypass_rnn=False,
        )
        if args.finetune.lower() == "true":

            if os.path.exists(path_ + "/../model/Unisal/weights/" + "weights_best.pth"):
                print("Model Load")
                model.load_weights(path_ + "/../model/Unisal/weights/" + "weights_best.pth")
            else:
                print("Model not found")

        layers_index=[
            [4,5],
            [7,8],
            [10,11],
            [13,14],
            [16,17]
        ]

    elif args.model == "TranSalNetDense":
        model = TranSalNetDense()
        model.load_state_dict(torch.load(path_ + '/../model/TranSalNet/pretrained_models/TranSalNet_Dense.pth', map_location=DEFAULT_DEVICE))
        layers_index=[
            # [2,3],
            # [4,5],
            [6,7],
            [8],
            [9],
            [10]
        ]

    elif args.model == "TranSalNetRes":
        model = TranSalNetRes()
        model.load_state_dict(torch.load(path_ + '/../model/TranSalNet/pretrained_models/TranSalNet_Res.pth', map_location=DEFAULT_DEVICE))

        layers_index=[
            # [2,3],
            # [4,5],
            [6],
            [7],
            # [8],
            
        ]

    elif args.model == "TempSal":
        
        model_checkpoint_path= "./weights/multilevel_tempsal.pt"
        model = TempSal(
            device=DEFAULT_DEVICE,
            model_path=model_checkpoint_path,
            model_vol_path= model_checkpoint_path,
            time_slices=5,
            train_model=0
        )

        layers_index=[
            [1],
            [2],
            [3],
            [4],
            [5]
        ]




    return model, layers_index