
import os
from . import unisal_file_opener
from . import transal_file_opener


def load_dataloader(args):
    path_ = os.path.dirname(os.path.abspath(__file__))
    print(path_)

    if args.model == "Unisal":
        file_opener = unisal_file_opener
    elif args.model == "TranSalNetDense":
        file_opener = transal_file_opener
    elif args.model == "TranSalNetRes":
        file_opener = transal_file_opener
    return file_opener
