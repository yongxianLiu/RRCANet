from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='RRCANet',
                        help='model name: ResUNet、ResUNet_RuCB')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='two',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='RRCANet',
                        help='ResUNet,ResUNet_RuCB, resnet_10, resnet_34 , DNANet_RuCB')
    parser.add_argument('--deep_supervision', type=str, default='False', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name: NUDT-SIRST, NUAA-SIRST, IRSTD-1k')
    parser.add_argument('--st_model', type=str, default= 'NUAA-SIRST_RRCANet',
                        help='IRSTD-1k_ResUnet_RuCB_04_11_2024_20_53_40_wDS')
    parser.add_argument('--model_dir', type=str,
                        default = 'NUAA-SIRST_RRCANet/mIoU__RRCANet_NUAA-SIRST_epoch.pth.tar',
                        help    = 'IRSTD-1k_ResUNet_BiOv8_26_07_2024_10_00_11_wDS/mIoU__ResUNet_BiOv8_IRSTD-1k_epoch.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 80_20')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110,1500)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    parser.add_argument('--iter', default=2, type=int, help='recurrent iteration')
    parser.add_argument('--integrate', default='True', type=str, help='integrate all decoded features')
    parser.add_argument('--multiplier', default=1.0, type=float, help='parameter multiplier')
    parser.add_argument('--num_layers', default=4, type=int, help='depth of network')


    args = parser.parse_args()

    # the parser
    return args