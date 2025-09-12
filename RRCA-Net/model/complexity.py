from model.parse_args_test import *
from model.load_param_data import *

from model.model_res_UNet import *
from RRCANet import *

from model_res_UNet import *
from torchstat import stat
from torchsummary import summary
from model.utils import *
from model.model_DNANet import *
from model.ResUNet_RuCB import *
from model.model_res_UNet import *
from model.AMFU import *


args = parse_args()
nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

net = AMFU()
# net = AMFU_RuCB(n_classes=1,in_channels=3,iterations=2, multiplier=1.0, num_layers=5, integrate='True', deep_supervision='True')
#net = DNANet(num_classes=1, input_channels=3, nb_filter=[16,32,64,128,256], num_blocks=[2,2,2,2], block=Res_CBAM_block, deep_supervision=True)
# net = res_UNet(num_classes=1, input_channels=3, block=Res_block, num_blocks=[2,2,2,2],nb_filter=[8,16,32,64,128])
# net = ResUNet_BiOv8(num_classes=1, input_channels=3, block=Res_block, num_blocks=num_blocks, iterations=2, num_layers=4,
#                    multiplier=1.0, integrate='True', deep_supervision='False')
# net = ResUNet_RuCB(num_classes=1, input_channels=3, block=Res_block, num_blocks=num_blocks, iterations=2, num_layers=5,
#                    multiplier=1.0, integrate='True', deep_supervision='False')
# net = res_UNet(num_classes=1, input_channels=3, block=Res_block, num_blocks=[2,2,2,2], nb_filter=nb_filter)

# net = DNANet_RuCB2(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks, iterations=2, num_layers=5,
#                    multiplier=2.0, integrate='True', deep_supervision='True')

timef=0
for i in range(100):
    inputs = torch.randn((1, 3, 256, 256))
    start = time.perf_counter()

    out = net(inputs)

    end = time.perf_counter()

    timef += end-start
stat(net, (3, 256, 256))


running_FPS = 1 / (timef/100)
print('running_FPS:', running_FPS)
print(out[0].size())
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('parameters_count:',count_parameters(net))





