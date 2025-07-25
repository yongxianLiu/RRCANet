# torch and visulization
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import load_dataset, load_param

# model
from model.model_DNANet import *
from model.RRCANet import *
from model.ResUNet_RuCB import *
from model.model_res_UNet import *
from model.AMFU import *
from model.UIUNet import *
from model.ACM import *

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                  transform=input_transform, suffix=args.suffix)
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                transform=input_transform, suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        # if args.model == 'DNANet_RuCB':
        #     model = DNANet_RuCB(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
        #                          iterations=args.iter,num_layers=args.num_layers,multiplier=args.multiplier,integrate=args.integrate,deep_supervision=args.deep_supervision)
        if args.model == 'DNANet':
            model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,nb_filter=nb_filter,
                    deep_supervision=args.deep_supervision)
        elif args.model == 'ResUNet':
            model = res_UNet(num_classes=1, input_channels=args.in_channels, block=Res_block, num_blocks=num_blocks,nb_filter=nb_filter)

        elif args.model == 'RRCANet':
            model = RRCANet(num_classes=1, input_channels=args.in_channels, block=Res_block,
                                  num_blocks=num_blocks, iterations=args.iter, num_layers=args.num_layers,
                                  multiplier=args.multiplier, integrate=args.integrate, deep_supervision=args.deep_supervision)
        elif args.model == 'ResUnet_RuCB':
            model = ResUNet_RuCB(num_classes=1, input_channels=args.in_channels, block=Res_block,
                                 num_blocks=num_blocks, iterations=args.iter, num_layers=args.num_layers,
                                 multiplier=args.multiplier, integrate=args.integrate, deep_supervision=args.deep_supervision)
        elif args.model == 'AMFU':
            model = AMFU(n_classes=1, in_channels=args.in_channels)
        # elif args.model == 'AMFU_RuCB':
            # model = AMFU_RuCB(n_classes=1, in_channels=args.in_channels,iterations=args.iter, num_layers=args.num_layers,
                                 # multiplier=args.multiplier, integrate=args.integrate, deep_supervision=args.deep_supervision)
        elif args.model == 'UIUNet':
            model = UIUNET()
        elif args.model == 'ACM':
            model = ASKCResUNet()

        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Optimizer and lr scheduling
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'MultiStepLR':
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[150], gamma=0.2)
        elif args.scheduler == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.8)

        # Evaluation metrics
        self.best_iou = 0
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Training
    def training(self, epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += DicepolyTopK(pred, labels)
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = DicepolyTopK(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, lr %f, training loss %.4f' % (epoch, self.optimizer.param_groups[0]['lr'], losses.avg))
        self.train_loss = losses.avg
        #self.scheduler.step()

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.ROC.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += DicepolyTopK(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = DicepolyTopK(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            test_loss = losses.avg
            # with SummaryWriter(log_dir='./curve/'+self.save_dir+'/loss', comment='loss') as writer:
            # writer.add_scalars('add_scalars/loss', {'train': self.train_loss, 'test': test_loss}, epoch)
            # with SummaryWriter(log_dir='./curve/'+self.save_dir+'/mIOU', comment='mIOU') as writer:
            # writer.add_scalar('add_scalar/mIOU', mean_IOU, epoch)
        # save high-performance model
        best_iou = save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                              self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())

        self.best_iou = best_iou


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)
