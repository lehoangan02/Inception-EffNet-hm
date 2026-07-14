import argparse
import train
import test
import eval
from checkpoint_paths import default_save_dir
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', '--epochs', dest='num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-3, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_50.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--save_dir', type=str, default=default_save_dir(),
                        help='Root directory for weights_<dataset>; defaults to BBAV_SAVE_DIR, then /dev/shm if writable')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')
    parser.add_argument('--phase', type=str, default='eval', choices=['train', 'test', 'eval', 'loss'],
                        help='Phase choice= {train, test, eval, loss}')
    parser.add_argument('--loss_epochs', type=int, nargs='+', default=[9, 10],
                        help='Checkpoint epochs to evaluate loss for when --phase loss is used')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use ImageNet-pretrained EfficientNetV2 backbone weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Disable pretrained backbone weights (recommended on offline clusters)')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--heatmap_only', action='store_true',
                        help='Use heatmap (focal) loss only, skip wh/offset/theta losses')
    parser.add_argument('--reset_lr', action='store_true',
                        help='Reset LR to init_lr after resuming checkpoint (as if starting fresh)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC}
    num_classes = {'dota': 15, 'hrsc': 1}
    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX_EfficientNetV2(heads=heads,
                              pretrained=args.pretrained,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])

    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio)

        ctrbox_obj.train_network(args)
    elif args.phase == 'loss':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio)
        ctrbox_obj.calculate_checkpoint_losses(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)
