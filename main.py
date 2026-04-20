import argparse
import train
import test
import eval
from coreml_backend import CoreMLModelRunner
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_50.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')
    parser.add_argument('--phase', type=str, default='eval', help='Phase choice= {train, test, eval}')
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
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'coreml'],
                        help='Inference backend. Use coreml for Apple Core ML runtime (eval/test only).')
    parser.add_argument('--coreml_model', type=str, default='',
                        help='Path to .mlmodel or .mlpackage for Core ML inference backend')
    parser.add_argument('--coreml_input_name', type=str, default='input',
                        help='Core ML model input feature name for image tensor')
    parser.add_argument('--coreml_hm_name', type=str, default='hm',
                        help='Core ML output feature name for heatmap tensor')
    parser.add_argument('--coreml_wh_name', type=str, default='wh',
                        help='Core ML output feature name for wh tensor')
    parser.add_argument('--coreml_reg_name', type=str, default='reg',
                        help='Core ML output feature name for regression tensor')
    parser.add_argument('--coreml_cls_theta_name', type=str, default='cls_theta',
                        help='Core ML output feature name for cls_theta tensor')
    parser.add_argument('--coreml_compute_units', type=str, default='all',
                        choices=['all', 'cpu_only', 'cpu_and_gpu', 'cpu_and_ne'],
                        help='Core ML compute unit preference')
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

    if args.backend == 'coreml':
        if args.phase == 'train':
            raise ValueError('Core ML backend is inference-only. Use --backend pytorch for training.')
        if not args.coreml_model:
            raise ValueError('When --backend coreml is set, --coreml_model must be provided.')
        model = CoreMLModelRunner(
            model_path=args.coreml_model,
            input_name=args.coreml_input_name,
            hm_name=args.coreml_hm_name,
            wh_name=args.coreml_wh_name,
            reg_name=args.coreml_reg_name,
            cls_theta_name=args.coreml_cls_theta_name,
            compute_units=args.coreml_compute_units,
        )

    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio)

        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)
