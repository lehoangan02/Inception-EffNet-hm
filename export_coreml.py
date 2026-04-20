import argparse
import torch

from models import ctrbox_net


class CoreMLExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        dec = self.model(x)
        return dec['hm'], dec['wh'], dec['reg'], dec['cls_theta']


def parse_args():
    parser = argparse.ArgumentParser(description='Export BBAV CTRBOX model to Core ML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--dataset', type=str, default='dota', choices=['dota', 'hrsc'])
    parser.add_argument('--input_h', type=int, default=608)
    parser.add_argument('--input_w', type=int, default=608)
    parser.add_argument('--output', type=str, default='bbav_coreml.mlpackage',
                        help='Output Core ML package path (.mlpackage)')
    return parser.parse_args()


def load_model(dataset, checkpoint_path):
    num_classes = {'dota': 15, 'hrsc': 1}
    heads = {
        'hm': num_classes[dataset],
        'wh': 10,
        'reg': 2,
        'cls_theta': 1,
    }

    model = ctrbox_net.CTRBOX_EfficientNetV2(
        heads=heads,
        pretrained=False,
        down_ratio=4,
        final_kernel=1,
        head_conv=256,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    args = parse_args()

    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError('coremltools is required. Install with: pip install coremltools') from exc

    model = load_model(args.dataset, args.checkpoint)
    wrapped = CoreMLExportWrapper(model)

    example = torch.randn(1, 3, args.input_h, args.input_w)
    traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        convert_to='mlprogram',
        inputs=[ct.TensorType(name='input', shape=example.shape)],
        outputs=[
            ct.TensorType(name='hm'),
            ct.TensorType(name='wh'),
            ct.TensorType(name='reg'),
            ct.TensorType(name='cls_theta'),
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.save(args.output)
    print('Saved Core ML model to {}'.format(args.output))


if __name__ == '__main__':
    main()
