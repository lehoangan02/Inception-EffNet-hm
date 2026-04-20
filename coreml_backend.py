import numpy as np
import torch


class CoreMLModelRunner:
    def __init__(
        self,
        model_path,
        input_name='input',
        hm_name='hm',
        wh_name='wh',
        reg_name='reg',
        cls_theta_name='cls_theta',
        compute_units='all',
    ):
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                'coremltools is required for --backend coreml. '
                'Install it with: pip install coremltools'
            ) from exc

        self._ct = ct
        self.device = torch.device('cpu')
        self.input_name = input_name
        self.hm_name = hm_name
        self.wh_name = wh_name
        self.reg_name = reg_name
        self.cls_theta_name = cls_theta_name

        compute_map = {
            'all': ct.ComputeUnit.ALL,
            'cpu_only': ct.ComputeUnit.CPU_ONLY,
            'cpu_and_gpu': ct.ComputeUnit.CPU_AND_GPU,
            'cpu_and_ne': ct.ComputeUnit.CPU_AND_NE,
        }
        self.model = ct.models.MLModel(model_path, compute_units=compute_map[compute_units])

    def to(self, device):
        # Core ML runtime is managed by compute units; keep compatibility with PyTorch call sites.
        return self

    def eval(self):
        return self

    def _find_output_key(self, output_dict, preferred_key):
        if preferred_key in output_dict:
            return preferred_key

        lower_map = {k.lower(): k for k in output_dict.keys()}
        if preferred_key.lower() in lower_map:
            return lower_map[preferred_key.lower()]

        for k in output_dict.keys():
            if preferred_key.lower() in k.lower():
                return k

        raise KeyError(
            "Core ML output '{}' not found. Available outputs: {}".format(
                preferred_key, list(output_dict.keys())
            )
        )

    def _as_numpy(self, image):
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def _as_tensor(self, output):
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)

    def __call__(self, image):
        input_arr = self._as_numpy(image)
        outputs = self.model.predict({self.input_name: input_arr})

        hm_key = self._find_output_key(outputs, self.hm_name)
        wh_key = self._find_output_key(outputs, self.wh_name)
        reg_key = self._find_output_key(outputs, self.reg_name)
        cls_theta_key = self._find_output_key(outputs, self.cls_theta_name)

        return {
            'hm': self._as_tensor(outputs[hm_key]),
            'wh': self._as_tensor(outputs[wh_key]),
            'reg': self._as_tensor(outputs[reg_key]),
            'cls_theta': self._as_tensor(outputs[cls_theta_key]),
        }
