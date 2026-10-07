#!python3

# pylint: disable=R, W0401, W0614, W0703
from enum import Enum
from lib.meta import Meta
from os import path

alt_names = None

darknet_ready = True
try:
    from lib.darknet import YoloNet
except Exception as e:
    print(f'Error during importing YoloNet! - {e}')
    darknet_ready = False

onnx_ready = True
try:
    from lib.onnx import OnnxNet
except Exception as e:
    print(f'Error during importing OnnxNet! - {e}')
    onnx_ready = False

rknn_ready = True
try:
    from lib.rknn import RknnNet
except Exception as e:
    print(f'Error during importing RknnNet! - {e}')
    rknn_ready = False

xdna_ready = True
try:
    from lib.xdna_net import XdnaNet
except Exception as e:
    print(f'Error during importing XdnaNet! - {e}')
    xdna_ready = False

torch_ready = True
try:
    from lib.torch_onnx import TorchOnnxNet
except Exception as e:
    print(f'Error during importing TorchOnnxNet! - {e}')
    torch_ready = False


def load_net(config_path, meta_path, weights_path=None):

    def try_loading_net(net_config_priority):
        for net_config in net_config_priority:
            weights = net_config['weights_path']
            use_gpu = net_config['use_gpu']
            runtime = net_config.get('runtime')

            net_main = None
            try:
                print(f'----- Trying to load weights: {weights} - use_gpu = {use_gpu}' + (f' - runtime = {runtime}' if runtime else '') + ' -----')
                if runtime == 'xdna':
                    if not xdna_ready:
                        raise Exception('Not loading the ONNX net on the NPU due to previous import failure. Check earlier log for errors.')
                    net_main = XdnaNet(weights, meta_path, use_gpu)

                elif runtime == 'torch':
                    if not torch_ready:
                        raise Exception('Not loading the ONNX net in PyTorch due to previous import failure. Check earlier log for errors.')
                    net_main = TorchOnnxNet(weights, meta_path, use_gpu)

                elif weights.endswith(".onnx"):
                    if not onnx_ready:
                        raise Exception('Not loading ONNX net due to previous import failure. Check earlier log for errors.')
                    net_main = OnnxNet(weights, meta_path, use_gpu)

                elif weights.endswith(".darknet"):
                    if not darknet_ready:
                        raise Exception('Not loading darknet net due to previous import failure. Check earlier log for errors.')
                    net_main = YoloNet(weights, meta_path, config_path, use_gpu)

                elif weights.endswith(".rknn"):
                    if not rknn_ready:
                        raise Exception('Not loading RKNN net due to previous import failure. Check earlier log for errors.')
                    net_main = RknnNet(weights, meta_path, use_gpu)

                else:
                    raise Exception(f'Can not recognize net from weights file surfix: {weights}')

                print('Succeeded!')
                return net_main
            except Exception as e:
                print(f'Failed! - {e}')

        raise Exception(f'Failed to load any net after trying: {net_config_priority}')

    global alt_names  # pylint: disable=W0603

    model_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'model')
    net_config_priority = [
            # The ONNX model on an AMD XDNA1 NPU, through precompiled mlir-aie programs.
            dict(weights_path='/model_cache/ml_api/onnx/model-weights.onnx', use_gpu=True, runtime='xdna'),
            dict(weights_path='/model_cache/ml_api/darknet/model-weights.darknet', use_gpu=True),
            dict(weights_path='/model_cache/ml_api/darknet/model-weights.darknet', use_gpu=False),
            # The ONNX model on an AMD (ROCm) GPU, which ONNX Runtime cannot use.
            dict(weights_path='/model_cache/ml_api/onnx/model-weights.onnx', use_gpu=True, runtime='torch'),
            dict(weights_path='/model_cache/ml_api/onnx/model-weights.onnx', use_gpu=True),
            dict(weights_path='/model_cache/ml_api/onnx/model-weights.onnx', use_gpu=False),
            dict(weights_path='/model_cache/ml_api/rknn/model-weights.rknn', use_gpu=False),
        ]
    if weights_path is not None:
        net_config_priority = [ dict(weights_path=weights_path, use_gpu=True), dict(weights_path=weights_path, use_gpu=False) ]

    net_main = try_loading_net(net_config_priority)

    if alt_names is None:
        try:
            meta = Meta(meta_path)
            alt_names = meta.names
        except Exception:
            pass

    return net_main

def detect(net, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    return net.detect(net.meta, image, alt_names, thresh, hier_thresh, nms, debug)
