import sys
from pathlib import Path
d = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(d))
from lib.modelzoo.retinanet import retinanet_resnet50_v1_coco
from lib.modelzoo.centernet import centernet_resnet18_v1_coco

_models = {
    'retinanet_resnet50_v1_coco': retinanet_resnet50_v1_coco,
    'centernet_resnet18_v1_coco': centernet_resnet18_v1_coco
}


def get_model(model_name):
    return _models[model_name]()


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.expanduser('~/incubator-mxnet/python'))
    from mxnet import nd
    from mxnet import autograd
    # model_name = 'retinanet_resnet50_v1_coco'
    model_name = 'centernet_resnet18_v1_coco'
    net = get_model(model_name)
    input_size = 512
    x = nd.random.uniform(0, 1, (1, 3, input_size, input_size))
    with autograd.record():
        outputs = net(x)


