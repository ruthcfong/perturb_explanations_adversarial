import os

# TODO change
caffe_dir = '/users/ruthfong/sample_code/Caffe-ExcitationBP/'
alexnet_prototxt = '/users/ruthfong/packages/caffe/models/bvlc_alexnet/deploy_force_backward.prototxt'
alexnet_model = '/users/ruthfong/packages/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
#alexnet_prototxt = '/users/ruthfong/packages/caffe/models/bvlc_reference_caffenet/deploy_force_backward.prototxt'
#alexnet_model = '/users/ruthfong/packages/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
vgg16_prototxt = '/users/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers_deploy_force_backward.prototxt'
vgg16_model = '/users/ruthfong/packages/caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel'
googlenet_prototxt = '/users/ruthfong/packages/caffe/models/bvlc_googlenet/deploy_force_backward.prototxt'
googlenet_model = '/users/ruthfong/packages/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# computed on range(100,4000) on heldout set
mus = {'defaults_iter_50': 0.917910, 'saliency': 0.076455, 'guided_backprop': 0.039671,
        'grad_cam': 0.353798, 'contrast_excitation_backprop': 0.049973,
        'excitation_backprop': 0.206981}
sigmas = {'defaults_iter_50': 0.157649, 'saliency': 0.070174, 'guided_backprop': 0.054773,
        'grad_cam': 0.301801, 'contrast_excitation_backprop': 0.124677,
        'excitation_backprop': 0.147678}

voc_dir = '/users/ruthfong/sample_code/Caffe-ExcitationBP/models/finetune_googlenet_voc_pascal'
googlenet_voc_prototxt = os.path.join(voc_dir, 'deploy_force_backward.prototxt')
googlenet_voc_model = '/data/ruthfong/VOCdevkit/VOC2007/caffe/snapshots/finetune_googlenet_voc_pascal_iter_5000.caffemodel'

googlenet_coco_prototxt = os.path.join(caffe_dir, 'models/COCO/deploy.prototxt')
googlenet_coco_model = os.path.join(caffe_dir, 'models/COCO/GoogleNetCOCO.caffemodel')

voc_labels_desc = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
# default hyperparameters for optimize_mask.py
num_iters = 300
lr = 1e-1
l1_lambda = 1e-4
l1_ideal = 1
l1_lambda_2 = 0
tv_lambda = 1e-2
tv_beta = 3
jitter = 4
num_top = 0
noise = 0
null_type = 'blur'
given_gradient = True
norm_score = False
end_layer = 'prob'
use_conv_norm = False
blur_mask = 5
mask_scale = 8
