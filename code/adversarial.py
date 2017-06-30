import caffe

import time, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from helpers import *
from optimize_mask import generate_learned_mask
from heatmaps import compute_heatmap

def compute_adversarial_example(net, transformer, path, label, method = 'fgsm', step = 1, 
                                alpha = 1.0, eps = 16, adv_label = 'min', top = 'prob', show_fig = False):
    assert(method == 'fgsm' or method == 'one_step') # fast gradient sign method or one-step target class
    
    img = transformer.preprocess('data', caffe.io.load_image(path))
    orig_img = transformer.deprocess('data', img)
    
    # forward pass
    net.blobs['data'].data[...] = img
    net.forward(start = 'data', end = top)

    orig_scores = np.squeeze(net.blobs[top].data.copy())
    max_label = np.argmax(orig_scores)
    
    # set target class
    if method == 'fgsm':
        target_label = label 
    elif method == 'one_step':
        if adv_label == 'min':
            target_label = np.argmin(orig_scores)
        elif adv_label == 'rand':
            target_label = np.random.randint(net.blobs[top].data.shape[-1])
        else:
            target_label = adv_label
    else:
        assert(False)
        
    if step is None:
        step = int(np.minimum(eps+4, 1.25*eps))
                
    adv_img = orig_img.copy()
    acc_noise = np.zeros(adv_img.shape)
    for i in range(step):
        if i > 0:
            # forward pass
            net.blobs['data'].data[...] = transformer.preprocess('data', adv_img)
            net.forward(start = 'data', end = top)

        # compute derivative of cross-entropy loss (w.r.t softmax)
        curr_scores = np.squeeze(net.blobs[top].data.copy())
        top_grad = np.zeros(net.blobs[top].data.shape) # (1,1000)
        top_grad[0][target_label] = -1/float(curr_scores[target_label]) 

        # backward pass
        net.blobs[top].diff[...] = top_grad
        net.backward(start = top, end = 'data')

        # construct adversarial example using bottom gradient
        bottom_grad = net.blobs['data'].diff.copy()
        if method == 'fgsm':
            #adv_img += alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
            acc_noise += alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
        else:
            #adv_img -= alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
            acc_noise -= alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
        
        # clip if not one-step method
        if step != 1:
            #adv_img[...] = np.clip(np.clip(adv_img, orig_img - eps/float(255), 
            #                       orig_img + eps/float(255)),0,1)
            acc_noise[...] = np.clip(acc_noise, -eps/float(255), eps/float(255))
        adv_img = np.clip(orig_img + acc_noise, 0,1)

    # forward pass with adversarial example
    net.blobs['data'].data[...] = transformer.preprocess('data', adv_img)
    net.forward(end = top)
    adv_scores = np.squeeze(net.blobs[top].data.copy())
    adv_label = np.argmax(adv_scores)

    if show_fig:
        f, ax = plt.subplots(1,3)
        f.set_size_inches(12, 6)
        ax[0].imshow(orig_img)
        ax[0].set_title('%s: %.2f (%s: %.2f)' % (get_short_class_name(max_label), orig_scores[max_label], 
                                             get_short_class_name(label), orig_scores[label]))
        #vis = np.transpose(np.squeeze(bottom_grad),(1,2,0))
        #ax[1].imshow((vis-np.min(vis))/(np.max(vis) - np.min(vis)))
        #ax[1].imshow(np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0)))
        #ax[1].imshow(acc_noise)
        ax[1].imshow(orig_img-adv_img)
        ax[1].set_title(r'$\alpha = %.3f$' % alpha)
        ax[2].imshow(adv_img)
        ax[2].set_title('%s: %.2f (%s: %.2f)' % (get_short_class_name(adv_label), adv_scores[adv_label], 
                                             get_short_class_name(label), adv_scores[label]))
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()
    
    return (adv_img, adv_label, adv_label == label, label in np.argsort(adv_scores)[-5:])

def compute_adversarial_example_2(net, transformer, path, label, method = 'fgsm', step = 1, 
                                alpha = 1.0, eps = 16, adv_label = 'min', top = 'loss3/classifier', show_fig = False):
    assert(method == 'fgsm' or method == 'one_step') # fast gradient sign method or one-step target class
    
    img = transformer.preprocess('data', caffe.io.load_image(path))
    orig_img = transformer.deprocess('data', img)
    
    # forward pass
    net.blobs['data'].data[...] = img
    net.forward(start = 'data', end = 'prob')

    orig_scores = np.squeeze(net.blobs['prob'].data.copy())
    max_label = np.argmax(orig_scores)
    
    # set target class
    if method == 'fgsm':
        target_label = label 
    elif method == 'one_step':
        if adv_label == 'min':
            target_label = np.argmin(orig_scores)
        elif adv_label == 'rand':
            target_label = np.random.randint(net.blobs[top].data.shape[-1])
        else:
            target_label = adv_label
    else:
        assert(False)
    target_vector = np.zeros(net.blobs[top].data.shape)
    target_vector[0][target_label] = 1
        
    if step is None:
        step = int(np.minimum(eps+4, 1.25*eps))
                
    adv_img = orig_img.copy()
    for i in range(step):
        if i > 0:
            # forward pass
            net.blobs['data'].data[...] = transformer.preprocess('data', adv_img)
            net.forward(start = 'data', end = 'prob')

        # compute derivative of cross-entropy loss (w.r.t softmax)
        #curr_scores = np.squeeze(net.blobs[top].data.copy())
        #top_grad = np.zeros(net.blobs[top].data.shape) # (1,1000)
        #top_grad[0][target_label] = -1/float(curr_scores[target_label]) 
        top_grad = net.blobs['prob'].data.copy() - target_vector

        # backward pass
        net.blobs[top].diff[...] = top_grad
        net.backward(start = top, end = 'data')

        # construct adversarial example using bottom gradient
        bottom_grad = net.blobs['data'].diff.copy()
        if method == 'fgsm':
            adv_img += alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
        else:
            adv_img -= alpha/float(255)*np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0))
        
        # clip if not one-step method
        if step != 1:
            adv_img[...] = np.clip(np.clip(adv_img, orig_img - eps/float(255), 
                                   orig_img + eps/float(255)),0,1)

    if show_fig:
        # forward pass with adversarial example
        net.blobs['data'].data[...] = transformer.preprocess('data', adv_img)
        net.forward(start = 'data', end = 'prob')
        adv_scores = np.squeeze(net.blobs['prob'].data.copy())
        adv_label = np.argmax(adv_scores)

        f, ax = plt.subplots(1,3)
        f.set_size_inches(12, 6)
        ax[0].imshow(orig_img)
        ax[0].set_title('%s: %.2f (%s: %.2f)' % (get_short_class_name(max_label), orig_scores[max_label], 
                                             get_short_class_name(label), orig_scores[label]))
        #vis = np.transpose(np.squeeze(bottom_grad),(1,2,0))
        #ax[1].imshow((vis-np.min(vis))/(np.max(vis) - np.min(vis)))
        ax[1].imshow(np.transpose(np.squeeze(np.sign(bottom_grad)), (1,2,0)))
        ax[1].set_title(r'$\alpha = %.3f$' % alpha)
        ax[2].imshow(adv_img)
        ax[2].set_title('%s: %.2f (%s: %.2f)' % (get_short_class_name(adv_label), adv_scores[adv_label], 
                                             get_short_class_name(label), adv_scores[label]))
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()
    
    return (adv_img, np.argmax(np.squeeze(net.blobs[top].data.copy()))) 

def check_clean_image(net, transformer, path, label, top = 'prob'):
    img = transformer.preprocess('data', caffe.io.load_image(path))
    net.blobs['data'].data[...] = img
    net.forward(start = 'data', end = top)
    orig_scores = np.squeeze(net.blobs[top].data.copy())
    max_label = np.argmax(orig_scores)
    return (max_label, max_label == label, label in np.argsort(orig_scores)[-5:])


def main():
    gpu = 1
    net_type = 'googlenet'

    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    net = get_net(net_type)

    labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')
    (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/val_imdb.txt')
    #(paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/annotated_train_heldout_imdb.txt')
    paths = np.array(paths)
    labels = np.array(labels)

    task_type = 'adversarial'
    
    epsilons = np.array([1,2,4,8,12,16])
    methods = ['fgsm', 'one_step', 'fgsm_iter', 'one_step_iter']

    if task_type == 'mask':
        num_iters = 50

        start_i = 350 
        end_i = 625 
    elif task_type == 'heatmaps':
        start_i = 0
        end_i = 5000

        heatmap_type = 'excitation_backprop'
    elif task_type == 'adversarial':
        start_i = 0  
        end_i = 9375 
        adv_img_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_val'
        generate_adversarial_examples(net, paths, labels, adv_img_dir, start_i, end_i)
    elif task_type == 'graph':
        generate_adversarial_graph(net, paths, labels)
        return
    else:
        assert(False)

    for eps in epsilons:
        for adv_type in methods:

            if task_type == 'mask':
                adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_train_heldout/%s/eps_%d/' % (adv_type, eps)
                adv_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/defaults_iter_%d/' % (adv_type, eps, num_iters)
                true_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/defaults_iter_%d/' % num_iters
                fig_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_comparison_figs/imagenet_train_heldout/%s/eps_%d/defaults_iter_%d/' % (adv_type, eps, num_iters)
                learn_masks(net, paths, labels, adv_dir, adv_mask_dir, true_mask_dir, fig_dir, gpu, start_i, end_i, labels_desc)
            elif task_type == 'heatmaps':
                adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_train_heldout/%s/eps_%d/' % (adv_type, eps)
                adv_mask_true_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_true_masks/imagenet_train_heldout/%s/eps_%d/%s/' % (adv_type, eps, heatmap_type)
                adv_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/%s/' % (adv_type, eps, heatmap_type)
                true_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/%s/' % heatmap_type
                fig_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_comparison_figs/imagenet_train_heldout/%s/eps_%d/%s/' % (adv_type, eps, heatmap_type)

                #heatmap_type = 'contrast_excitation_backprop'
                generate_heatmaps(net, paths, labels, heatmap_type, adv_dir, adv_mask_true_dir, adv_mask_dir, true_mask_dir, fig_dir, 
                    gpu, start_i, end_i)
            else:
                assert(False)
                generate_adversarial_examples(net, paths, labels)
    

def generate_heatmaps(net, paths, labels, heatmap_type, adv_dir, adv_mask_true_dir, adv_mask_dir, true_mask_dir, fig_dir, 
        gpu = None, start_i = 0, end_i = 5000):
    transformer = get_ILSVRC_net_transformer(net)

    top = 'loss3/classifier'
    if heatmap_type == 'saliency' or heatmap_type == 'guided_backprop':
        bottom = 'data'
        norm_deg = np.inf
    elif heatmap_type == 'excitation_backprop':
        bottom = 'pool2/3x3_s2'
        norm_deg = -1
    elif heatmap_type == 'contrast_excitation_backprop':
        bottom = 'pool2/3x3_s2'
        norm_deg = -2
    elif heatmap_type == 'grad_cam':
        bottom = 'inception_5b/output'
        norm_deg = None

    if not os.path.exists(adv_mask_true_dir):
        os.makedirs(adv_mask_true_dir)
    if not os.path.exists(adv_mask_dir):
        os.makedirs(adv_mask_dir)
    if not os.path.exists(true_mask_dir):
        os.makedirs(true_mask_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for ind in range(start_i, end_i):
        if os.path.exists(os.path.join(adv_mask_dir, '%d.png' % ind)):
            print '%s already exists so skipping' % os.path.join(adv_mask_dir, '%d.png' % ind)
            continue
        start = time.time()
        true_label = labels[ind]
        true_path = paths[ind]
        adv_path = os.path.join(adv_dir, '%d.png' % ind)

        # compute adversarial label
        adv_img = transformer.preprocess('data', caffe.io.load_image(adv_path))
        net.blobs['data'].data[...] = adv_img
        net.forward()
        scores = np.squeeze(net.blobs['prob'].data)
        adv_label = np.argmax(scores)

        if os.path.exists(os.path.join(true_mask_dir, '%d.png' % ind)):
            orig_heatmap = np.array(caffe.io.load_image(os.path.join(true_mask_dir, '%d.png' % ind)))
        else:
            orig_heatmap = compute_heatmap(net, transformer, true_path, true_label, heatmap_type, top, top,
                                outputBlobName = bottom, outputLayerName = bottom, secondTopBlobName = 'pool5/7x7_s1',
                                secondTopLayerName = 'pool5/7x7_s1', norm_deg = norm_deg, gpu = gpu)
        adv_true_heatmap = compute_heatmap(net, transformer, adv_path, true_label, heatmap_type, top, top,
                            outputBlobName = bottom, outputLayerName = bottom, secondTopBlobName = 'pool5/7x7_s1',
                            secondTopLayerName = 'pool5/7x7_s1', norm_deg = norm_deg, gpu = gpu)
        adv_max_heatmap = compute_heatmap(net, transformer, adv_path, adv_label, heatmap_type, top, top,
                            outputBlobName = bottom, outputLayerName = bottom, secondTopBlobName = 'pool5/7x7_s1',
                            secondTopLayerName = 'pool5/7x7_s1', norm_deg = norm_deg, gpu = gpu)

        # save comparison figures for first 100
        #if ind < 100:
        #    try:
        #        f, ax = plt.subplots(1,4)
        #        f.set_size_inches(16,4)
        #        ax[0].imshow(caffe.io.load_image(adv_path))
        #        ax[0].set_title('adv img (true: %s, adv: %s)' % (get_short_class_name(true_label), 
        #                                                        get_short_class_name(adv_label)))
        #        ax[1].imshow(orig_heatmap)
        #        ax[1].set_title('true (%s)' % heatmap_type)
        #        ax[2].imshow(adv_true_heatmap)
        #        ax[2].set_title('adv true (%s)' % heatmap_type)
        #        ax[3].imshow(adv_max_heatmap)
        #        ax[3].set_title('adv max (%s)' % heatmap_type)
        #        for a in ax:
        #            a.set_xticks([])
        #            a.set_yticks([])
        #        plt.savefig(os.path.join(fig_dir, '%d.png' % ind))
        #        plt.close()
        #    except:
        #        pass
        #    #plt.show()
        
        # save saliency masks
        imsave(os.path.join(adv_mask_dir, '%d.png' % ind), adv_max_heatmap)
        imsave(os.path.join(adv_mask_true_dir, '%d.png' % ind), adv_true_heatmap)
        if not os.path.exists(os.path.join(true_mask_dir, '%d.png' % ind)):
            imsave(os.path.join(true_mask_dir, '%d.png' % ind), orig_heatmap)

        print ind, time.time() - start, os.path.join(adv_mask_dir, '%d.png' % ind)

def learn_masks(net, paths, labels, adv_dir, adv_mask_dir, true_mask_dir, fig_dir, gpu = None, start_i = 0, end_i = 5000, 
        labels_desc = np.loadtxt('/home/ruthfong/packages/caffe/data/ilsvrc12/synset_words.txt', str, delimiter='\t')):
    transformer = get_ILSVRC_net_transformer(net)

    # default parameters
    num_iters = 50 
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

    plot_step = None # 10
    debug = False
    verbose = False
    mask_init_type = 'circle' # None

    show_fig = False # True

    #eps = 4 
    #adv_type = 'fgsm_iter'

    #adv_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_train_heldout/%s/eps_%d/' % (adv_type, eps)
    #adv_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_masks/imagenet_train_heldout/%s/eps_%d/defaults_iter_%d/' % (adv_type, eps, num_iters)
    #true_mask_dir = '/data/ruthfong/perturb_explanations_adversarial/true_masks/imagenet_train_heldout/defaults_iter_%d/' % num_iters
    #fig_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_comparison_figs/imagenet_train_heldout/%s/eps_%d/defaults_iter_%d/' % (adv_type, eps, num_iters)

    if not os.path.exists(adv_mask_dir):
        os.makedirs(adv_mask_dir)
    if not os.path.exists(true_mask_dir):
        os.makedirs(true_mask_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for ind in range(start_i, end_i):
        if os.path.exists(os.path.join(adv_mask_dir, '%d.png' % ind)):
            print '%s exists so skipping' % os.path.join(adv_mask_dir, '%d.png' % ind)
            continue
        start = time.time()
        true_label = labels[ind]
        true_path = paths[ind]
        adv_path = os.path.join(adv_dir, '%d.png' % ind)

        # compute adversarial label
        adv_img = transformer.preprocess('data', caffe.io.load_image(adv_path))
        net.blobs['data'].data[...] = adv_img
        net.forward()
        scores = np.squeeze(net.blobs['prob'].data)
        adv_label = np.argmax(scores)

        adv_mask = generate_learned_mask(net, transformer, adv_path, adv_label, given_gradient = given_gradient, 
                              norm_score = norm_score, num_iters = num_iters, lr = lr, l1_lambda = l1_lambda, 
                              l1_ideal = l1_ideal, l1_lambda_2 = l1_lambda_2, tv_lambda = tv_lambda, tv_beta = tv_beta, 
                              mask_scale = mask_scale, use_conv_norm = use_conv_norm, blur_mask = blur_mask, 
                              jitter = jitter, noise = noise, null_type = null_type, gpu = gpu, 
                              start_layer = 'data', end_layer = 'prob', plot_step = plot_step, debug = debug, 
                              fig_path = None, mask_path = None, verbose = verbose, show_fig = show_fig, 
                              mask_init_type = mask_init_type, num_top = num_top, labels = labels_desc)

        if os.path.exists(os.path.join(true_mask_dir,'%d.png' % ind)):
            true_mask = caffe.io.load_image(os.path.join(true_mask_dir,'%d.png' % ind)) 
        else:
            true_mask = generate_learned_mask(net, transformer, true_path, true_label, given_gradient = given_gradient, 
                                  norm_score = norm_score, num_iters = num_iters, lr = lr, l1_lambda = l1_lambda, 
                                  l1_ideal = l1_ideal, l1_lambda_2 = l1_lambda_2, tv_lambda = tv_lambda, tv_beta = tv_beta, 
                                  mask_scale = mask_scale, use_conv_norm = use_conv_norm, blur_mask = blur_mask, 
                                  jitter = jitter, noise = noise, null_type = null_type, gpu = gpu, 
                                  start_layer = 'data', end_layer = 'prob', plot_step = plot_step, debug = debug, 
                                  fig_path = None, mask_path = None, verbose = verbose, show_fig = show_fig, 
                                  mask_init_type = mask_init_type, num_top = num_top, labels = labels_desc)

        # save comparison figures for first 100
        #if ind < 100:
        #    try:
        #        f, ax = plt.subplots(1,3)
        #        f.set_size_inches(12,4)
        #        ax[0].imshow(caffe.io.load_image(adv_path))
        #        ax[0].set_title('adv img (true: %s, adv: %s)' % (get_short_class_name(true_label), 
        #                                                        get_short_class_name(adv_label)))
        #        ax[1].imshow(true_mask)
        #        ax[1].set_title('true mask (defaults, iters = %d)' % num_iters)
        #        ax[2].imshow(adv_mask)
        #        ax[2].set_title('adv mask (defaults, iters = %d)' % num_iters)
        #        for a in ax:
        #            a.set_xticks([])
        #            a.set_yticks([])
        #        plt.savefig(os.path.join(fig_dir, '%d.png' % ind))
        #        plt.close()
        #        #plt.show()
        #    except:
        #        pass
        
        # save saliency masks
        imsave(os.path.join(adv_mask_dir, '%d.png' % ind), adv_mask)
        if not os.path.exists(os.path.join(true_mask_dir, '%d.png' % ind)):
            imsave(os.path.join(true_mask_dir, '%d.png' % ind), true_mask)
        print ind, time.time() - start, os.path.join(adv_mask_dir, '%d.png' % ind)


def generate_adversarial_examples(net, paths, labels, adv_img_dir, start_i = 0, end_i = 5000):
    transformer = get_ILSVRC_net_transformer(net)

    #fig_path = 'adversarial_accuracy.png'
    save_adv_img = True
    #epsilons = np.array([2,4,8,12,16,20,24,28,32,40,48,56,64,96,112,128])
    epsilons = np.array([1,2,4,8,12,16])
    methods = ['fgsm', 'one_step', 'fgsm_iter', 'one_step_iter']

    top = 'prob'
    show_fig = False

    num_top1 = np.zeros((len(methods), len(epsilons)))
    num_top5 = np.zeros((len(methods), len(epsilons)))

    for i in range(start_i, end_i):
        start = time.time()
        for j in range(len(methods)):
            m = methods[j]
            if m == 'clean':
                (adv_label, is_top1, is_top5) = check_clean_image(net, transformer, paths[i], 
                                                                  labels[i], top)
                if is_top1:
                    num_top1[j] += 1
                if is_top5:
                    num_top5[j] += 1
                continue
            for k in range(len(epsilons)):
                e = epsilons[k]
                adv_dir = os.path.join(adv_img_dir, m, 'eps_%d' % e)
                adv_path = os.path.join(adv_dir, '%d.png' % i)
                if os.path.exists(adv_path):
                    print '%s exists so skipping' % adv_path
                    continue
                if 'fgsm' in m:
                    method = 'fgsm'
                elif 'one_step' in m:
                    method = 'one_step'
                else:
                    assert(False)
                    
                if 'iter' in m:    
                    alpha = 1.0
                    step = None
                    eps = e
                else:
                    alpha = e
                    step = 1
                    eps = None
                    
                adv_label_type = 'min' # only used for one_step, g.t. label used for fgsm

                (adv_img, adv_label, is_top1, is_top5) = compute_adversarial_example(net, transformer, 
                                                             paths[i], labels[i], 
                                                             method = method, step = step, 
                                                             alpha = alpha, eps = eps, 
                                                             adv_label = adv_label_type, 
                                                             top = top, show_fig = show_fig)
                if save_adv_img:
                    adv_dir = os.path.join(adv_img_dir, m, 'eps_%d' % e)
                    adv_path = os.path.join(adv_dir, '%d.png' % i)
                    if not os.path.exists(adv_dir):
                        os.makedirs(adv_dir)
                    imsave(adv_path, adv_img)
                if is_top1:
                    num_top1[j][k] += 1
                if is_top5:
                    num_top5[j][k] += 1
                continue
        print i, time.time() - start


    #f, ax = plt.subplots(1,2)
    #f.set_size_inches(12,4)
    #ax[0].plot(epsilons, np.transpose(num_top1/float(num_examples)), marker='o')
    #ax[0].set_ylabel('top-1 accuracy')
    #ax[1].plot(epsilons, np.transpose(num_top5/float(num_examples)), marker='o')
    #ax[1].set_ylabel('top-5 accuracy')
    #for a in ax:
    #    a.set_ylim([0,1])
    #    a.set_xlabel(r'$\epsilon$')
    #    a.legend(methods)
    #    a.set_title(r'$N = %d$' % num_examples)
    #plt.savefig(fig_path)
    #plt.close()
    #plt.show()


def generate_adversarial_graph(net, paths, labels):
    transformer = get_ILSVRC_net_transformer(net)

    adv_img_dir = '/data/ruthfong/perturb_explanations_adversarial/adv_imgs/imagenet_train_heldout'
    #epsilons = np.array([2,4,8,12,16,20,24,28,32,40,48,56,64,96,112,128])
    epsilons = np.array([1,2,4,8,12,16])
    methods = ['clean', 'fgsm', 'one_step', 'fgsm_iter', 'one_step_iter']

    top = 'prob'
    show_fig = False

    num_examples = 5000 

    fig_path = 'adversarial_accuracy_%d.png' % num_examples

    num_top1 = np.zeros((num_examples, len(methods), len(epsilons)))
    num_top5 = np.zeros((num_examples, len(methods), len(epsilons)))

    adv_acc_top1_ind_f = 'adversarial_accuracy_top1_indicator_%d.npy' % num_examples
    adv_acc_top5_ind_f = 'adversarial_accuracy_top5_indicator_%d.npy' % num_examples
    adv_acc_top1_sum_f = 'adversarial_accuracy_top1_summary_%d.txt' % num_examples
    adv_acc_top5_sum_f = 'adversarial_accuracy_top5_summary_%d.txt' % num_examples

    if os.path.exists(adv_acc_top1_ind_f) and os.path.exists(adv_acc_top5_ind_f):
        num_top1 = np.load(adv_acc_top1_ind_f)
        num_top5 = np.load(adv_acc_top5_ind_f)
    else:
        for i in range(num_examples):
            start = time.time()
            for j in range(len(methods)):
                m = methods[j]
                if m == 'clean':
                    (adv_label, is_top1, is_top5) = check_clean_image(net, transformer, paths[i],
                                                                      labels[i], top)
                    if is_top1:
                        num_top1[i][j] = 1
                    if is_top5:
                        num_top5[i][j] = 1
                    continue
                for k in range(len(epsilons)):
                    e = epsilons[k]
                    adv_dir = os.path.join(adv_img_dir, m, 'eps_%d' % e)
                    adv_path = os.path.join(adv_dir, '%d.png' % i)
                    (adv_label, is_top1, is_top5) = check_clean_image(net, transformer, adv_path,
                                                                      labels[i], top) 
                    if is_top1:
                        num_top1[i][j][k] = 1
                    if is_top5:
                        num_top5[i][j][k] = 1
                    continue
            print i, time.time() - start

        np.save('adversarial_accuracy_top1_indicator_%d.npy' % num_examples, num_top1)
        np.save('adversarial_accuracy_top5_indicator_%d.npy' % num_examples, num_top5)
        np.savetxt('adversarial_accuracy_top1_summary_%d.txt' % num_examples, np.sum(num_top1,0))
        np.savetxt('adversarial_accuracy_top5_summary_%d.txt' % num_examples, np.sum(num_top5,0))

    f, ax = plt.subplots(1,2)
    f.set_size_inches(12,4)
    ax[0].plot(epsilons, np.transpose(np.sum(num_top1,0)/float(num_examples)), marker='o')
    ax[0].set_ylabel('top-1 accuracy')
    ax[1].plot(epsilons, np.transpose(np.sum(num_top5,0)/float(num_examples)), marker='o')
    ax[1].set_ylabel('top-5 accuracy')
    for a in ax:
        a.set_ylim([0,1])
        a.set_xlabel(r'$\epsilon$')
        a.legend(methods)
        a.set_title(r'$N = %d$' % num_examples)
    plt.savefig(fig_path)
    plt.close()
    #plt.show()

if __name__ == '__main__':
    main()

