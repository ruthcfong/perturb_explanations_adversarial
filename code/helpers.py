import caffe
import numpy as np
from bs4 import BeautifulSoup
import os, time
from shutil import copyfile
import torch
from torch.autograd import Variable

from defaults import (caffe_dir, alexnet_prototxt, alexnet_model, googlenet_prototxt, googlenet_model, googlenet_voc_prototxt, googlenet_voc_model, 
    googlenet_coco_prototxt, googlenet_coco_model, vgg16_prototxt, vgg16_model)

def create_heldout_annotated_dir(old_ann_dir, new_heldout_ann_dir, imdb='../data/ilsvrc12/annotated_train_heldout_imdb.txt'):
    (paths, _) = read_imdb(imdb)
    if not os.path.exists(new_heldout_ann_dir):
        os.makedirs(new_heldout_ann_dir)
    
    for i in range(len(paths)):
        old_ann_file = os.path.join(old_ann_dir, paths[i].split('/')[-2], paths[i].split('/')[-1]).split('.')[0] + '.xml'
        new_ann_file = os.path.join(new_heldout_ann_dir, '%.6d.xml' % i)
        copyfile(old_ann_file, new_ann_file)

def write_imdb(split_dir, gt_file, out_file, idx_file = None):   
    '''
    write_imdb('/data/datasets/ILSVRC2012/images/train', '../../../data/ilsvrc12/annotated_train_heldout.txt', '../../../data/ilsvrc12/annotated_train_heldout_imdb.txt')
    '''
    f = open(gt_file)
    rel_paths = []
    labels = []
    for line in f.readlines():
        s = line.split()
        rel_paths.append(s[0])
        labels.append(int(s[1]))

    abs_paths = []
    for p in rel_paths:
        abs_paths.append(os.path.join(split_dir, p))

    out_f = open(out_file, 'w')
    for i in range(len(abs_paths)):
        out_f.write('%s %d\n' % (abs_paths[i], labels[i]))
    out_f.close()

def read_imdb(imdb_file):
    '''
    (paths, labels) = read_imdb('/home/ruthfong/packages/caffe/data/ilsvrc12/train_imdb.txt')
    '''
    f = open(imdb_file)
    paths = []
    labels = []
    for line in f.readlines():
        s = line.split()
        paths.append(s[0])
        labels.append(int(s[1]))
        
    return (np.array(paths), np.array(labels))

def create_animal_parts_imdb(out_file = None, foot_num_file = None, eye_num_file = None, require_eye_and_foot = True, min_per_class = 10,
                             animal_parts_ann_dir = '/data/ruthfong/ILSVRC2012/animal_parts_dataset/xml/foot',
                             train_img_dir = '/data/datasets/ILSVRC2012/images/train',
                             val_img_dir = '/data/ruthfong/ILSVRC2012/images/val', 
                             val_imdb_path = '/users/ruthfong/packages/caffe/data/ilsvrc12/val_imdb.txt', 
                             synsets = np.loadtxt('/users/ruthfong/packages/caffe/data/ilsvrc12/synsets.txt', str, delimiter='\t'),
                             foot_dir = '/data/ruthfong/ILSVRC2012/animal_parts_dataset/xml/foot',
                             eye_dir = '/data/ruthfong/ILSVRC2012/animal_parts_dataset/xml/eye'):
    animal_parts_ann_paths = os.listdir(animal_parts_ann_dir)
    imagenet_class = animal_parts_ann_paths[0].split('_')[0]
    (val_paths, val_labels) = read_imdb(val_imdb_path)
    animal_parts_img_paths = []
    animal_parts_labels = []
    for i in range(len(animal_parts_ann_paths)):
        f = animal_parts_ann_paths[i]
        file_name = f.strip('.xml') + '.JPEG'
        imagenet_class = f.split('_')[0]
        if imagenet_class == 'ILSVRC2012': # validation set
            full_path = os.path.join(val_img_dir, file_name)
        else: # training set
            full_path = os.path.join(train_img_dir, imagenet_class, file_name)
        if not os.path.exists(full_path):
            print '%s does not exist so skipping' % full_path
            continue

        if imagenet_class == 'ILSVRC2012':
            img_i = int(f.strip('.xml').split('_')[-1]) - 1
            assert(val_paths[img_i].split('/')[-1] == file_name)
            label = val_labels[img_i]
        else:
            label = np.where(synsets == imagenet_class)[0][0]
        
        animal_parts_img_paths.append(full_path)
        animal_parts_labels.append(label)

    animal_parts_img_paths = np.array(animal_parts_img_paths)
    animal_parts_labels = np.array(animal_parts_labels)

    if require_eye_and_foot:
        ann_foot_paths = [os.path.join(foot_dir, f) for f in os.listdir(foot_dir)]
        ann_eye_paths = [os.path.join(eye_dir, f) for f in os.listdir(foot_dir)]
        both_idx = []
        foot_num = []
        eye_num = []
        for i in range(len(animal_parts_labels)):
            f = animal_parts_img_paths[i].split('/')[-1].strip('.JPEG') + '.xml'
            ann_foot_path = os.path.join(foot_dir, f)
            ann_eye_path = os.path.join(eye_dir, f)
            foot_objs = load_objs(ann_foot_path)
            eye_objs = load_objs(ann_eye_path)
            if 'foot' in foot_objs:
                foot_num.append(len(foot_objs['foot']))
            else:
                foot_num.append(0)
            if 'eye' in eye_objs:
                eye_num.append(len(eye_objs['eye']))
            else:
                eye_num.append(0)
            if 'foot' in foot_objs and 'eye' in eye_objs:
                both_idx.append(i)
       
        foot_num = np.array(foot_num)
        eye_num = np.array(eye_num)
 
        print 'Number of images with at least one eye and foot: %d' % len(both_idx)

        unique_parts_labels = np.unique(animal_parts_labels[both_idx])

        print 'Number of classes with at least one eye and foot: %d' % len(unique_parts_labels)
        if min_per_class is not None:
            thres_inds = []
            num_thres = 0
            for i in range(len(unique_parts_labels)):
                label = unique_parts_labels[i]
                inds = np.intersect1d(np.where(animal_parts_labels==label)[0], both_idx)
                if len(inds) > min_per_class:
                    thres_inds.extend(inds)
                    num_thres += 1
            
            print 'Number of classes with at least %d images containing at least one eye and foot: %d' % (min_per_class, num_thres)
            both_inds = thres_inds
        animal_parts_img_paths = animal_parts_img_paths[both_inds]
        animal_parts_labels = animal_parts_labels[both_inds]
        foot_num = foot_num[both_inds]
        eye_num = eye_num[both_inds]

    print 'Number of images in imdb: %d' % len(animal_parts_labels)

    if out_file is not None:
        out_f = open(out_file, 'w')
        for i in range(len(animal_parts_img_paths)):
            out_f.write('%s %d\n' % (animal_parts_img_paths[i], animal_parts_labels[i]))
        out_f.close()

    if foot_num_file is not None:
        out_f = open(foot_num_file, 'w')
        for i in range(len(foot_num)):
            out_f.write('%s %d\n' % (animal_parts_img_paths[i], foot_num[i]))
        out_f.close()

    if eye_num_file is not None:
        out_f = open(eye_num_file, 'w')
        for i in range(len(eye_num)):
            out_f.write('%s %d\n' % (animal_parts_img_paths[i], eye_num[i]))
        out_f.close()
        
    return np.array(animal_parts_img_paths), np.array(animal_parts_labels)

def get_net(net_type):
    if net_type == 'alexnet':
        net = caffe.Net(alexnet_prototxt, alexnet_model, caffe.TEST)
    elif net_type == 'vgg16':
        net = caffe.Net(vgg16_prototxt, vgg16_model, caffe.TEST)
    elif net_type == 'googlenet':
        net = caffe.Net(googlenet_prototxt, googlenet_model, caffe.TEST)
    elif net_type == 'googlenet_voc':
        net = caffe.Net(googlenet_voc_prototxt, googlenet_voc_model, caffe.TEST)
    elif net_type == 'googlenet_coco':
        net = caffe.Net(googlenet_coco_prototxt, googlenet_coco_model, caffe.TEST)	
    else:
        assert(False)
   
    net_shape = net.blobs['data'].data.shape
    net.blobs['data'].reshape(1,3,net_shape[2],net_shape[3])
    return net

def get_ILSVRC_mean(print_mean = False):
    mu = np.load(os.path.join(caffe_dir, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)
    if print_mean:
        print 'mean-subtracted values:', zip('BGR', mu)
    return mu

def get_ILSVRC_net_transformer(net):
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mu = get_ILSVRC_mean()
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer

def get_VOC_mean():
    mu = np.load('/data/ruthfong/VOCdevkit/VOC2007/caffe/mean.npy')
    mu = mu.mean(1).mean(1)
    return mu

def get_VOC_net_transformer(net):
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mu = get_VOC_mean()
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer

def get_COCO_net_transformer(net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    return transformer

def get_ILSVRC_net_transformer_with_shape(shape):
    transformer = caffe.io.Transformer({'data':shape})
    transformer.set_transpose('data', (2,0,1))
    mu = get_ILSVRC_mean()
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer

def transform_batch(transformer_shape, data):
    transformer = get_ILSVRC_net_transformer_with_shape(transformer_shape)
    output = np.empty((data.shape[-1], transformer_shape[1], transformer_shape[2], transformer_shape[3]))
    for i in range(data.shape[-1]):
        output[i,:,:,:] = transformer.preprocess('data', data[:,:,:,i])

def load_annotation(ann_filename):
    """
    Load annotation file.

    Args:
        annotation file path

    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    xml = ""
    with open(ann_filename) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "lxml")

def load_objs(ann_filename):
    anno = load_annotation(ann_filename)
    data = {}
    objs = anno.findAll('object')
    for obj in objs:
        obj_names = obj.findChildren('name')
        for name_tag in obj_names:
            label = str(name_tag.contents[0])
            #fname = anno.findChild('filename').contents[0]
            bbox = obj.findChildren('bndbox')[0]
            xmin = int(bbox.findChildren('xmin')[0].contents[0])
            ymin = int(bbox.findChildren('ymin')[0].contents[0])
            xmax = int(bbox.findChildren('xmax')[0].contents[0])
            ymax = int(bbox.findChildren('ymax')[0].contents[0])
            if label in data:
                data[label].append([xmin, ymin, xmax, ymax])
            else:
                data[label] = [[xmin, ymin, xmax, ymax]]
            for point in obj.findChildren('point'):
                x = float(point.findChildren('x')[0].contents[0])
                y = float(point.findChildren('y')[0].contents[0])
                keypoint = str(point.findChildren('class')[0].contents[0])
                if keypoint in data:
                    data[keypoint].append([x,y])
                else:
                    data[keypoint] = [[x,y]]
    return data

def compute_overlap(bb, objs, label):
    ov_vector = []
    for k in objs.keys():
        if k != label:
            continue
        for i in range(len(objs[k])):
            bbgt = objs[k][i]
            bi=[max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), 
                min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw=bi[2]-bi[0]+1;
            ih=bi[3]-bi[1]+1;
            ov = -1;
            if iw>0 and ih>0:
                # compute overlap as area of intersection / area of union
                ua=((bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)
                       +(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)
                       -iw*ih)
                ov=iw*ih/float(ua);
            ov_vector.append(ov)
    return ov_vector

def compute_localization_results(bb_file, ann_paths, verbose=False, synsets=np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synsets.txt'), 
    str, delimiter='\t'), reverse_indexing=np.loadtxt('../data/ilsvrc12/synset_order_to_ascii_order.txt', dtype=int)):
    if verbose:
        print 'Loading bounding boxes from', bb_file
    bb_data = np.loadtxt(bb_file)
    bb_labels = bb_data[:,0].astype(int)
    bbs = bb_data[:,1:].astype(int)

    num_examples = len(ann_paths)

    assert(num_examples == len(bb_labels))

    blacklist = np.zeros(num_examples)
    if num_examples == 50000:
        blacklist_idx = np.loadtxt('../../../data/ilsvrc12/ILSVRC2014_clsloc_validation_blacklist.txt', dtype=int) - 1
        blacklist[blacklist_idx] = 1

    res = np.zeros(num_examples, dtype=int)
    overlap = np.zeros(num_examples)
    for i in range(num_examples):
        if blacklist[i]:
            continue
        objs = load_objs(ann_paths[i])
        ov_vector = compute_overlap(bbs[i], objs, synsets[reverse_indexing[bb_labels[i]-1]])
        try:
            res[i] = max(ov_vector) < 0.5
        except:
            print i, ov_vector
        overlap[i] = max(ov_vector)
    acc = res.sum()/float(num_examples - blacklist.sum())
    if verbose:
        print 'Localization Accuracy:', acc
    return (acc, 1-res, overlap)

def find_labels(phrase, labels_desc = np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synset_words.txt'), str, delimiter='\t')):
    indicator = [phrase in label for label in labels_desc]
    try:
        return np.where(indicator)[0]
    except:
        return None
    
def get_short_class_name(label_i, labels_desc = np.loadtxt(os.path.join(caffe_dir, 'data/ilsvrc12/synset_words.txt'), str, delimiter='\t')
):
    return ' '.join(labels_desc[label_i].split(',')[0].split()[1:])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_state(fn, state, best_fn = None, is_best=False):
    torch.save(state, fn)
    if is_best:
        assert(best_fn is not None)
        copyfile(fn, best_fn)
    print 'saved state to', fn

class GetFirstChannel(object):
    def __call__(self, img):
        return img[0,:,:].unsqueeze(0)

def adjust_learning_rate(optimizer, epoch, starting_lr, freq = 30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = starting_lr * (0.1 ** (epoch // freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print output.size(), target.size()
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print pred, target
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_correct_indicator(output, target):
    topk=(1,)
    maxk = max(topk)
    batch_size = target.size(0)

    #print output.size(), target.size()
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print pred, target
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    k = topk[0]
    correct_k = correct[:k].view(-1).float().numpy()
    return correct_k

def get_correctly_classified(loader, model, cuda = False, print_freq = 100):
    model.eval()
    volatile=True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    start = time.time()
    indicator = []
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - start)

        input_var = (Variable(input.cuda(), volatile=volatile) if cuda
                     else Variable(input, volatile=volatile))
        target_var = (Variable(target.cuda(), volatile=volatile) if cuda
                     else Variable(target, volatile=volatile))
        output_var = model(input_var)
        indicator.extend(get_correct_indicator(output_var.data.cpu(), target))
        if i % print_freq == 0:
            print '%d/%d: avg acc=%.4f' % (i+1, len(loader), np.sum(indicator)/float(len(indicator)))  
    return indicator

def run_epoch(loader, model, criterion, optimizer, epoch, train = True, cuda = False,
        print_freq = 100, save_freq = 100, checkpoint_fn = None, best_fn = None):
    if train:
        model.train()
        volatile=False
    else:
        model.eval()
        volatile=True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    start = time.time()
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - start)

        input_var = (Variable(input.cuda(), volatile=volatile) if cuda
                     else Variable(input, volatile=volatile))
        target_var = (Variable(target.cuda(), volatile=volatile) if cuda
                     else Variable(target, volatile=volatile))
        output_var = model(input_var)
        loss = criterion(output_var, target_var)
        acc = accuracy(output_var.data.cpu(), target)
        losses.update(loss.data[0], input.size(0))
        accs.update(acc[0].numpy()[0], input.size(0))

        if train:
            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        batch_time.update(time.time() - start)

        if i % print_freq == 0:
            if train:
                print('Epoch {0}[{1}/{2}]\t'
                        'Time {batch_time.val:.3f} {batch_time.avg:.3f}\t'
                        'Data {data_time.val:.3f} {data_time.avg:.3f}\t'
                        'Loss {losses.val:.4f} {losses.avg:.4f}\t'
                        'Acc {accs.val:.4f} {accs.avg:.4f}\t'.format(epoch, i, len(loader),
                        batch_time=batch_time, data_time=data_time, losses=losses, accs=accs
                     ))
            else:
                print('Test [{}/{}]\t'
                  'Time {batch_time.val:.3f} {batch_time.avg:.3f}\t'
                  'Data {data_time.val:.3f} {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f} {loss.avg:.4f}\t'
                  'Acc {accs.val:.4f} {accs.avg:.4f}\t'.format(i, len(loader),
                                   batch_time=batch_time,
                                   data_time=data_time,
                                   loss=losses, accs=accs
                 ))

        if (save_freq is not None and i % save_freq == 0) and checkpoint_fn is not None:
            if train:
                state = {'epoch': epoch+1,
                         'batch':i+1,
                         'state_dict': model.state_dict(),
                         'avg_loss': losses.avg,
                }
                save_state(checkpoint_fn, state)

        start = time.time()

    if train:
        print ('Epoch {0}\t'
                'Avg Time {batch_time.avg:.3f}\t'
                'Avg Data {data_time.avg:.3f}\t'
                'Avg Loss {losses.avg:.4f}\t'
                'Avg Acc {accs.avg:.4f}\t'.format(epoch, batch_time=batch_time,
                    data_time=data_time, losses=losses, accs=accs))
    else:
        print ('Test \t'
                'Avg Time {batch_time.avg:.3f}\t'
                'Avg Data {data_time.avg:.3f}\t'
                'Avg Loss {losses.avg:.4f}\t'
                'Avg Acc {accs.avg:.4f}\t'.format(batch_time=batch_time,
                    data_time=data_time, losses=losses, accs=accs))

    return (losses.avg, accs.avg)
