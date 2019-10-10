__author__ = 'kang huang'

from flask import Flask, render_template, request, jsonify, json
# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from scene_parsing import TestDataset
from scene_parsing import ModelBuilder, SegmentationModule
from scene_parsing import plot_colortable, colorEncode, find_recursive, setup_logger
from scene_parsing.lib.nn import user_scattered_collate, async_copy_to
from scene_parsing.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pprint import pprint

IP = '0.0.0.0:8080'

colors = loadmat('data/color150.mat')['colors']
#print (colors)
#print (mcolors.BASE_COLORS)
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

examples = [
    [['wall', '54.03'], ['floor', '4.99'], ['ceiling', '3.51'], ['table', '16.2'], ['plant', '1.28'], ['chair','11.75']],
    [['wall', '34.13'], ['floor', '8.01'], ['ceiling', '15.5'], ['table', '17.76'], ['plant', '5.86'], ['chair', '6.94']],    
    [['wall', '42.94'], ['floor', '14.92'], ['ceiling', '12.86'], ['chair', '10.4'], ['sofa', '4.62']],
]
def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    ratios = [0]*len(names)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        ratios[uniques[idx] + 1] = ratio
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color, COLORS, class_ratio = colorEncode(pred, colors, names, ratios)
    pred_color = pred_color.astype(np.uint8)

    # plot_colortable(COLORS, "Customized Colors",
    #             sort_colors=False, emptycols=1)
    # aggregate images and save
    #im_vis = np.concatenate((img, pred_color), axis=1)
    im_vis = pred_color
    img_name = info.split('/')[-1]
    # print (os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))
    Image.fromarray(im_vis).save(
        os.path.join('static', cfg.TEST.result, img_name.replace('.jpg', '.png')))
    return list(class_ratio[0]), list(class_ratio[1])

def test(segmentation_module, loader):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            #scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                #feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        classes, ratios = visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)
    print('Inference done!')
    return classes, ratios

def main(cfg):
    #torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    #segmentation_module.cuda()

    # Main loop
    return test(segmentation_module, loader_test)

 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/imgseg', methods=['POST'])
def segmentation():
    upload_filename = None
    # print ('filename', request.values['url'])
    if request.files['data']:
        upload_data = request.files['data'] 
 
        upload_filename =  upload_data.filename
        path = os.path.join('upload', upload_filename)
        upload_data.save(path)
        cfg.list_test = [{'fpath_img': path}] 
        classes, ratios = main(cfg)
    elif request.values['url']:
        url = request.values['url']
        upload_filename = url.replace('http://'+IP+'/static/teaser/', '')
        idx = int(upload_filename.strip('.jpg'))
        ex = list(zip(*examples[idx]))
        classes, ratios = list(ex[0]), list(ex[1])
    # print ('filename', upload_filename)

    #return json.dumps({"object": {"classes": ["earth", "sky", "tree", "field", "mountain", "building", "road", "plant", "grass", "fence", "wall", "animal", "rock", "water", "path"], "segment": "2.png", "ratios": ["34.52%", "21.96%", "16.31%", "5.49%", "4.64%", "4.41%", "4.11%", "3.73%", "2.28%", "0.60%", "0.53%", "0.53%", "0.49%", "0.27%", "0.13%"]}})
    return json.dumps({"object": {"classes":classes, "ratios": ratios, "segment": upload_filename.replace('.jpg', '.png')}})

@app.route('/select', methods=['POST'])
def cache():
    print (request.data)
    return json.dumps({"object": {"classes": ["earth", "sky", "tree", "field", "mountain", "building", "road", "plant", "grass", "fence", "wall", "animal", "rock", "water", "path"], "segment": "2.png", "ratios": ["34.52%", "21.96%", "16.31%", "5.49%", "4.64%", "4.41%", "4.11%", "3.73%", "2.28%", "0.60%", "0.53%", "0.53%", "0.49%", "0.27%", "0.13%"]}})

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image paths, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet18dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    print (cfg.TEST.result)
    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)
    ip, port = IP.split(':')
    app.run(host=ip, port=port, debug=True)
    