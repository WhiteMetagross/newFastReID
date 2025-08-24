#!/usr/bin/env python3
# encoding: utf-8
"""
FastReID Demo Script

This script provides a command-line interface for running FastReID demos and feature extraction.
Based on the original demo/demo.py but adapted for the package structure.

@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.file_io import PathManager


class FeatureExtractionDemo:
    """
    Demo class for feature extraction using FastReID models.
    """
    
    def __init__(self, cfg, parallel: bool = False):
        """
        Args:
            cfg: FastReID config
            parallel: Whether to use parallel processing
        """
        self.cfg = cfg.clone()
        self.parallel = parallel
        
        self.model = build_model(cfg)
        self.model.eval()
        
        if cfg.MODEL.WEIGHTS:
            Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        
    def run_on_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract features from a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Feature tensor
        """
        # Preprocess image
        height, width = image.shape[:2]
        image = cv2.resize(image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]))
        image = image[:, :, ::-1]  # BGR to RGB
        image = image.copy()  # Make a copy to avoid negative stride issues
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image)
            
        return features


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False  # Don't load pretrained weights
    cfg.freeze()
    return cfg


def get_parser():
    """Create argument parser for demo script."""
    parser = argparse.ArgumentParser(description="Feature extraction with FastReID models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        required=True,
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features: torch.Tensor) -> np.ndarray:
    """
    Postprocess features for saving.
    
    Args:
        features: Feature tensor
        
    Returns:
        Normalized features as numpy array
    """
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def main(args=None):
    """
    Main demo function.
    
    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    if args is None:
        args = get_parser().parse_args()
        
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    PathManager.mkdirs(args.output)
    
    if args.input:
        if PathManager.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
            
        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image {path}")
                continue
                
            feat = demo.run_on_image(img)
            feat = postprocess(feat)
            
            output_path = os.path.join(
                args.output, 
                os.path.basename(path).split('.')[0] + '.npy'
            )
            np.save(output_path, feat)
            
        print(f"Features saved to {args.output}")
    else:
        print("No input images specified. Use --input to specify image paths.")


def cli_main():
    """Entry point for the fastreid-demo command."""
    main()


if __name__ == '__main__':
    cli_main()
