#!/usr/bin/env python3
# encoding: utf-8
"""
FastReID Testing Script

This script provides a command-line interface for testing/evaluating FastReID models.

@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from typing import Any, Dict

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank = getattr(args, "local_rank", 0)
    setup_logger(output_dir, distributed_rank=rank, name="fastreid")
    
    return cfg


def main(args=None):
    """
    Main testing function.
    
    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    if args is None:
        parser = default_argument_parser()
        parser.add_argument(
            "--eval-only",
            action="store_true",
            help="perform evaluation only"
        )
        args = parser.parse_args()
    
    print("Command Line Args:", args)
    
    cfg = setup(args)
    
    # Force evaluation mode
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    
    model = DefaultTrainer.build_model(cfg)
    
    if cfg.MODEL.WEIGHTS:
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    else:
        print("Warning: No model weights specified. Using random initialization.")

    res = DefaultTrainer.test(cfg, model)
    return res


def cli_main():
    """Entry point for the fastreid-test command."""
    parser = default_argument_parser()
    parser.add_argument(
        "--eval-only",
        action="store_true", 
        help="perform evaluation only"
    )
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    cli_main()
