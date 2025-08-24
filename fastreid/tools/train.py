#!/usr/bin/env python3
# encoding: utf-8
"""
FastReID Training Script

This script provides a command-line interface for training FastReID models.
It's based on the original tools/train_net.py but adapted for the package structure.

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
from fastreid.utils.collect_env import collect_env_info
from fastreid.utils.env import seed_all_rng
from fastreid.utils.logger import setup_logger


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    default_setup(cfg, args)
    return cfg


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    """
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank = getattr(args, "local_rank", 0)
    setup_logger(output_dir, distributed_rank=rank, name="fastreid")

    logger = setup_logger(__name__)

    logger.info("Rank of current process: {}. World size: {}".format(rank, args.num_gpus))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)


def main(args=None):
    """
    Main training function.
    
    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    if args is None:
        args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def cli_main():
    """Entry point for the fastreid-train command."""
    args = default_argument_parser().parse_args()
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
