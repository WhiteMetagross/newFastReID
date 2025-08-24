import os
import sys
import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer


class FastReIDModelAnalyzer:
    
    def __init__(self):
        self.dataset_configs = {
            "Market1501": {"num_classes": 751, "name": "Market1501"},
            "DukeMTMC": {"num_classes": 702, "name": "DukeMTMC"}, 
            "DukeMTMCreID": {"num_classes": 702, "name": "DukeMTMCreID"},
            "MSMT17": {"num_classes": 1041, "name": "MSMT17"},
            "VeRiWild": {"num_classes": 30671, "name": "VeRiWild"},
            "VehicleID": {"num_classes": 13164, "name": "VehicleID"},
            "VeRi": {"num_classes": 576, "name": "VeRi"},
            "CUHK03": {"num_classes": 767, "name": "CUHK03"},
            "SenseReID": {"num_classes": 1404, "name": "SenseReID"},
            "GRID": {"num_classes": 250, "name": "GRID"},
            "iLIDS": {"num_classes": 119, "name": "iLIDS"},
            "PRID": {"num_classes": 200, "name": "PRID"},
        }
        
        self.backbone_patterns = {
            'resnet': ['layer1', 'layer2', 'layer3', 'layer4'],
            'resnext': ['layer1', 'layer2', 'layer3', 'layer4'],
            'resnest': ['layer1', 'layer2', 'layer3', 'layer4'],
            'densenet': ['denseblock', 'transition'],
            'mobilenet': ['features'],
            'efficientnet': ['blocks'],
            'swinformer': ['layers'],
            'vit': ['blocks', 'transformer']
        }
        
        self.architecture_mapping = {
            'sbs': 'sbs',
            'agw': 'agw', 
            'mgn': 'mgn',
            'bot': 'bot',
            'baseline': 'Baseline'
        }
    
    def analyze_model_architecture(self, model_path: str) -> Dict[str, Any]:
        print(f"Analyzing model: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            metadata = checkpoint.get('meta', {})
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = {}
        else:
            state_dict = checkpoint
            metadata = {}
        
        layer_names = list(state_dict.keys())
        print(f"Total parameters: {len(layer_names)}")
        
        analysis = {
            'meta_architecture': self._detect_meta_architecture(layer_names, metadata, model_path),
            'backbone_type': self._detect_backbone_type(layer_names, state_dict),
            'backbone_depth': self._detect_backbone_depth(layer_names),
            'has_ibn': self._detect_ibn(layer_names),
            'has_non_local': self._detect_non_local(layer_names),
            'has_se': self._detect_se(layer_names),
            'has_cbam': self._detect_cbam(layer_names),
            'feature_dim': self._detect_feature_dim(state_dict),
            'num_classes': self._detect_num_classes(state_dict),
            'pixel_normalization': self._extract_pixel_normalization(state_dict),
            'head_type': self._detect_head_type(layer_names),
            'has_bnneck': self._detect_bnneck(layer_names),
            'pooling_type': self._detect_pooling_type(layer_names),
            'loss_types': self._detect_loss_types(layer_names),
            'last_stride': self._detect_last_stride(state_dict),
            'input_size': self._detect_input_size(metadata, state_dict),
            'neck_feat': self._detect_neck_feat(layer_names),
            'with_center_loss': self._detect_center_loss(layer_names),
        }
        
        analysis['likely_dataset'] = self._guess_dataset(analysis['num_classes'])
        
        print("Analysis Results:")
        for key, value in analysis.items():
            print(f"   {key}: {value}")
        
        return analysis
    
    def _detect_meta_architecture(self, layer_names: List[str], metadata: Dict, model_path: str = None) -> str:
        if 'arch' in metadata:
            return metadata['arch']

        # First check layer names for architecture-specific patterns
        if any('mgn' in name.lower() for name in layer_names):
            return 'MGN'

        # For now, map all other architectures to Baseline since they're not implemented
        # SBS, AGW, BOT are typically training techniques or variations of Baseline
        return 'Baseline'
    
    def _detect_backbone_type(self, layer_names: List[str], state_dict: Dict) -> str:
        for backbone_type, patterns in self.backbone_patterns.items():
            if any(any(pattern in name for pattern in patterns) for name in layer_names):
                if backbone_type == 'resnet':
                    if any('resnext' in name.lower() for name in layer_names):
                        return 'resnext'
                    elif any('resnest' in name.lower() for name in layer_names):
                        return 'resnest'
                    else:
                        return 'resnet'
                return backbone_type
        return 'resnet'
    
    def _detect_backbone_depth(self, layer_names: List[str]) -> str:
        layer_counts = {}

        for name in layer_names:
            if 'backbone.' in name:
                for layer_num in ['layer1', 'layer2', 'layer3', 'layer4']:
                    if f'{layer_num}.' in name:
                        parts = name.split('.')
                        for i, part in enumerate(parts):
                            if part == layer_num and i + 1 < len(parts) and parts[i + 1].isdigit():
                                block_num = int(parts[i + 1])
                                if layer_num not in layer_counts:
                                    layer_counts[layer_num] = set()
                                layer_counts[layer_num].add(block_num)

        # Convert sets to counts
        layer_block_counts = {k: len(v) for k, v in layer_counts.items()}

        layer3_blocks = layer_block_counts.get('layer3', 0)
        layer2_blocks = layer_block_counts.get('layer2', 0)
        layer4_blocks = layer_block_counts.get('layer4', 0)

        print(f"   Layer counts: {layer_block_counts}")

        # ResNet architectures have specific block counts:
        # ResNet-18: [2, 2, 2, 2]
        # ResNet-34: [3, 4, 6, 3]
        # ResNet-50: [3, 4, 6, 3]
        # ResNet-101: [3, 4, 23, 3]
        # ResNet-152: [3, 8, 36, 3]

        if layer3_blocks >= 36:
            return '152x'
        elif layer3_blocks >= 23:
            return '101x'
        elif layer3_blocks >= 6:
            return '50x'
        elif layer3_blocks >= 2 and layer2_blocks >= 4:
            return '34x'
        elif layer3_blocks >= 2:
            return '18x'
        else:
            return '50x'  # Default fallback
    
    def _detect_ibn(self, layer_names: List[str]) -> bool:
        ibn_patterns = ['.IN.', '.ibn.', 'ibn_', '.IN_', '.instance_norm', '.inst_norm']
        return any(any(pattern in name for pattern in ibn_patterns) for name in layer_names)
    
    def _detect_non_local(self, layer_names: List[str]) -> bool:
        nl_patterns = ['NL_', 'non_local', 'nonlocal', '.nl.', '_nl_', 'NonLocal']
        return any(any(pattern in name for pattern in nl_patterns) for name in layer_names)
    
    def _detect_se(self, layer_names: List[str]) -> bool:
        se_patterns = ['.se.', '.SE.', 'squeeze', 'excitation', 'se_module']
        return any(any(pattern in name for pattern in se_patterns) for name in layer_names)
    
    def _detect_cbam(self, layer_names: List[str]) -> bool:
        cbam_patterns = ['.cbam.', 'CBAM', 'channel_att', 'spatial_att']
        return any(any(pattern in name for pattern in cbam_patterns) for name in layer_names)
    
    def _detect_feature_dim(self, state_dict: Dict) -> int:
        for name, tensor in state_dict.items():
            if 'heads' in name and 'weight' in name and len(tensor.shape) >= 2:
                if 'bnneck' in name or 'bottleneck' in name:
                    return tensor.shape[1]
                elif 'classifier' in name:
                    return tensor.shape[1]
            elif 'bnneck' in name and 'weight' in name and len(tensor.shape) >= 1:
                return tensor.shape[0]
        
        for name, tensor in state_dict.items():
            if 'backbone' in name and 'fc' in name and 'weight' in name:
                if len(tensor.shape) >= 2:
                    return tensor.shape[1]
        
        return 2048
    
    def _detect_num_classes(self, state_dict: Dict) -> Optional[int]:
        for name, tensor in state_dict.items():
            if ('classifier' in name or 'cls_layer' in name) and 'weight' in name and len(tensor.shape) == 2:
                return tensor.shape[0]
        return None
    
    def _extract_pixel_normalization(self, state_dict: Dict) -> Dict[str, List[float]]:
        pixel_norm = {}
        
        if 'pixel_mean' in state_dict:
            pixel_mean = state_dict['pixel_mean']
            if hasattr(pixel_mean, 'squeeze'):
                pixel_mean = pixel_mean.squeeze().tolist()
            pixel_norm['mean'] = pixel_mean
        
        if 'pixel_std' in state_dict:
            pixel_std = state_dict['pixel_std']
            if hasattr(pixel_std, 'squeeze'):
                pixel_std = pixel_std.squeeze().tolist()
            pixel_norm['std'] = pixel_std
        
        return pixel_norm
    
    def _detect_head_type(self, layer_names: List[str]) -> str:
        head_patterns = {
            'EmbeddingHead': ['embedding', 'heads.embedding'],
            'BNNeckHead': ['bnneck', 'bottleneck'],
            'ReductionHead': ['reduction', 'heads.reduction'],
            'LinearHead': ['linear', 'heads.linear'],
        }
        
        for head_type, patterns in head_patterns.items():
            if any(any(pattern in name for pattern in patterns) for name in layer_names):
                return head_type
        
        if any('heads.' in name for name in layer_names):
            return 'EmbeddingHead'
        
        return 'EmbeddingHead'
    
    def _detect_bnneck(self, layer_names: List[str]) -> bool:
        bnneck_patterns = ['bnneck', 'bottleneck', 'bn_neck']
        return any(any(pattern in name.lower() for pattern in bnneck_patterns) for name in layer_names)
    
    def _detect_pooling_type(self, layer_names: List[str]) -> str:
        pooling_patterns = {
            'gem': ['gem', 'gempoolp'],
            'avg': ['avgpool', 'adaptiveavgpool'],
            'max': ['maxpool', 'adaptivemaxpool'], 
            'attention': ['attention', 'att_pool'],
            'spp': ['spp', 'spatial_pyramid'],
        }
        
        for pool_type, patterns in pooling_patterns.items():
            if any(any(pattern in name.lower() for pattern in patterns) for name in layer_names):
                return f'{pool_type.upper()}Pool' if pool_type != 'gem' else 'gempoolP'
        
        return 'GlobalAvgPool'
    
    def _detect_loss_types(self, layer_names: List[str]) -> List[str]:
        loss_types = []
        
        loss_patterns = {
            'CrossEntropyLoss': ['classifier', 'cls_layer', 'ce_loss'],
            'TripletLoss': ['triplet', 'tri_loss'],
            'CenterLoss': ['center', 'center_loss'],
            'CircleLoss': ['circle', 'circle_loss'],
            'FocalLoss': ['focal', 'focal_loss'],
        }
        
        for loss_name, patterns in loss_patterns.items():
            if any(any(pattern in name.lower() for pattern in patterns) for name in layer_names):
                loss_types.append(loss_name)
        
        if not loss_types:
            loss_types = ["CrossEntropyLoss", "TripletLoss"]
        
        return loss_types
    
    def _detect_last_stride(self, state_dict: Dict) -> int:
        for name, tensor in state_dict.items():
            if 'layer4' in name and 'conv1.weight' in name:
                if len(tensor.shape) >= 4 and tensor.shape[2] == 1 and tensor.shape[3] == 1:
                    return 1
            elif 'layer4' in name and 'downsample.0.weight' in name:
                if len(tensor.shape) >= 4 and tensor.shape[2] == 1 and tensor.shape[3] == 1:
                    return 1
                elif len(tensor.shape) >= 4 and tensor.shape[2] == 2 and tensor.shape[3] == 2:
                    return 2
        return 1
    
    def _detect_input_size(self, metadata: Dict, state_dict: Dict) -> Tuple[int, int]:
        if 'input_size' in metadata:
            size = metadata['input_size']
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                return tuple(size[:2])
        
        for name, tensor in state_dict.items():
            if 'pos_embed' in name and len(tensor.shape) >= 3:
                h = w = int((tensor.shape[1] - 1) ** 0.5)
                return (256, 128) if h * w > 0 else (256, 128)
        
        return (256, 128)
    
    def _detect_neck_feat(self, layer_names: List[str]) -> str:
        if any('bnneck' in name or 'bottleneck' in name for name in layer_names):
            return 'before'
        return 'after'
    
    def _detect_center_loss(self, layer_names: List[str]) -> bool:
        return any('center' in name.lower() for name in layer_names)
    
    def _guess_dataset(self, num_classes: Optional[int]) -> str:
        if num_classes is None:
            return "Unknown"
        
        for dataset, info in self.dataset_configs.items():
            if info['num_classes'] == num_classes:
                return dataset
        
        return f"Custom ({num_classes} classes)"
    
    def _build_backbone_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        backbone_config = {
            'NAME': 'build_resnet_backbone',
            'NORM': 'BN',
            'DEPTH': analysis['backbone_depth'],
            'LAST_STRIDE': analysis['last_stride'],
            'FEAT_DIM': analysis['feature_dim'],
            'WITH_IBN': analysis['has_ibn'],
            'PRETRAIN': False,
        }
        
        if analysis['has_non_local']:
            backbone_config['WITH_NL'] = True
        
        if analysis['has_se']:
            backbone_config['WITH_SE'] = True
        
        if analysis['has_cbam']:
            backbone_config['WITH_CBAM'] = True
        
        backbone_type = analysis['backbone_type']
        if backbone_type == 'resnext':
            backbone_config['NAME'] = 'build_resnext_backbone'
        elif backbone_type == 'resnest':
            backbone_config['NAME'] = 'build_resnest_backbone'
        elif backbone_type == 'densenet':
            backbone_config['NAME'] = 'build_densenet_backbone'
        elif backbone_type == 'mobilenet':
            backbone_config['NAME'] = 'build_mobilenet_backbone'
        elif backbone_type == 'efficientnet':
            backbone_config['NAME'] = 'build_efficientnet_backbone'
        
        return backbone_config
    
    def _build_heads_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        heads_config = {
            'NAME': analysis['head_type'],
            'NORM': 'BN',
            'WITH_BNNECK': analysis['has_bnneck'],
            'POOL_LAYER': analysis['pooling_type'],
            'NECK_FEAT': analysis['neck_feat'],
            'CLS_LAYER': 'Linear',
        }
        
        if analysis['num_classes']:
            heads_config['NUM_CLASSES'] = analysis['num_classes']
        
        return heads_config
    
    def _build_losses_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        losses_config = {
            'NAME': tuple(analysis['loss_types']),
        }
        
        if 'CrossEntropyLoss' in analysis['loss_types']:
            losses_config['CE'] = {
                'EPSILON': 0.1,
                'SCALE': 1.0,
            }
        
        if 'TripletLoss' in analysis['loss_types']:
            losses_config['TRI'] = {
                'MARGIN': 0.3,
                'HARD_MINING': True,
                'NORM_FEAT': False,
                'SCALE': 1.0,
            }
        
        if 'CenterLoss' in analysis['loss_types']:
            losses_config['CENTER'] = {
                'SCALE': 0.0005,
                'ALPHA': 0.5,
            }
        
        if 'CircleLoss' in analysis['loss_types']:
            losses_config['CIRCLE'] = {
                'MARGIN': 0.25,
                'ALPHA': 256,
                'SCALE': 1.0,
            }
        
        if 'FocalLoss' in analysis['loss_types']:
            losses_config['FL'] = {
                'ALPHA': 0.25,
                'GAMMA': 2.0,
                'SCALE': 1.0,
            }
        
        return losses_config
    
    def generate_config(self, model_path: str, output_path: str, 
                       dataset_override: Optional[str] = None) -> str:
        
        analysis = self.analyze_model_architecture(model_path)
        
        dataset = dataset_override or analysis['likely_dataset']
        if dataset in self.dataset_configs:
            dataset_name = self.dataset_configs[dataset]['name']
        else:
            dataset_name = dataset.split('(')[0].strip() if '(' in dataset else dataset
        
        config = {
            'MODEL': {
                'META_ARCHITECTURE': analysis['meta_architecture'],
                'BACKBONE': self._build_backbone_config(analysis),
                'HEADS': self._build_heads_config(analysis),
                'LOSSES': self._build_losses_config(analysis),
            },
            'INPUT': {
                'SIZE_TRAIN': list(analysis['input_size']),
                'SIZE_TEST': list(analysis['input_size']),
            },
            'DATASETS': {
                'NAMES': [dataset_name],
                'TESTS': [dataset_name],
            },
            'TEST': {
                'EVAL_PERIOD': 50,
                'IMS_PER_BATCH': 128,
                'METRIC': 'cosine',
            },
            'CUDNN_BENCHMARK': True,
        }
        
        if analysis['pixel_normalization']:
            if 'mean' in analysis['pixel_normalization']:
                config['MODEL']['PIXEL_MEAN'] = analysis['pixel_normalization']['mean']
            if 'std' in analysis['pixel_normalization']:
                config['MODEL']['PIXEL_STD'] = analysis['pixel_normalization']['std']
        else:
            config['MODEL']['PIXEL_MEAN'] = [123.675, 116.28, 103.53]
            config['MODEL']['PIXEL_STD'] = [58.395, 57.12, 57.375]
        
        if analysis['with_center_loss']:
            config['SOLVER'] = {
                'CENTER_LOSS_WEIGHT': 0.0005,
                'CENTER_LR': 0.5,
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to: {output_path}")
        return str(output_path)
    
    def validate_config(self, config_path: str, model_path: str) -> bool:
        try:
            print("Validating configuration...")
            
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.DEVICE = 'cpu'
            cfg.freeze()
            
            model = build_model(cfg)
            print("   Model building successful")
            
            if os.path.exists(model_path):
                checkpointer = Checkpointer(model)
                checkpointer.load(model_path)
                print("   Weight loading successful")
            
            dummy_input = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
            model.eval()
            with torch.no_grad():
                inputs = {"images": dummy_input}
                output = model(inputs)
                print(f"   Inference test successful (output shape: {output.shape})")
            
            return True
            
        except Exception as e:
            print(f"   Validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced FastReID configuration generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_model_config.py --model market_sbs_R101-ibn.pth --output config.yml
  python generate_model_config.py --model veriwild_bot_R50-ibn.pth --output config.yml --dataset VeRiWild
  python generate_model_config.py --model model.pth --output config.yml --no-validate
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the pre-trained model file (.pth)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_config.yml",
        help="Output configuration file path"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Override dataset name"
    )
    parser.add_argument(
        "--no-validate", 
        action="store_true",
        help="Skip configuration validation"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        print("Enhanced FastReID Model Configuration Generator")
        print("=" * 60)
        
        analyzer = FastReIDModelAnalyzer()
        
        config_path = analyzer.generate_config(args.model, args.output, args.dataset)
        
        if not args.no_validate:
            success = analyzer.validate_config(config_path, args.model)
            if not success:
                print("Configuration validation failed, but file was still generated.")
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print(f"Model: {args.model}")
        print(f"Config: {config_path}")
        print("\nUsage:")
        print(f"   from fastreid.config import get_cfg")
        print(f"   cfg = get_cfg()")
        print(f"   cfg.merge_from_file('{config_path}')")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()