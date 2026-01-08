import argparse
from PG_trainer import ProgressiveTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Progressive Growing Training for PGU-Net on Herlev nuclei dataset'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=160,
                        help='Total number of training epochs (40 per stage recommended)')
    parser.add_argument('--stage-epochs', type=int, default=40,
                        help='Number of epochs per stage before upgrading model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.95,
                        help='Learning rate decay factor')
    parser.add_argument('--lr-decay-every', type=int, default=40,
                        help='Decay learning rate every N epochs')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs_PG',
                        help='Output directory for checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Wandb arguments
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='nuclei-segmentation',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = ProgressiveTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
