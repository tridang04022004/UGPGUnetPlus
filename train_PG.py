import argparse
from PG_trainer import ProgressiveTrainer


def main():
    parser = argparse.ArgumentParser(
        description='PG-UNet+'
    )
    
    parser.add_argument('--data-dir', type=str, default='./dataREAL')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stage-epochs', type=int, default=40)
    parser.add_argument('--max-stage', type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--disable-plateau-detection', action='store_true')
    parser.add_argument('--plateau-patience', type=int, default=7)
    parser.add_argument('--plateau-min-delta', type=float, default=1e-4)
    parser.add_argument('--plateau-boundary-boost', type=float, default=0.15)
    parser.add_argument('--plateau-lr-factor', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='./outputs_PG')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='nuclei-segmentation')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--use-uncertainty-guidance', action='store_true', help='Enable uncertainty-guided iterative refinement training')
    parser.add_argument('--use-tta', action='store_true', help='Enable Test-Time Augmentation (TTA) during evaluation for more robust predictions')
    parser.add_argument('--tta-mode', type=str, default='standard', choices=['standard', 'flips_only', 'rotations_only', 'minimal'])
    args = parser.parse_args()
    trainer = ProgressiveTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
