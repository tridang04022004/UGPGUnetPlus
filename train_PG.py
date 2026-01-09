import argparse
from PG_trainer import ProgressiveTrainer


def main():
    parser = argparse.ArgumentParser(
        description='PG-UNet+'
    )
    
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--stage-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-decay', type=float, default=0.95,)
    parser.add_argument('--lr-decay-every', type=int, default=40)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='./outputs_PG')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='nuclei-segmentation')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    args = parser.parse_args()
    trainer = ProgressiveTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
