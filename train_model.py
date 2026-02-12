"""
Quick-start script to generate data and train the PINN.

Usage:
    python train_model.py [--scenarios 50] [--epochs 100] [--device cpu]
"""

import argparse
import os
import numpy as np
import torch

from cascade_predict.ml import CascadePINN, CascadeDataset, generate_training_data, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train cascade PINN")
    parser.add_argument("--scenarios", type=int, default=30, help="Number of training scenarios")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--physics-weight", type=float, default=0.3, help="Physics loss weight")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Model save directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate data
    print(f"Generating {args.scenarios} training scenarios...")
    inputs, targets = generate_training_data(n_scenarios=args.scenarios)
    print(f"  Total samples: {len(inputs)}")
    print(f"  Input shape:   {inputs.shape}")
    print(f"  Target shape:  {targets.shape}")

    # Train/val split
    n = len(inputs)
    split = int(0.8 * n)
    train_ds = CascadeDataset(inputs[:split], targets[:split])
    val_ds = CascadeDataset(inputs[split:], targets[split:])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Build model
    model = CascadePINN(hidden_dim=args.hidden_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    # Train
    trainer = Trainer(
        model=model,
        lr=args.lr,
        physics_weight=args.physics_weight,
        device=args.device,
    )

    print(f"\nTraining for {args.epochs} epochs...\n")
    history = trainer.fit(
        train_ds, val_ds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save
    save_path = os.path.join(args.save_dir, "cascade_pinn.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Final metrics
    final = history[-1]
    print(f"\nFinal metrics:")
    print(f"  Train loss: {final['train_total']:.4f}")
    print(f"  Val MAE:    {final.get('val_temp_mae', 'N/A')}")
    print(f"  Val Acc:    {final.get('val_runaway_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()
