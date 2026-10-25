"""
train_model.py - Fine-tunes YOLO on the balanced ArUco + hard negative dataset.
Includes:
  - GPU acceleration (RTX 3050 CUDA device 0).
  - Background hard negative learning (suppresses face/clothing false alarms).
  - Model checkpointing and safe backup.
"""
import argparse
from datetime import datetime
import os
import shutil
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model with hard negatives")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320)")
    parser.add_argument("--weights", type=str, default="best.pt", help="Starting weights (default: best.pt)")
    parser.add_argument("--replace", action="store_true", help="Replace best.pt with fine-tuned model upon completion")
    args = parser.parse_args()

    yaml_path = os.path.abspath(os.path.join("dataset", "dataset.yaml"))
    if not os.path.exists(yaml_path):
        print(f"Error: dataset config not found at {yaml_path}. Run build_dataset.py first!")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60)
    print(f"Starting YOLO Training on {device_name} (device={device})")
    print(f"Base Weights: {args.weights}")
    print(f"Dataset:      {yaml_path}")
    print(f"Image Size:   {args.imgsz}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch}")
    print("=" * 60)

    # Load existing model
    model = YOLO(args.weights)

    # Train with balanced hard negatives
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        project="runs/train",
        name="finetuned",
        patience=10,
        save=True,
        verbose=True
    )

    trained_best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(trained_best):
        out_model = "best_finetuned.pt"
        shutil.copy2(trained_best, out_model)
        print("\n" + "=" * 60)
        print(f"Fine-tuning complete! Model saved to: {out_model}")

        if args.replace:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"best_backup_{timestamp}.pt"
            shutil.copy2("best.pt", backup_name)
            shutil.copy2(out_model, "best.pt")
            print(f"Backed up original model to: {backup_name}")
            print(f"Replaced best.pt with {out_model}")

        print("You can now test the model by running:")
        print("  python main.py")
        print("=" * 60)
    else:
        print("Warning: Could not locate trained weights file.")


if __name__ == "__main__":
    main()
