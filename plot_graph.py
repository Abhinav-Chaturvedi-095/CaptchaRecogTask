import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_and_save_metrics(csv_path, save_dir="training_plots"):
    # Load the training metrics CSV
    df = pd.read_csv(csv_path)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot Losses
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))  # Save plot
    plt.close()  # Close figure to prevent overlapping plots

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))  # Save plot
    plt.close()  # Close figure

    print(f"Plots saved in {save_dir}")


plot_and_save_metrics("/home/abhinav/TASK/models_pth/training_metrics.csv")
