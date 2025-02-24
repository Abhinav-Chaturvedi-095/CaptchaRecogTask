import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import pandas as pd
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Character Mapping
characters = string.digits  # "0123456789"
CTC_BLANK = len(characters)  
num_classes = len(characters) + 1  

char2idx = {char: idx for idx, char in enumerate(characters)}
idx2char = {idx: char for char, idx in char2idx.items()}
idx2char[CTC_BLANK] = '*'

# Text Encoding & Decoding
def encode_text(text):
    """Encodes text into numerical labels, ensuring fixed 6-digit format."""
    text = text.zfill(6)  # Ensure 6-digit labels
    return torch.tensor([char2idx[char] for char in text], dtype=torch.long)

def decode_text(indices):
    """CTC Decoding: Extracts exactly 6-digit numbers while ignoring blanks (index 10)."""
    filtered = [idx2char[idx] for idx in indices if idx != CTC_BLANK]  # Remove blanks (10)
    
    # Ensure exactly 6 digits (merge first 6 non-blank indices)
    result = "".join(filtered[:6])  

    # If less than 6 digits, add leading zeros
    return result.zfill(6)

# Focal CTCLoss to improve stability
class FocalCTCLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalCTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)  # Ensure no division by zero
        self.gamma = gamma

    def forward(self, logits, targets, input_lengths, target_lengths):
        loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        focal_loss = loss * ((1 - torch.exp(-loss)) ** self.gamma)
        return focal_loss.mean()



# Dataset
class CaptchaDataset(Dataset):
    def __init__(self, df, img_width=200, img_height=50):
        self.df = df
        self.img_width = img_width
        self.img_height = img_height

        self.augment = transforms.Compose([
            transforms.RandomRotation(degrees=5),  # Small rotations
            transforms.GaussianBlur(kernel_size=3),  # Blur to simulate noise
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift image slightly
        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        image = cv2.resize(image, (self.img_width, self.img_height)).astype('float32') / 255.0
        image = torch.tensor(image).unsqueeze(0)  # Convert to tensor

        #Apply augmentation randomly during training
        if torch.rand(1).item() > 0.5:
            image = self.augment(image)

        label_encoded = torch.tensor(encode_text(str(row['solution'])))  
        label_length = torch.tensor(len(label_encoded), dtype=torch.long)

        return image, label_encoded, label_length

    

def create_dataloader(df, batch_size=32, shuffle=True):
    dataset = CaptchaDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, dim=0)

    max_len = max(lengths)
    padded_labels = torch.full((len(batch), max_len), CTC_BLANK, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :lengths[i]] = label

    return images, padded_labels, torch.tensor(lengths)

# Basic CRNN Model Architecture
class BasicCRNN(nn.Module):
    def __init__(self, imgH, num_classes, hidden_size=128):
        super(BasicCRNN, self).__init__()
        input_size=512
        num_lstm_layers = 2
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),
            nn.Dropout(0.3),  
        )

        reduced_height = imgH // 8
        self.feature_projection = nn.Linear(256 * reduced_height, input_size)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            bidirectional=True, batch_first=True, dropout=0.3)  

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images):
        x = self.cnn(images)  
        B, C, H, W = x.shape  

        x = x.permute(0, 3, 2, 1).reshape(B, W, C * H)  
        x = self.feature_projection(x)

        x, _ = self.lstm(x)  
        x = self.fc(x)  

        return F.log_softmax(x.permute(1, 0, 2), dim=2)  

def extract_ground_truth(sample_label, sample_lengths):
    """Extracts ground truth text from label indices while ignoring CTC blanks."""
    ground_truth = []
    for i in range(sample_lengths[0].item()):  
        index_value = sample_label[0][i].item()
        if index_value != CTC_BLANK and index_value in idx2char:  #Ignore CTC blank
            ground_truth.append(idx2char[index_value])  
    return "".join(ground_truth)

def save_validation_image(image_tensor, ground_truth, predicted_text, epoch):
    """
    Saves the validation image with ground truth and predicted text.
    """
    # Convert tensor to NumPy image (ensure it’s a valid array)
    image = image_tensor.squeeze(0) * 255.0  # Convert back to grayscale
    image = image.astype(np.uint8)

    # Create figure
    plt.figure(figsize=(4, 2))
    plt.imshow(image, cmap='gray')
    plt.title(f"GT: {ground_truth} | Pred: {predicted_text}", fontsize=10)
    plt.axis('off')

    # Save Image
    filename = f"validation_predictions/epoch_{epoch}_gt_{ground_truth}_pred_{predicted_text}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved Validation Image: {filename}")

# Training and Evaluation
def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, lengths in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if torch.isnan(outputs).any():
                print("NaN detected in outputs, skipping this batch")
                continue
            
            # Ensure input_lengths match width of CNN output
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)

            # Ensure target lengths are valid
            lengths = torch.clamp(lengths, min=1, max=outputs.size(0))


            loss = loss_fn(outputs, labels, input_lengths, lengths.to(device))
            
            if torch.isnan(loss):
                print("NaN detected in loss, skipping this batch")
                continue
            
            total_loss += loss.item()

            _, max_indices = torch.max(outputs, dim=2)
            for i in range(images.shape[0]):
                pred_text = decode_text(max_indices[:, i].tolist())
                true_text = "".join([idx2char[idx] for idx in labels[i].tolist() if idx in idx2char])
                if pred_text == true_text:
                    correct += 1
                total += 1

    return total_loss / max(1, len(data_loader)), correct / max(1, total)

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=200, save_dir="crnn_model"):
    model.to(device)
    os.makedirs("validation_predictions", exist_ok=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=100)

    # Lists to store training and validation losses/accuracy
    log_data = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    metrics_file = os.path.join(save_dir, "training_metrics.csv")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f" Epoch {epoch+1}: Current Learning Rate: {current_lr:.6f}")

        for images, labels, lengths in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  

            # Ensure input_lengths match width of CNN output
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)

            # Ensure target lengths are valid
            lengths = torch.clamp(lengths, min=1, max=outputs.size(0))


            loss = loss_fn(outputs, labels, input_lengths, lengths.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device)

        # Log metrics
        log_data["epoch"].append(epoch + 1)
        log_data["train_loss"].append(total_loss / max(1, len(train_loader)))
        log_data["val_loss"].append(val_loss)
        log_data["val_accuracy"].append(val_acc)

        # Save logs to CSV after each epoch
        df_log = pd.DataFrame(log_data)
        df_log.to_csv(metrics_file, index=False)


        # scheduler.step(val_loss)

        # updated_lr = optimizer.param_groups[0]['lr']
        # print(f" Updated Learning Rate After Scheduler Step: {updated_lr:.6f}")

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss / max(1, len(train_loader)):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        if (epoch + 1) % 2 == 0:
            model.eval()
            sample_image, sample_label, sample_lengths = next(iter(val_loader))
            sample_image = sample_image.to(device)
            #print(f"Raw Label Tensor: {sample_label}")  # Print tensor before decoding
            ground_truth = extract_ground_truth(sample_label, sample_lengths)

            with torch.no_grad():
                output = model(sample_image)  
                _, max_indices = torch.max(output, dim=2)
                predicted_text = decode_text(max_indices[:, 0].tolist())

            print(f"**Epoch {epoch+1} Validation Sample**")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Predicted   : {predicted_text}")
            print(f"   Raw Indices : {max_indices[:, 0].tolist()}")
            print("-" * 50)


            save_validation_image(sample_image[0].cpu().numpy(), ground_truth, predicted_text, epoch+1)
        
        save_path = os.path.join(save_dir, f"crnn_model_{epoch}.pth")
   
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")


def load_model(path="crnn_model.pth", device=None):
    model = BasicCRNN(imgH=50, num_classes=num_classes).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded successfully!")
    return model

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['solution'] = df['solution'].astype(str).apply(lambda x: x.zfill(6))  
    test_df = df[df['image_path'].str.contains("test-images")].copy()
    val_df = df[df['image_path'].str.contains("validation-images")].copy()
    train_df = df[~df['image_path'].str.contains("test-images|validation-images")].copy()

    train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(TRAIN_IMG_BASE_PATH, x))
    val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(VAL_IMG_BASE_PATH, x))
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(TEST_IMG_BASE_PATH, x))

    return train_df,val_df,test_df

def load_data_new(csv_path):
    df = pd.read_csv(csv_path)
    df['solution'] = df['solution'].astype(str).apply(lambda x: x.zfill(6))  
    test_df = df[df['image_path'].str.contains("test-images")].copy()
    val_df = df[df['image_path'].str.contains("validation-images")].copy()
    train_df = df[~df['image_path'].str.contains("test-images|validation-images")].copy()

    # train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(TRAIN_IMG_BASE_PATH, x))
    val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(VAL_IMG_BASE_PATH, x))
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(TEST_IMG_BASE_PATH, x))

    return train_df,val_df,test_df

def predict_captcha(model, image_path, device):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return ""

    image = cv2.resize(image, (200, 50)).astype('float32') / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, max_indices = torch.max(output, dim=2)

        raw_indices = max_indices.squeeze().tolist()
        print(f"Raw Output Indices Before Decoding: {raw_indices}")  # Debug Print

        pred_text = decode_text(raw_indices)  #Extract 6 digits with leading zeros if needed

    print(f"Final Decoded Text: {pred_text}")
    return pred_text


# Run Training or Load Model
TRAIN_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/train-images/"
VAL_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/validation-images/"
TEST_IMG_BASE_PATH = "/home/abhinav/TASK/dataset/test-images/"

if __name__ == "__main__":
    csv_path = "/home/abhinav/TASK/dataset/captcha_data.csv"
    new_csv_path = "/home/abhinav/TASK/dataset/captcha_data_augmented.csv"
    # train_df, val_df, test_df = load_data(csv_path)
    train_df, val_df, test_df = load_data_new(new_csv_path)
    train_loader = create_dataloader(train_df)
    val_loader = create_dataloader(val_df, shuffle=False)

    device = get_device()
    model_path = "/home/abhinav/TASK/crnn_model.pth"
    save_dir = "/home/abhinav/TASK/models_pth"
    if os.path.exists(model_path):
        model = load_model(model_path, device)
    else:
        model = BasicCRNN(imgH=50, num_classes=num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-3, amsgrad=True)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        loss_fn = FocalCTCLoss()
        train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=200,save_dir=save_dir)

    # # Predict a Sample
    # test_image = "/home/abhinav/Downloads/dataset/train-images/train-images/image_train_9.png"
    # predicted_text = predict_captcha(model, test_image, device)
    # print(f"Predicted Captcha: {predicted_text}")
