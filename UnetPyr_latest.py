import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from torchmetrics import Dice
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
import platform
import psutil
from scipy.spatial.distance import directed_hausdorff

def checking_cuda_and_cpu():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(torch.__version__)

    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"CUDA version: {cuda_version}")
        print(f"Number of GPUs: {gpu_count}")
        print(f"Current GPU: {device_name}")

    # CPU info
    cpu_info = platform.processor()
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)

    print(f"\nCPU: {cpu_info}")
    print(f"Physical cores: {cpu_cores}")
    print(f"Logical cores: {cpu_threads}")

if __name__ == "__main__":
    checking_cuda_and_cpu()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetBackbone(nn.Module):
    def __init__(self, in_channels):
        super(UNetBackbone, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 64)
        self.conv4 = DoubleConv(64, 64)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        c4 = self.conv4(p3)
        return [c1, c2, c3, c4]

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        p5 = self.conv4(c4)
        p4 = self.conv3(c3) + F.interpolate(p5, scale_factor=2)
        p3 = self.conv2(c2) + F.interpolate(p4, scale_factor=2)
        p2 = self.conv1(c1) + F.interpolate(p3, scale_factor=2)
        return [p2, p3, p4, p5]

class SmoothConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmoothConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.unet_backbone = UNetBackbone(64)
        self.fpn = FeaturePyramidNetwork(64, 256)
        self.smooth_conv = SmoothConvolution(256, 256)
        self.output_layer = nn.Sequential(
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.input_layer(x)
        features = self.unet_backbone(x)
        fpn_features = self.fpn(features)
        smoothed = self.smooth_conv(fpn_features[0])
        output = self.output_layer(smoothed)
        return output

# Testing UNetPyr Model
model = SegmentationModel(num_classes=1)
input_tensor = torch.randn(1, 3, 256, 256)  # RGB image
output = model(input_tensor)
print(output.shape)
print(output.dtype)

class UltrasoundDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if not f.endswith('_mask.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, img_name.replace('.jpg', '_mask.jpg'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping_patience=50):
    train_losses = []
    train_accuracies = []
    train_dice_coefficients = []
    train_jaccard_scores = []
    val_losses = []
    val_accuracies = []
    val_dice_coefficients = []
    val_jaccard_scores = []
    
    # best validation loss and early stopping counter
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_dice = 0.0
        running_jaccard = 0.0
        batch_count = 0

        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            # Removing extra dimensions from masks
            masks = masks.squeeze(1).squeeze(1)
            loss = criterion(outputs, masks.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            # Metrics
            predicted = torch.sigmoid(outputs)
            predicted = predicted.squeeze(1)

            # Converting to binary predictions and masks
            binary_preds = (predicted > 0.5).cpu().numpy().astype(int).reshape(-1)
            binary_masks = masks.cpu().numpy().astype(int).reshape(-1)

            # Batch metrics
            batch_accuracy = accuracy_score(binary_masks, binary_preds)
            batch_jaccard = jaccard_score(binary_masks, binary_preds, average='weighted', zero_division=1)
            batch_dice = Dice()(torch.tensor(binary_preds, dtype=torch.int64),
                              torch.tensor(binary_masks, dtype=torch.int64)).item()

            running_loss += loss.item()
            running_acc += batch_accuracy
            running_dice += batch_dice
            running_jaccard += batch_jaccard
            batch_count += 1

        # Training metrics averages
        epoch_train_loss = running_loss / batch_count
        epoch_train_acc = running_acc / batch_count
        epoch_train_dice = running_dice / batch_count
        epoch_train_jaccard = running_jaccard / batch_count

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                masks = masks.squeeze(1).squeeze(1)
                val_loss = criterion(outputs, masks.unsqueeze(1).float())
                val_running_loss += val_loss.item()
                val_batch_count += 1

        # Validation metrics
        val_metrics = compute_metrics(model, val_loader)
        epoch_val_loss = val_running_loss / val_batch_count
        epoch_val_acc = val_metrics[0]
        epoch_val_dice = val_metrics[5]
        epoch_val_jaccard = val_metrics[4]

        # Metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        train_dice_coefficients.append(epoch_train_dice)
        train_jaccard_scores.append(epoch_train_jaccard)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        val_dice_coefficients.append(epoch_val_dice)
        val_jaccard_scores.append(epoch_val_jaccard)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Train Dice: {epoch_train_dice:.4f}, Train IoU: {epoch_train_jaccard:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(f'Val Dice: {epoch_val_dice:.4f}, Val IoU: {epoch_val_jaccard:.4f}')

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                model.load_state_dict(best_model_state)
                break

    return (train_losses, train_accuracies, train_dice_coefficients, train_jaccard_scores,
            val_losses, val_accuracies, val_dice_coefficients, val_jaccard_scores)

def compute_metrics(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    hausdorff_distances = []

    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images)
            predicted = torch.sigmoid(outputs)
            predicted = predicted.squeeze(1)
            masks = masks.squeeze(1).squeeze(1)

            # Binary predictions and masks
            binary_preds = (predicted > 0.5).cpu().numpy().astype(int)
            binary_masks = masks.cpu().numpy().astype(int)

            for binary_pred, binary_mask in zip(binary_preds, binary_masks):
                pred_coords = np.argwhere(binary_pred == 1)
                mask_coords = np.argwhere(binary_mask == 1)
                
                if len(pred_coords) > 0 and len(mask_coords) > 0:
                    try:
                        forward_hdist = directed_hausdorff(pred_coords, mask_coords)[0]
                        backward_hdist = directed_hausdorff(mask_coords, pred_coords)[0]
                        hausdorff_distances.append((forward_hdist, backward_hdist))
                    except Exception as e:
                        print(f"Error calculating Hausdorff distance: {e}")

            # Flatten binary predictions and masks
            binary_preds = binary_preds.reshape(-1)
            binary_masks = binary_masks.reshape(-1)
            
            all_preds.extend(binary_preds)
            all_labels.extend(binary_masks)

    # Converting lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    jaccard = jaccard_score(all_labels, all_preds, average='weighted', zero_division=1)
    dice_coeff = Dice()(torch.tensor(all_preds, dtype=torch.int64), torch.tensor(all_labels, dtype=torch.int64))

    # Calculate Hausdorff metrics
    if hausdorff_distances:
        forward_distances, backward_distances = zip(*hausdorff_distances)
        percentile_hdist = np.percentile(forward_distances + backward_distances, 95)
    else:
        percentile_hdist = float('inf')

    return accuracy, precision, recall, f1, jaccard, dice_coeff.item(), percentile_hdist

def visualize_predictions(model, dataloader, num_images=20, save_path = 'UnetPyr_latest_Normal.png'):
    model.eval()
    lesion_cases = []

    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            predicted = torch.sigmoid(outputs).cpu().numpy()

            for i in range(len(images)):
                lesion_cases.append((images[i], masks[i], outputs[i], predicted[i]))

                if len(lesion_cases) == num_images:
                    break
            if len(lesion_cases) == num_images:
                break

    fig, axs = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    # Plot
    for i, (image, mask, output, pred) in enumerate(lesion_cases):
        # Convert tensors to numpy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = pred

        # Threshold predicted probabilities to get binary predictions
        # Converting to binary
        binary_pred_np = (pred_np > 0.5).astype(float)

        # Normalizing image
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Ensure image is in the right format
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = image_np.squeeze(axis=2)

        # If mask is (1, H, W), squeeze to (H, W)
        mask_np = mask_np.squeeze()
        binary_pred_np = binary_pred_np.squeeze()

        # Plot
        axs[i, 0].imshow(image_np, cmap='gray')
        axs[i, 0].set_title(f'Lesion Image {i + 1}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask_np, cmap='gray')
        axs[i, 1].set_title('Ground Truth')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(binary_pred_np, cmap='gray')
        axs[i, 2].set_title('Prediction')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

#from google.colab import drive
#drive.mount('/content/drive')

file_dir = '/home/rizk_lab/shared/Sony/BUSI'
# Main execution
if __name__ == "__main__":
    # Data loading and preprocessing
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    dataset = UltrasoundDataset(data_dir=file_dir, transform=transform)
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    # Model initialization
    model = SegmentationModel(num_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    (train_losses, train_accuracies, train_dice_coefficients, train_jaccard_scores,
        val_losses, val_accuracies, val_dice_coefficients, val_jaccard_scores
    ) = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200)

    # Plot metrics
    (train_losses, train_accuracies, train_dice_coefficients, train_jaccard_scores,
        val_losses, val_accuracies, val_dice_coefficients, val_jaccard_scores)

# Compute test metrics
accuracy, precision, recall, f1, jaccard, dice_coeff, hausdorff = compute_metrics(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Jaccard Score: {jaccard:.6f}")
print(f"Dice Coefficient: {dice_coeff:.6f}")
print(f"hausdorff distance: {hausdorff:.6f}")

# Visualize segmentation
visualize_predictions(model, test_loader, num_images=20, save_path='UnetPyr_latest_Normal.png')