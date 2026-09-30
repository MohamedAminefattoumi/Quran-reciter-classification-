




# 1. Importing libraries
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



# 2. Device configuration
# Use the GPU if CUDA is available.
# Otherwise, use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Available device: {device}")



# 3. Project paths
# Get the directory containing this Python script.
proj_dir = Path(__file__).resolve().parent

# Main data directory.
data_dir = proj_dir / "data"

# Directory containing the audio files.
recitations_path = data_dir / "Dataset"

# CSV file containing the audio file paths and their classes.
csv_path = data_dir / "files_paths.csv"



# 4. Loading the dataset
# Load the CSV file into a pandas DataFrame.
data_df = pd.read_csv(csv_path)

# The CSV contains relative paths such as:
#
# ./Dataset/Class/file.wav
#
# We replace "./" with the absolute path to the Dataset folder.
data_df["FilePath"] = (
    str(recitations_path) + data_df["FilePath"].str[1:]
)

print(f"Dataset shape: {data_df.shape}")
print("\nFirst five samples:")
print(data_df.head())



# 5. Encoding the class labels
label_encoder = LabelEncoder()

data_df["Class"] = label_encoder.fit_transform(
    data_df["Class"]
)

# Number of different classes in the dataset.
num_classes = len(label_encoder.classes_)

print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {list(label_encoder.classes_)}")



# 6. Splitting the dataset
# First split:
#
# 70% -> training
# 30% -> temporary dataset
#
# stratify ensures that the class distribution remains
# approximately the same in each subset.

train, temp = train_test_split(
    data_df,
    test_size=0.30,
    random_state=7,
    stratify=data_df["Class"]
)

# Second split:
#
# 15% -> validation
# 15% -> test
#
# The temporary 30% is divided equally.

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=7,
    stratify=temp["Class"]
)

print("\nDataset splits:")
print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")
print(f"Test shape: {test.shape}")



# 7. Custom Audio Dataset
class CustomAudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading audio files.

    Each audio file is:
        audio file
            ↓
        Mel-spectrogram
            ↓
        Convert to decibels
            ↓
        Resize to 128 x 256
            ↓
        Convert to PyTorch tensor

    The final tensor returned by the Dataset has the shape:

        (1, 128, 256)

    The first dimension represents the single input channel.
    """

    def __init__(self, dataframe):
        """
        Initialize the dataset.

        Args:
            dataframe (pd.DataFrame):
                DataFrame containing file paths and class labels.
        """

        self.dataframe = dataframe.reset_index(drop=True)

        # Store labels as CPU tensors.
        # We move the batch to the GPU later in the training loop.
        self.labels = torch.tensor(
            self.dataframe["Class"].to_numpy(),
            dtype=torch.long
        )

        # Pre-compute all spectrograms.
        #
        # This makes training faster because librosa does not
        # need to process the audio file at every epoch.
        self.audios = [
            torch.tensor(
                self.get_spectrogram(path),
                dtype=torch.float32
            )
            for path in self.dataframe["FilePath"]
        ]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Return one sample and its corresponding label.

        Returns:
            audio: Tensor with shape (1, 128, 256)
            label: Integer class label
        """

        # Add a channel dimension.
        #
        # Original:
        #     (128, 256)
        #
        # After unsqueeze:
        #     (1, 128, 256)
        #
        # CNNs expect the channel dimension.
        audio = self.audios[idx].unsqueeze(0)

        label = self.labels[idx]

        return audio, label

    @staticmethod
    def get_spectrogram(file_path):
        """
        Convert an audio file into a resized Mel-spectrogram.

        Args:
            file_path (str or Path):
                Path to the audio file.

        Returns:
            np.ndarray:
                Mel-spectrogram with shape (128, 256).
        """

        # Target sampling rate.
        sample_rate = 22050

        # Load only the first 5 seconds of the audio.
        duration = 5

        # Target spectrogram dimensions.
        img_height = 128
        img_width = 256

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        signal, _ = librosa.load(
            file_path,
            sr=sample_rate,
            duration=duration
        )

        # ----------------------------------------------------
        # Generate Mel-spectrogram
        # ----------------------------------------------------

        spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # ----------------------------------------------------
        # Convert power spectrogram to decibel scale
        # ----------------------------------------------------

        spectrogram_db = librosa.power_to_db(
            spectrogram,
            ref=np.max
        )

        # ----------------------------------------------------
        # Make the time dimension consistent
        # ----------------------------------------------------

        expected_width = (duration * sample_rate) // 512 + 1

        spectrogram_db = librosa.util.fix_length(
            spectrogram_db,
            size=expected_width
        )

        # ----------------------------------------------------
        # Resize to exactly 128 x 256
        # ----------------------------------------------------

        spectrogram_resized = resize(
            spectrogram_db,
            (img_height, img_width),
            anti_aliasing=True
        )

        return spectrogram_resized


# ============================================================
# 8. Creating Dataset objects
# ============================================================

train_dataset = CustomAudioDataset(train)
val_dataset = CustomAudioDataset(val)
test_dataset = CustomAudioDataset(test)


# ============================================================
# 9. Creating DataLoaders
# ============================================================

# Training parameters.
BATCH_SIZE = 16

# Shuffle the training dataset so that the model does not
# always see the samples in the same order.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# No need to shuffle validation and test datasets.
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 10. CNN Model
# ============================================================

class Net(nn.Module):
    """
    Convolutional Neural Network for audio classification.

    Input:
        (batch_size, 1, 128, 256)

    Output:
        (batch_size, number_of_classes)
    """

    def __init__(self, num_classes):
        super().__init__()

        # ----------------------------------------------------
        # Convolutional layers
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Max pooling reduces the spatial dimensions by 2.
        self.pooling = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ReLU introduces non-linearity.
        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # Fully connected layers
        # ----------------------------------------------------

        self.flatten = nn.Flatten()

        # After three MaxPool2d operations:
        #
        # Input:
        #     128 x 256
        #
        # After pool 1:
        #     64 x 128
        #
        # After pool 2:
        #     32 x 64
        #
        # After pool 3:
        #     16 x 32
        #
        # Number of features:
        #     64 * 16 * 32 = 32768

        self.linear1 = nn.Linear(
            64 * 16 * 32,
            4096
        )

        self.linear2 = nn.Linear(
            4096,
            1024
        )

        self.linear3 = nn.Linear(
            1024,
            512
        )

        # Final layer produces one value per class.
        self.output = nn.Linear(
            512,
            num_classes
        )

        # Dropout helps reduce overfitting.
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        Define the forward pass of the network.
        """

        # -------------------------
        # Convolution block 1
        # -------------------------

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)

        # -------------------------
        # Convolution block 2
        # -------------------------

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        # -------------------------
        # Convolution block 3
        # -------------------------

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)

        # -------------------------
        # Fully connected layers
        # -------------------------

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final classification layer.
        #
        # CrossEntropyLoss expects raw logits,
        # so we do NOT apply Softmax here.
        x = self.output(x)

        return x



# 11. Creating the model
model = Net(num_classes=num_classes).to(device)
print("\nModel:")
print(model)



# 12. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Adam updates the model parameters during training.
LEARNING_RATE = 1e-4

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



# 13. Training configuration
EPOCHS = 25

# Lists used to store metrics for plotting later.
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# 14. Training loop

start_time = time.time()

for epoch in tqdm(range(EPOCHS), desc="Training"):

    # ========================================================
    # Training phase
    # ========================================================

    # Enables training behavior such as Dropout.
    model.train()

    total_train_loss = 0.0
    total_train_correct = 0

    for inputs, labels in train_loader:

        # Move the current batch to the selected device.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)

        # Calculate loss.
        train_loss = criterion(outputs, labels)

        # Backpropagation.
        train_loss.backward()

        # Update model parameters.
        optimizer.step()

        # Store batch loss.
        total_train_loss += train_loss.item()

        # Get the predicted class.
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Count correct predictions.
        total_train_correct += (
            predictions == labels
        ).sum().item()

    # Average training loss across batches.
    avg_train_loss = (
        total_train_loss / len(train_loader)
    )

    # Training accuracy.
    train_accuracy = (
        total_train_correct / len(train_dataset)
    ) * 100


    # ========================================================
    # Validation phase
    # ========================================================

    # Disables Dropout and other training-specific behavior.
    model.eval()

    total_val_loss = 0.0
    total_val_correct = 0

    # No gradients are needed during validation.
    with torch.no_grad():

        for inputs, labels in val_loader:

            # Move the batch to the selected device.
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(inputs)

            # Calculate validation loss.
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss.item()

            # Get predictions.
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Count correct predictions.
            total_val_correct += (
                predictions == labels
            ).sum().item()

    # Average validation loss.
    avg_val_loss = (
        total_val_loss / len(val_loader)
    )

    # Validation accuracy.
    val_accuracy = (
        total_val_correct / len(val_dataset)
    ) * 100


    # ========================================================
    # Store metrics
    # ========================================================

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # ========================================================
    # Display epoch results
    # ========================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f" | Train Loss: {avg_train_loss:.4f}"
        f" | Train Accuracy: {train_accuracy:.2f}%"
        f" | Val Loss: {avg_val_loss:.4f}"
        f" | Val Accuracy: {val_accuracy:.2f}%"
    )


# ============================================================
# 15. Training time
# ============================================================

training_time = time.time() - start_time

print(
    f"\nTraining time: {training_time:.2f} seconds"
)


# ============================================================
# 16. Test evaluation
# ============================================================

# Put the model into evaluation mode.
model.eval()

total_test_loss = 0.0
total_test_correct = 0

# Disable gradient calculation because we are not training.
with torch.no_grad():

    for inputs, labels in test_loader:

        # Move the batch to the selected device.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass.
        outputs = model(inputs)

        # Calculate test loss.
        test_loss = criterion(outputs, labels)

        total_test_loss += test_loss.item()

        # Get predictions.
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Count correct predictions.
        total_test_correct += (
            predictions == labels
        ).sum().item()


# Calculate final test metrics.
avg_test_loss = (
    total_test_loss / len(test_loader)
)

test_accuracy = (
    total_test_correct / len(test_dataset)
) * 100


print("\n==============================")
print("Final Test Results")
print("==============================")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")


# ============================================================
# 17. Plot training and validation loss
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 18. Plot training and validation accuracy
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 19. Optional: Save the trained model
# ============================================================

# Save the model parameters so that the trained model
# can be loaded later without training it again.

model_path = proj_dir / "audio_classifier.pth"

torch.save(
    model.state_dict(),
    model_path
)

print(f"\nModel saved to: {model_path}")

