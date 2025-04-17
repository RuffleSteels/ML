import glob
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def generate_dataset(size, offset = 0):
    labels_df = pd.read_csv("data/train_solution_bounding_boxes (1).csv")
    image_dir = "data/training_images"
    jpg_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[offset:size+offset]
    image_names = [os.path.basename(f) for f in jpg_files]

    data = labels_df[["image", "xmin", "ymin", "xmax", "ymax"]]

    nested_list = [
        [row["image"], [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]]
        for _, row in data.iterrows()
    ]

    data_dict = {item[0]: item[1] for item in nested_list}
    Y_values = []

    for item in image_names:
        if item in data_dict:
            Y_values.append(data_dict[item])
        else:
            Y_values.append([0.0,0.0,0.0,0.0])

    Y_value = torch.tensor(Y_values, dtype=torch.float)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensors = []

    for path in jpg_files:
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        image_tensors.append(tensor)

    X_value = torch.stack(image_tensors)


    X_value = X_value.to(device)
    Y_value = Y_value.to(device)
    return X_value, Y_value

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1, stride=1),  # [n, 5, 380, 676]
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),               # [n, 5, 190, 338]

            nn.Conv2d(5, 5, kernel_size=3, padding=1, stride=1),  # [n, 5, 190, 338]
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),               # [n, 5, 95, 169]

            nn.Conv2d(5, 5, kernel_size=3, padding=1, stride=1),  # [n, 5, 95, 169]
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(5*95*169, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def train(model, X, Y):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    convergence_threshold = 1e-4
    patience = 10

    last_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(2000):
        prediction = model(X)

        loss = loss_fn(prediction, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}")

        if abs(last_loss - loss.item()) < convergence_threshold:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        last_loss = loss.item()

        if epochs_since_improvement >= patience:
            print("Convergence reached. Stopping early.")
            break

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_params_epoch_{epoch+1}.pth')
            print(f"Model parameters saved at epoch {epoch+1}.")

    torch.save(model.state_dict(), 'model_params.pth')


# X, Y = generate_dataset(800)
# model = Model().to(device)
# train(model, X, Y)

def show_image(x, y, y_truee):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # Convert image tensor to NumPy

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()  # Convert coordinates to NumPy

    if isinstance(y_truee, torch.Tensor):
        y_truee = y_truee.detach().cpu().numpy()  # Convert coordinates to NumPy

    # Ensure the image is in the shape (height, width, channels)
    if x.ndim == 3 and x.shape[0] == 3:  # Channels first (C, H, W)
        x = x.transpose(1, 2, 0)  # Convert to (H, W, C) for matplotlib

    # Create the plot
    fig, ax = plt.subplots()
    ax.imshow(x)

    h, w, _ = x.shape

    x1, y1, x2, y2 = y
    y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)

    rect = patches.Rectangle((x2, y2), x1 - x2, y1 - y2, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    x1, y1, x2, y2 = y_truee
    y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)


    rect2 = patches.Rectangle((x2, y2), x1 - x2, y1 - y2, linewidth=2, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect2)


    plt.axis('off')
    plt.show()

X, Y = generate_dataset(30, 300)

model = Model().to(device)
model.load_state_dict(torch.load('model_params_epoch_4550.pth', weights_only=True))

model.eval()

with torch.no_grad():
    X_new = X.to(device)
    y_true = Y

    output = model(X_new)
    for i in range(X_new.shape[0]):
        show_image(X[i], output[i], y_true[i])
