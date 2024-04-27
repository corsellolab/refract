import os
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def train_model(model, train_dataloader, val_dataloader, num_epochs, patience, chkpt_dir):
    # Assuming the loss function and optimizer are predefined globally or are parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize TensorBoard writer
    writer = SummaryWriter()
    checkpoint_files = os.path.join(chkpt_dir, "chkpt_epoch_{}.pth")

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0
        for inputs, cell_line, targets in tqdm.tqdm(train_dataloader):
            inputs, cell_line, targets = inputs.to(device), cell_line.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs, cell_line).squeeze().float()
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataloader.dataset)

        # Log training loss
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for inputs, cell_line, targets in val_dataloader:
                inputs, cell_line, targets = inputs.to(device), cell_line.to(device), targets.to(device)
                outputs = model(inputs, cell_line).squeeze().float()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataloader.dataset)

        # Log validation loss
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        save_checkpoint(model, optimizer, epoch, loss, checkpoint_files)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Close the writer
    writer.close()

    # Load the best model back
    model.load_state_dict(torch.load('best_model.pth'))