from train import train 
from dataset import * 
from utils import *
from models import *
from evaluation import *
from loss import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--loss', type=str, default='entropy', choices=['focal', 'dice', 'bce', 'entropy'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_raw = PointCloudDataset(args.train_dir)
    test_raw = PointCloudDataset(args.test_dir)
    val_raw = PointCloudDataset(args.val_dir)

    train_dataset = SelectedPointDataset(train_raw)
    test_dataset = SelectedPointDataset(test_raw)
    val_dataset = SelectedPointDataset(val_raw)
    sampler = get_weighted_sampler(train_dataset, num_classes=10)
    # Dataloaders
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=pad_collate_fn)

    # Model, optimizer, scheduler
    model = GeoPointNet(input_dim=6, num_classes=10).to(device)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = optim.Lookahead(base_opt)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    weights = compute_class_weights(train_loader, num_classes=10).to(device)
    start_epoch = 0
    checkpoint_path = 'modelFocal.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for input_data, labels in train_loader:
            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            logits = logits.reshape(-1, 10)
            labels = labels.reshape(-1)

            #loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=0)
            loss_fn = FocalLoss(alpha=weights.to(device), gamma=2.0)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
           
        avg_loss = epoch_loss / count if count > 0 else float('nan')
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

    # Evaluation
    print("Evaluating on validation set...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_data, labels in train_loader:
            input_data, labels = input_data.to(device), labels.to(device)
            input_data = normalize_input(input_data)

            logits = model(input_data)
            preds = torch.argmax(logits, dim=2)

            predictions.append(preds.cpu())
            true_labels.append(labels.cpu())

            # Optional: insert debugger
            #import ipdb; ipdb.set_trace()

    # Evaluate predictions
    evaluate_model(predictions, true_labels, num_classes=10, ignore_index=0)
if __name__ == "__main__":
    main()