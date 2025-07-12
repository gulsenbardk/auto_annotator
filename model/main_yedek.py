from train import train 
from dataset import * 
from utils import *
from models import *
from evaluation import *
from loss import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss', type=str, default='entropy', choices=['focal', 'dice', 'bce', "entropy"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = PointCloudDataset(args.train_dir)
    test_raw = PointCloudDataset(args.test_dir)
    val_raw = PointCloudDataset(args.val_dir)

    train_dataset = SelectedPointDataset(train_raw)
    test_dataset = SelectedPointDataset(test_raw)
    val_dataset = SelectedPointDataset(val_raw)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn,  drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=pad_collate_fn)
    start_epoch = 0
    

    model = GeoPointNet(input_dim=6, num_classes=10)
    model.to(device)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  
    optimizer = optim.Lookahead(base_opt)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if os.path.exists("model.pth"):
        checkpoint = torch.load("model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    train_losses = []
    accs = []
    precisions = []
    recalls = []
    f1s = []
  

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for input_data, labels in train_loader:
            input_data = input_data.to(device)
            labels = labels.to(device)

            input_data = (input_data - input_data.mean(dim=1, keepdim=True)) / (input_data.std(dim=1, keepdim=True) + 1e-5)
            logits = model(input_data)

            logits = logits.reshape(-1, 10)
            labels = labels.reshape(-1)

            loss = F.cross_entropy(logits, labels, ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        avg_loss = epoch_loss / count if count > 0 else float('nan')
        print(f"Epoch {epoch + 1}, avg loss: {avg_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, 'model100epoch8batch.pth')

    model.eval()
    model.to(device)

    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_data, labels in val_loader:
            input_data = input_data.to(device) 
            labels = labels.to(device)

            input_data = (input_data - input_data.mean(dim=1, keepdim=True)) / (input_data.std(dim=1, keepdim=True) + 1e-5)

            logits = model(input_data)              
            preds = torch.argmax(logits, dim=2)
            
            evaluate_model(preds, labels)     
            predictions.append(preds)
            true_labels.append(labels)
    
            import ipdb
            ipdb.set_trace()