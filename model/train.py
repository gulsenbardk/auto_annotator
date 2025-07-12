from header import *
from loss import *


def train(args, model, dataloader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0.0

    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    elif args.loss == 'dice':
        criterion = MultiClassDiceLoss()
    elif args.loss == 'bce':  # NOT recommended for multi-class
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif args.loss == "entropy":
        print("entropy")
        criterion = nn.CrossEntropyLoss(ignore_index=0) 
    else: 
        print("loss is not avaliable")

    for points, labels in tqdm(dataloader):
        points = points.to(device)      # (B, N, 3)
        #points = normalize_points(points) 
        labels = labels.to(device).long()  # (B, N)
        #import ipdb 
        #ipdb.set_trace()
        optimizer.zero_grad()
        logits = model(points)          # (B, N, C)
        num_classes = logits.shape[-1]
        #import ipdb 
        #ipdb.set_trace()
        if args.loss == 'dice':
            loss = criterion(logits, labels, num_classes)
        elif args.loss == 'bce':
            labels_onehot = F.one_hot(labels, num_classes).float()
            loss = criterion(logits.view(-1, num_classes), labels_onehot.view(-1, num_classes))
        elif args.loss == "focal":
            loss = criterion(logits.view(-1, num_classes), labels.view(-1), num_classes=num_classes)
        elif args.loss == "entropy":
            loss = criterion(logits.view(-1, num_classes), labels.view(-1))
        else: 
            print("Loss cannot avaliable")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)