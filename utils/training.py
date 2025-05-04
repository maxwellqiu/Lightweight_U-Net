import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm, trange
import numpy as np
import torch


def train_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, msks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(train_loader.dataset)


def eval_epoch(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    ios, dices, precs = [], [], []
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)
            loss = criterion(preds, msks)
            total_loss += loss.item() * imgs.size(0)
            bp = (preds > 0.5).float()
            tp = (bp * msks).sum((1, 2, 3))
            fp = (bp * (1 - msks)).sum((1, 2, 3))
            fn = ((1 - bp) * msks).sum((1, 2, 3))
            precs.extend((tp / (tp + fp + 1e-6)).cpu().numpy())
            dices.extend((2 * tp / (2 * tp + fp + fn + 1e-6)).cpu().numpy())
            ios.extend((tp / (tp + fp + fn + 1e-6)).cpu().numpy())
    return (total_loss / len(val_loader.dataset), np.mean(ios), np.mean(dices),
            np.mean(precs))


def train(model,
          device,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          scheduler,
          num_epochs=60,
          save_path='best.pth'):
    history = {
        k: []
        for k in
        ['train_loss', 'val_loss', 'val_precision', 'val_dice', 'val_miou']
    }
    best = -np.inf

    pbar = tqdm(range(num_epochs), desc="Epoch")
    for epoch in pbar:
        tr_loss = train_epoch(model, device, train_loader, criterion,
                              optimizer)
        history['train_loss'].append(tr_loss)
        val_loss, val_iou, val_dice, val_prec = eval_epoch(
            model, device, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_precision'].append(val_prec)

        if val_iou > best:
            best, best_dice, best_prec = val_iou, val_dice, val_prec
            torch.save(model.state_dict(), save_path)

        scheduler.step(val_iou)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}  "
            f"Train Loss: {tr_loss:.4f}  Validaton Loss: {val_loss:.4f}  "
            f"IoU: {val_iou:.4f}  Dice: {val_dice:.4f}  Prec: {val_prec:.4f}")
    return history, best, best_dice, best_prec


def pretrain_barlow_twins(bt_model,
                          dataloader,
                          optimizer,
                          criterion_bt,
                          epochs,
                          device,
                          encoder_save_path="bt_encoder_pretrained.pth"):
    bt_model.train()
    history = {'loss': []}

    for _ in trange(epochs):
        running_loss = 0.0
        processed_batches = 0

        for _, (view1, view2) in enumerate(dataloader):
            if view1 is None or view2 is None or view1.shape[0] < 2:
                continue

            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            z1 = bt_model(view1)
            z2 = bt_model(view2)

            loss = criterion_bt(z1, z2)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

        if processed_batches == 0:
            avg_loss = 0.0
        else:
            avg_loss = running_loss / processed_batches
            history['loss'].append(avg_loss)

    torch.save(bt_model.encoder.state_dict(), encoder_save_path)
    return history


def train_epoch_rl(model, device, train_loader, criterion, optimizer, img_size,
                   alpha, rl_weight):
    model.train()
    total_sup, total_pg = 0.0, 0.0
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        probs = model(imgs)
        sl = criterion(probs, msks)
        tp = (probs * msks).sum((1, 2, 3))
        fp = (probs * (1 - msks)).sum((1, 2, 3))
        fn = ((1 - probs) * msks).sum((1, 2, 3))
        soft_dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        non_overlap = (fp + fn) / (img_size[0] * img_size[1])
        reward = soft_dice - alpha * non_overlap
        reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        bern = torch.distributions.Bernoulli(probs)
        acts = bern.sample()
        logp = bern.log_prob(acts)
        pg_loss = -(reward.detach().view(-1, 1, 1, 1) * logp).mean()
        loss = sl + rl_weight * pg_loss
        loss.backward()
        optimizer.step()
        total_sup += sl.item() * imgs.size(0)
        total_pg += pg_loss.item() * imgs.size(0)
    return total_sup / len(train_loader.dataset), total_pg / len(
        train_loader.dataset)


def train_rl(model,
             device,
             train_loader,
             val_loader,
             criterion,
             optimizer,
             scheduler,
             img_size,
             alpha,
             rl_weight,
             preheat_epochs,
             num_epochs=60,
             save_path='best.pth'):
    history = {
        k: []
        for k in [
            'train_loss', 'pg_loss', 'val_loss', 'val_precision', 'val_dice',
            'val_miou'
        ]
    }
    best = -np.inf

    pbar = tqdm(range(num_epochs), desc="Epoch")
    for epoch in pbar:

        if epoch <= preheat_epochs:
            tr_loss = train_epoch(model, device, train_loader, criterion,
                                  optimizer)
            pg_loss = 0.0
        else:
            tr_loss, pg_loss = train_epoch_rl(model, device, train_loader,
                                              criterion, optimizer, img_size,
                                              alpha, rl_weight)

        history['train_loss'].append(tr_loss)
        history['pg_loss'].append(pg_loss)
        val_loss, val_iou, val_dice, val_prec = eval_epoch(
            model, device, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_precision'].append(val_prec)

        if val_iou > best:
            best, best_dice, best_prec = val_iou, val_dice, val_prec
            torch.save(model.state_dict(), save_path)

        scheduler.step(val_iou)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}  "
            f"Train Loss: {tr_loss:.4f}  pg Loss: {pg_loss:.4f} "
            f"Validaton Loss: {val_loss:.4f}  "
            f"IoU: {val_iou:.4f}  Dice: {val_dice:.4f}  Prec: {val_prec:.4f}")
    return history, best, best_dice, best_prec
