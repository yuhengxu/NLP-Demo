import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import MotionDataset
from utils import load_data
from models import MLP, RNN, myRNN, CNN
from sklearn.metrics import classification_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this model')
    parser.add_argument('--model', type=str, default='myRNN')
    parser.add_argument('--choose_vocab', type=str, default='glove')
    parser.add_argument('--pretrain', type=bool, default=False)
    args = parser.parse_args()
    cfg = Config(args)

    train_set, test_set, vocab, glove = load_data(cfg)
    train_loader = torch.utils.data.DataLoader(
        MotionDataset(train_set),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        MotionDataset(test_set),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    if cfg.model == 'MLP':
        model = MLP(cfg, vocab, glove).train().to(cfg.device)
    elif cfg.model == 'RNN':
        model = RNN(cfg, vocab, glove).train().to(cfg.device)
    elif cfg.model == 'myRNN':
        model = myRNN(cfg, vocab, glove).train().to(cfg.device)
    elif cfg.model == 'CNN':
        model = CNN(cfg, vocab, glove).train().to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learn_rate)

    writer = SummaryWriter(f'runs/{cfg.model}')
    train_steps, test_steps = 0, 0
    for epoch in range(1, cfg.epoch + 1):
        # Train
        hits, total = 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} Train')
        for data, label in pbar:
            data, label = data.to(cfg.device), label.to(cfg.device)
            logits = model(data)
            loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=label.double(),
                reduction='mean',
            )
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_hits = ((logits > 0).int() == label).sum().item()
            batch_total = len(label)
            hits += batch_hits
            total += batch_total
            pbar.set_postfix({
                'acc': f'{batch_hits / batch_total:.4f}',
                'loss': f'{loss.item():.4f}'
            })
            print(classification_report((logits > 0).int().cpu().detach().numpy(), label.cpu().detach().numpy()))
            train_steps += 1
            writer.add_scalar('Loss/Train-Loss', loss.item(), train_steps)

        train_acc = hits / total
        writer.add_scalar('Acc/Train-Acc', train_acc, epoch)
        print(f'Train Acc: {train_acc:.4f}')

        # Test
        hits, total = 0, 0
        pbar = tqdm(test_loader, desc=f'Epoch {epoch} Test')
        for data, label in pbar:
            data, label = data.to(cfg.device), label.to(cfg.device)
            with torch.no_grad():
                logits = model(data)
                test_loss = F.binary_cross_entropy_with_logits(
                    input=logits,
                    target=label.float(),
                    reduction='mean',
                )
                batch_hits = ((logits > 0).int() == label).sum().item()
                batch_total = len(label)
                hits += batch_hits
                total += batch_total
            pbar.set_postfix({
                'acc': f'{batch_hits / batch_total:.4f}',
                'loss': f'{test_loss.item():.4f}'
            })
            print(classification_report((logits > 0).int().cpu().detach().numpy(), label.cpu().detach().numpy()))
            test_steps += 1
            writer.add_scalar('Loss/Test-Loss', test_loss.item(), test_steps)

        test_acc = hits / total
        writer.add_scalar('Acc/Test-Acc', test_acc, epoch)
        print(f'Test Acc: {test_acc:.4f}')
