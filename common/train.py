import copy
import time

import numpy as np
import torch
from torch import nn, optim

from mnist1d.utils import ObjectView

def get_model_args(as_dict=False):
    
    """
    Get Model arguments from command Line or Config file. 
    If as_dict is True, return as a dictionary.
    """

    arg_dict = {
        'input_size': 40,
        'output_size': 10,
        'hidden_size': 256,
        'learning_rate' : 1e-2,
        'weight_decay' : 0,
        'batch_size' : 100,
        'total_steps' : 8000,
        'print_every' : 1000,
        'eval_every' : 250,
        'checkpoint_every' : 1000,
        'device' : 'cpu',
        'seed' : 42
    }

    return arg_dict if as_dict else ObjectView(arg_dict)


def accuracy(model, inputs, targets):
    preds = model(inputs).argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy().astype(np.float32)
    return 100 * sum(preds == targets) / len(targets)


def train_model(dataset, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    x_train, x_test = torch.Tensor(dataset['x']), torch.Tensor(dataset['x_test'])
    y_train, y_test = torch.LongTensor(dataset['y']), torch.LongTensor(dataset['y_test'])
    model = model.to(args.device)
    x_train, x_test, y_train, y_test = [v.to(args.device) for v in [x_train, x_test, y_train, y_test]]

    results = {'checkpoints': [], 'train_losses': [], 'test_losses': [], 'train_acc': [], 'test_acc': []}
    t0 = time.time()
    for step in range(args.total_steps + 1):
        bix = (step * args.batch_size) % len(x_train)
        x, y = x_train[bix:bix + args.batch_size], y_train[bix:bix + args.batch_size]

        loss = criterion(model(x), y)
        results['train_losses'].append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.eval_every > 0 and step % args.eval_every == 0:
            test_loss = criterion(model(x_test), y_test)
            results['test_losses'].append(test_loss.item())
            results['train_acc'].append(accuracy(model, x_train, y_train))
            results['test_acc'].append(accuracy(model, x_test, y_test))

        if step > 0 and step % args.print_every == 0:
            t1 = time.time()
            print("step {}, dt {:.2f}s, train_loss {:.3e}, test_loss {:.3e}, train_acc {:.1f}, test_acc {:.1f}"
                  .format(step, t1 - t0, loss.item(), results['test_losses'][-1],
                          results['train_acc'][-1], results['test_acc'][-1]))
            t0 = t1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            model.step = step
            results['checkpoints'].append(copy.deepcopy(model))
    return results


if __name__ == "__main__":
    from mnist1d.data import make_dataset

    from common.models import LinearBase

    d = make_dataset()
    args = get_model_args()
    args.total_steps = 200
    args.eval_every = 50
    args.print_every = 100
    args.checkpoint_every = 0

    r = train_model(d, LinearBase(args.input_size, args.output_size), args)
    assert len(r['train_losses']) == args.total_steps + 1
    assert r['test_acc'][-1] > 0
    print("train_model self-check passed: final test_acc", r['test_acc'][-1])