import torch

def consistency_loss(pred, target, eps=1e-10):
    pred = pred + eps
    target = target + eps
    kl_loss = target * torch.log(target / pred)
    kl_loss = kl_loss.sum(dim=1)
    kl_loss = kl_loss.mean(dim=1)
    return kl_loss.mean()