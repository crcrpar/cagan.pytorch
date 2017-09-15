import argparse
from datetime import datetime as dt
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tqdm

import dataset
import net


def asVariable(tensor, cuda):
    if cuda:
        tensor = tensor.cuda()
    return Variable(tensor)


def normalize(variable):
    var_size = variable.size()
    var_min, var_max = torch.min(variable), torch.max(variable)
    variable = (variable - var_min) / var_max
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1).expand_as(var_size)
    std = torch.FloatTensor([0.229, 0.234, 0.225]).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1).expand_as(var_size)
    return (variable - mean) / std


def patch_loss(y_in, y_out, same=True):
    batch_size, _, w, h = y_in.size()
    denom = batch_size * w * h
    if same:
        l1 = F.softplus(y_in).sum() / denom
    else:
        l1 = F.softplus(-y_in).sum() / denom
    l2 = F.softplus(y_out).sum() / denom
    return l1 + l2


def save_image_tensor(tensor, filename):
    root = os.path.dirname(filename)
    if not os.path.isdir(root):
        os.makedirs(root)
    torchvision.utils.save_image(tensor, filename, normalize=True,
                                 scale_each=True)


def main():
    parser = argparse.ArgumentParser(description='Train CAGAN')
    parser.add_argument('--n_iter', default=10000, type=int,
                        help='number of update')
    parser.add_argument('--deconv', default='upconv',
                        help='deconv or upconv')
    parser.add_argument('--relu', default='relu',
                        help='relu or relu6')
    parser.add_argument('--bias', default=True,
                        help='use bias or not')
    parser.add_argument('--init', default=None,
                        help='weight initialization')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate of Adam')
    parser.add_argument('--cuda', default=0,
                        help='0 indicates CPU')
    parser.add_argument('--batch_size', '-bs', default=16,
                        help='batch size')
    parser.add_argument('--gamma_cycle', default=1.0,
                        help='coefficient for cycle loss')
    parser.add_argument('--gamma_id', default=1.0,
                        help='coefficient for mask')
    parser.add_argument('--norm', default=1, type=int,
                        help='selct norm type. Default is 1')
    parser.add_argument('--root', default='data',
                        help='root directory')
    parser.add_argument('--base_root', default='images',
                        help='root directory to images')
    parser.add_argument('--triplet', default='triplet.json',
                        help='triplet list')
    parser.add_argument('--out', default='result',
                        help='save directory')
    parser.add_argument('--log', default='log',
                        help='log file')
    parser.add_argument('--log_interval', default=20, type=int,
                        help='log inteval')
    parser.add_argument('--ckpt_interval', default=50, type=int,
                        help='save interval')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    time = dt.now().strftime('%m%d_%H%M')
    print('+++++ begin at {} +++++'.format(time))
    for key, value in dict(args):
        print('### {}: {}'.format(key, value))
    print('+++++')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    out = os.path.join(args.out, time)
    if not os.path.isdir(out):
        os.makedirs(out)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset.TripletDataset(
            args.root,
            os.path.join(args.root, args.triplet),
            transform=transforms.Compose([
                transforms.Scale((132, 100)),
                transforms.RandomCrop((128, 96)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.234, 0.225]),
                transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    generator = net.Generator(args.deconv, args.relu, args.bias, args.norm)
    discriminator = net.Discriminator(args.relu, args.bias)
    opt_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    for iter_ in tqdm.tqdm(range(args.n_iter)):
        generator.train()
        discriminator.train()

        human, item, want = train_loader.next()
        human = asVariable(human, args.cuda)
        item = asVariable(item, args.cuda)
        want = asVariable(want, args.cuda)

        # Generator runs twice.
        # # human, item, want -> y
        # # y, want, item -> hat_human
        # L2 loss: hat_human <-> human
        # L1 loss: mask's L1 norm

        gen_input_1 = torch.cat([human, item, want], dim=1)
        y_1 = generator(gen_input_1)
        y_1, mask_l1_norm_1 = generator.render(y_1, human)

        gen_input_2 = torch.cat([normalize(y_1), want, item], dim=1)
        y_2 = generator(gen_input_2)
        y_2, _ = generator.render(y_2, y_1)

        loss_cycle = torch.norm(human - y_2, p=2)

        # Discriminator runs 3 times.
        # # positive examples (human, item)
        # # negative examples (y, want), (human, want)
        batch_size = human.size(0)
        feature_human = discriminator(human)
        feature_item = discriminator(item)
        feature_want = discriminator(want)
        feature_y_1 = discriminator(y_1)

        positive_loss = patch_loss(feature_human, feature_item, same=True)
        negative_loss = patch_loss(
            feature_y_1, want, same=False) + patch_loss(feature_item, feature_want, same=False)
        adversarial_loss = positive_loss + negative_loss

        # Backprop gradient and update parameters
        opt_G.zero_grad()
        opt_D.zero_grad()
        adversarial_loss.backward()
        normalizing_term = args.gamma_id * mask_l1_norm_1 + args.gamma_cycle * loss_cycle
        normalizing_term.backward()
        opt_G.step()
        opt_D.step()

        if iter_ % args.log_interval == 0:
            positive_loss = positive_loss.data[0] / batch_size
            negative_loss = negative_loss.data[0] / batch_size
            normalizing_term = normalizing_term.data[0] / batch_size
            msg = "Iter {} [{}/{}]\tPosi loss: {:.5f}\tNega loss: {:.5f}\tNorm: {:.5f}\n"
            tqdm.tqdm.write(msg.format(iter_, iter_, args.n_iter,
                                       positive_loss, negative_loss, normalizing_term))

        if iter_ % args.ckpt_interval == 0:
            # save checkpoint and images used in this iteration
            if args.cuda:
                generator = generator.cpu()
                discriminator = discriminator.cpu()
            generator.eval()
            discriminator.eval()
            torch.save({
                'iteration': iter_,
                'gen_state': generator.state_dict(),
                'dis_state': discriminator.state_dict(),
                'opt_gen': opt_G.state_dict(),
                'opt_dis': opt_D.state_dict()},
                os.path.join(out, 'ckpt_iter_{}.pth'.format(iter_)))

            # save images
            image_root = os.path.join(out, 'images', 'iter_{}'.format(iter_))
            save_image_tensor(human, os.path.join(image_root, 'human.png'))
            save_image_tensor(item, os.path.join(image_root, 'item.png'))
            save_image_tensor(want, os.path.join(image_root, 'want.png'))
            save_image_tensor(y_1, os.path.join(image_root, 'y_1.png'))
            save_image_tensor(y_2, os.path.join(image_root, 'y_2_cycle.png'))
            if args.cuda:
                generator = generator.cuda()
                discriminator = discriminator.cuda()

    # save checkpoint
    torch.save({
        'iteration': args.n_iter,
        'gen_state': generator.state_dict(),
        'dis_state': discriminator.state_dict(),
        'opt_gen': opt_G.state_dict(),
        'opt_dis': opt_D.state_dict()},
        os.path.join(out, 'ckpt_iter_{}.pth'.format(args.n_iter)))


if __name__ == '__main__':
    main()
