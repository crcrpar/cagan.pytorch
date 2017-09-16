import argparse
from datetime import datetime as dt
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def patch_dis_loss(y_positive, y_negative):
    batch_size, _, w, h = y_positive.size()
    denom = batch_size * w * h
    l1 = F.softplus(-y_positive).sum() / denom
    l2 = F.softplus(y_negative).sum() / denom
    return l1 + l2


def patch_gen_loss(y_out):
    batch_size, _, w, h = y_out.size()
    denom = batch_size * w * h
    loss_adv = F.softplus(-y_out).sum() / denom
    return loss_adv


def plot(label, x_ticks, y_values, path):
    root = os.path.dirname(path)
    if not os.path.isdir(root):
        os.makedirs(root)
    f = plt.figure()
    a = f.add_subplot(111)
    a.set_xlabel('iteration')
    a.set_xticks(label)
    a.set_ylabel(label)
    a.grid()
    a.plot(x_ticks, y_values, marker='x', label=label)
    l = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    f.savefig(path, bboox_extra_artists=(l,), bbox_inches='tight')


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

    x_ticks, dis_loss_list = list(), list()
    gen_loss_list, normalization_list = list(), list()
    plot_root = os.path.join(out, 'plot')
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
        human_out = generator(gen_input_1)
        human_out, mask_l1_norm_1 = generator.render(human_out, human)

        gen_input_2 = torch.cat([normalize(human_out), want, item], dim=1)
        human_cycle = generator(gen_input_2)
        human_cycle, _ = generator.render(human_cycle, human_out)

        loss_cycle = args.gamma_cycle * torch.norm(
            human - human_cycle, p=args.norm)

        # Discriminator runs 3 times.
        # # positive examples (human, item)
        # # negative examples (y, want), (human, want)
        batch_size = human.size(0)
        positive_input = F.cat([human, item], dim=1)
        negative_input_1 = F.cat([human_out, want], dim=1)
        negative_input_2 = F.cat([human, want], dim=1)

        y_true = discriminator(positive_input)
        y_fake_1 = discriminator(negative_input_1)
        y_fake_2 = discriminator(negative_input_2)

        loss_dis = patch_dis_loss(y_true, y_fake_2)
        loss_adv_gen = patch_gen_loss(y_fake_1)
        loss_gen = loss_adv_gen +\
            args.gamma_id * mask_l1_norm_1 + args.gamma_cycle * loss_cycle

        # Backprop gradient and update parameters
        opt_G.zero_grad()
        loss_gen.backward()
        opt_G.step()
        opt_D.zero_grad()
        loss_dis.backward()
        opt_D.step()

        if iter_ % args.log_interval == 0:
            batch_size = human.size(0)
            l_D = loss_dis.cpu().data[0] / batch_size
            l_G_adv = loss_gen.cpu().data[0] / batch_size
            l_G_norm = args.gamma_id * mask_l1_norm_1 + loss_cycle
            l_G_norm = l_G_norm.cpu().data[0] / batch_size
            line = '#Iteration {}/{}\tdis/loss: {:4f}\tgen/loss: {:.4f}\t'\
                'gen/normalization: {:4f}\n'
            tqdm.tqdm.write(line.format(
                iter_, args.n_iter, l_D, l_G_adv, l_G_norm))
            x_ticks.append(iter_)
            dis_loss_list.append(l_D)
            gen_loss_list.append(l_G_adv)
            normalization_list.append(l_G_norm)

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
            save_image_tensor(human_out, os.path.join(
                image_root, 'human_out.png'))
            save_image_tensor(human_cycle, os.path.join(
                image_root, 'human_cycle_cycle.png'))
            # save plot
            plot('dis_loss', x_ticks, dis_loss_list,
                 os.path.join(plot_root, 'dis_loss.png'))
            plot('gen_loss', x_ticks, gen_loss_list,
                 os.path.join(plot_root, 'gen_loss.png'))
            plot('gen_normalizing_term', x_ticks, normalization_list,
                 os.path.join(plot_root, 'normalizing_term.png'))

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
