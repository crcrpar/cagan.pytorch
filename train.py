import argparse
from datetime import datetime as dt
import json
from logging import DEBUG
from logging import FileHandler
from logging import getLogger
from logging import StreamHandler
import os

from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import tqdm

import dataset
import net


def as_variable(tensor, cuda):
    if cuda:
        tensor = tensor.cuda()
    return Variable(tensor)


def cat_2_images(img_1, img_2):
    return torch.cat([img_1, img_2], dim=1)


def cat_3_images(img_1, img_2, img_3):
    return torch.cat([img_1, img_2, img_3], dim=1)


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


def train(iter_, g, d, o_g, o_d, list_of_tensor, cycle_norm,
          gamma_cycle, gamma_id, cuda):
    g.train()
    d.train()

    human, item, want = [as_variable(t, cuda) for t in list_of_tensor]

    # 1st exchange
    input_1 = cat_3_images(human, item, want)
    y_1 = g(input_1)
    y_1, mask_1_norm = g.render(y_1, human)

    # 2nd exchange
    input_2 = cat_3_images(1, want, item)
    y_2 = g(input_2)
    y_2, mask_2_norm = g.render(y_2, y_1)

    # cycle loss
    cycle_loss = (human - y_2).norm(p=cycle_norm)

    # discrimination loss
    batch_size = human.size(0)
    positive_1 = cat_2_images(human, item)
    negative_1 = cat_2_images(y_1, want)
    negative_2 = cat_2_images(human, want)

    p_positive_1 = d(positive_1)
    p_negative_1 = d(negative_1)
    p_negative_2 = d(negative_2)

    # Backpropagation and Update
    # clear history
    o_g.zero_grad()
    o_d.zero_grad()
    p_positive_1
    loss_d = F.softplus(-p_positive_1) + \
        F.softplus(p_negative_1) + F.softplus(p_negative_2)
    loss_g = F.softplus(p_positive_1) + F.softplus(-p_negative_1) + F.softplus(-p_negative_2) + gamma_id * \
        (mask_1_norm + mask_2_norm) / (2.0 * batch_size) + \
        gamma_cycle * cycle_loss / batch_size
    loss_d.backward()
    loss_g.backward()
    o_g.step()
    o_d.step()

    return loss_d.data[0], loss_g.data[0], ((mask_1_norm + mask_2_norm) / 2.0).data[0], cycle_loss.data[0], y_1, y_2


def main():
    parser = argparse.ArgumentParser(description='Train CAGAN')
    # common
    parser.add_argument('--n_iter', default=10000, type=int,
                        help='number of update')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate of Adam')
    parser.add_argument('--cuda', default=0,
                        help='0 indicates CPU')
    parser.add_argument('--out', default='results',
                        help='log file')
    parser.add_argument('--log_interval', default=20, type=int,
                        help='log inteval')
    parser.add_argument('--ckpt_interval', default=50, type=int,
                        help='save interval')
    parser.add_argument('--seed', default=42)
    # model
    parser.add_argument('--deconv', default='upconv',
                        help='deconv or upconv')
    parser.add_argument('--relu', default='relu',
                        help='relu or relu6')
    parser.add_argument('--bias', default=True,
                        help='use bias or not')
    parser.add_argument('--init', default=None,
                        help='weight initialization')
    # loss
    parser.add_argument('--gamma_cycle', default=1.0, type=float,
                        help='coefficient for cycle loss')
    parser.add_argument('--gamma_id', default=1.0, type=float,
                        help='coefficient for mask')
    parser.add_argument('--norm', default=1, type=int,
                        help='selct norm type. Default is 1')
    parser.add_argument('--cycle_norm', default=2, type=int,
                        help='selct norm type. Default is 2')
    # dataset
    parser.add_argument('--root', default='data',
                        help='root directory')
    parser.add_argument('--base_root', default='images',
                        help='root directory to images')
    parser.add_argument('--triplet', default='triplet.json',
                        help='triplet list')
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
    log_dir = os.path.join(out, 'tensorboard')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    args.out = out
    args.log_dir = log_dir

    # logger
    logger = getLogger()
    f_h = FileHandler(os.path.join(out, 'log.txt'), 'a', 'utf-8')
    f_h.setLevel(DEBUG)
    s_h = StreamHandler()
    s_h.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(f_h)
    logger.addHandler(s_h)

    # tensorboard
    writer = SummaryWriter(log_dir)

    logger.debug("=====")
    logger.debug(json.dumps(args.__dict__, indent=2))
    logger.debug("=====")

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

    for iter_ in range(args.n_iter):
        # register params to tensorboard
        if args.cuda:
            generator = generator.cpu()
            discriminator = discriminator.cpu()
        for name, params in generator.named_children():
            if params.requires_grad():
                writer.add_histogram(tag='generator/' + name,
                                     values=params.data.numpy(),
                                     global_step=iter_)
        for name, params in discriminator.named_children():
            if params.requires_grad():
                writer.add_histogram(tag='discriminator/' + name,
                                     values=params.data.numpy(),
                                     global_step=iter_)
        if args.cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        list_of_tensor = train_loader.next()
        # y_1: exchanged, y_2: cycle
        loss_d, loss_g, mask_norm, cycle_loss, y_1, y_2 = train(
            iter_, generator, discriminator, opt_G, opt_D, list_of_tensor,
            args.cycle_norm, args.gamma_cycle, args.gamma_id, args.cuda)

        writer.add_scalars(
            main_tag='training',
            tag_scalar_dict={
                'discriminator/loss': loss_d,
                'generator/loss': loss_d,
                'generator/mask_norm': mask_norm,
                'generator/cycle_loss': cycle_loss
            }, global_step=iter_)

        if iter_ % args.log_interval == 0:
            msg = "Iter {} \tDis loss: {:.5f}\tGen loss: {:.5f}\tNorm: {:.5f}\n"
            logger.debug(
                msg.format(iter_, loss_d, loss_g, mask_norm + cycle_loss))

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
            if args.cuda:
                y_1 = y_1.cpu()
                y_2 = y_2.cpu()
            writer.add_image(tag='input/human',
                             image_tensor=list_of_tensor[0],
                             global_step=iter_)
            writer.add_image(tag='input/item',
                             image_tensor=list_of_tensor[1],
                             global_step=iter_)
            writer.add_image(tag='input/want'
                             image_tensor=list_of_tensor[2],
                             global_step=iter_)
            writer.add_image(tag='output/changed',
                             image_tensor=y_1.data,
                             global_step=iter_)
            writer.add_image(tag='output/cycle',
                             image_tensor=y_2.data,
                             global_step=iter_)
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

    writer.export_scalars_to_json(os.path.join(log_dir + 'scalars.json'))
    writer.close()

if __name__ == '__main__':
    main()
