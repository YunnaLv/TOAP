# UTRAP_672-toap
# CSQ-Res50-CASIA

import argparse
import random
import shutil

from torch import optim

from utils.noise_utils import *
from torchvision.utils import save_image
import os
from network import *
import time
from utils.tools import ImageList, CalcTopMap, image_transform, ImageList_
from utils.tools import compute_result_ as compute_result_org
from utils.votingForCenter import voting_center, voting_anchors, voting_center_
from utils.pic_quality import *
import torch.nn.functional as F
from GridCom import GridComDropout

# 316 改了loss输出的计算方法

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAH_0626/model/torch-model'


cpu_num = 5     # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

RANDOM_SEED = 333 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_subfold', type=str, default='UTRAP')
    parser.add_argument('--hash_bit', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1.0)  # tanh(αx)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument('--model_root', type=str, default='./save/')
    parser.add_argument('--retrieval_algo', type=str, default='CSQ')
    parser.add_argument('--model_type', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='CASIA')
    parser.add_argument('--n_class', type=int, default=28)
    parser.add_argument('--mAP', type=str, default='0.8754336332027824')
    parser.add_argument('--num_R', type=int, default=10)
    parser.add_argument('--num_M', type=int, default=10)

    parser.add_argument('--img_aug', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--test_txt', type=str, default='')

    parser.add_argument('--DI', type=str, default='True')
    parser.add_argument('--MI', type=str, default='True')
    parser.add_argument('--min_idx', type=int)
    parser.add_argument('--ComG_path', type=str, default='checkpoints/ComGAN/ComG_model.pt')
    parser.add_argument('--ComG_save_path', type=str, default='checkpoints/ComGAN/ComG_model_trained.pt')

    config = parser.parse_args()
    args = vars(config)
    return args


def args_setting(args):
    path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + args[
        'mAP']
    hashcenter_path = path + '/hashcenters.npy'
    model_path = path + '/model.pt'
    train_code_path = path + '/train_code.npy'
    args['hashcenters_path'] = hashcenter_path
    args['model_path'] = model_path
    args['train_code_path'] = train_code_path

    args['output_subfold'] = args['output_subfold'] + '_'

    if args['dataset'] == 'vggfaces2':
        args['data_path'] = '/data/UTAH_datasets/vggfaces2/'
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2/train.txt'
        args['test_txt'] = './data/vggfaces2/test.txt'
        args['min_idx'] = 3

    if args['dataset'] == 'CASIA':
        args['data_path'] = '/data/UTAH_datasets/CASIA-WebFace/'
        args['topk'] = 300
        args['train_txt'] = './data/CASIA/train.txt'
        args['test_txt'] = './data/CASIA/test.txt'
        args['min_idx'] = 2

    return args


def load_model(args):
    if 'ResNet' in args['model_type']:
        if args['retrieval_algo'] == "DHD":
            model = ResNet_Robust(args['model_type'])
            fc_dim, N_bits, NB_CLS = 2048, args['hash_bit'], args['n_class']
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = ResNet(args['hash_bit'], res_model=args['model_type'])
    elif 'Vgg' in args['model_type']:
        if args['retrieval_algo'] == "DHD":
            model = Vgg_Robust(args['model_type'])
            fc_dim, N_bits, NB_CLS = 4096, args['hash_bit'], args['n_class']
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = Vgg(args['hash_bit'], vgg_model=args['model_type'])
    else:
        raise NotImplementedError("Only ResNet and Vgg are implemented currently.")

    model.load_state_dict(torch.load(args['model_path'], map_location=args['device']))
    model.eval()
    return model


def load_model_and_hashcenter(args):
    hashcenters = np.load(args['hashcenters_path']).astype('float32')
    model = load_model(args).to(args['device'])
    target_hash = voting_center_(args['train_txt'], hashcenters, args['hash_bit']).to(args['device'])

    return model, hashcenters, target_hash


def load_data(args, list_path, data):
    dataset = ImageList(args['data_path'], open(list_path).readlines(),
                        transform=image_transform(256, 224, data, args['dataset']))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             shuffle=True, num_workers=args['num_workers'])
    return dataloader

def load_data_(args, list_path, target, data):
    dataset = ImageList_(args['data_path'], open(list_path).readlines(), target,
                        transform=image_transform(256, 224, data, args['dataset']))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             shuffle=True, num_workers=args['num_workers'])
    return dataloader

def exp_count(args):
    count = 0
    folder_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset']
    count_path = folder_path + '/count.txt'

    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)
        with open(count_path, 'a+') as f:
            f.write(str(count))
        return count
    else:
        with open(count_path) as f:
            count = int(f.readline()) + 1
        with open(count_path, 'w') as f:
            f.write(str(count))
        return count

def compute_loss_sub(batch_output, target):
    batch_size = batch_output.shape[0]

    products = 0
    k = torch.ones_like(batch_output).sum().item()
    for p in range(batch_size):
        product = batch_output[p] @ target[p].t()
        products += product

    loss = - products / k
    return loss, -loss, -loss

def compute_loss(batch_output, target_hash):
    products = batch_output @ target_hash.t()
    variant = torch.var(products)
    k = torch.ones_like(batch_output).sum().item() * len(target_hash)
    product_loss = products.sum() / k  # 越小越好 (和锚点的内积)
    loss = - (product_loss)  # 越大越好

    return loss, product_loss, variant


def attack(args, images, noise, hashcenters, model, dataset, target_hash, ComG):
    sub_loss = 0
    sub_prod_loss = 0
    sub_varient_loss = 0
    sub_loss_0 = 0
    # grads = torch.zeros_like(noise).to(args['device'])
    if noise.ndim == 4:
        grads = torch.zeros_like(noise[0]).to(args['device'])
    else:
        grads = torch.zeros_like(noise).to(args['device'])


    overall_anchor = voting_anchors(hashcenters, hash_bit=args['hash_bit'], is_father=True)
    sub_anchors = torch.as_tensor(
        voting_anchors(hashcenters, num_spts=args['num_R'], hash_bit=args['hash_bit'], is_father=False,
                       min_idx=args['min_idx']))

    adv_images = clamp_img(images + noise, dataset).to(args['device'])
    adv_images = adv_images.detach()

    img_size = images[0].size()
    task_images = torch.zeros(
        [args['img_aug'], len(adv_images), img_size[0], img_size[1], img_size[2]])  # img_aug, batch, 3, 224, 224
    for q in range(args['img_aug']):
        task_images[q] = adv_images
    task_images = task_images.view(-1, img_size[0], img_size[1], img_size[2])

    for o in range(1):
        spt_anchors = sub_anchors[o].unsqueeze(0)

        task_images = task_images.to(args['device'])
        spt_deltas = torch.zeros_like(task_images).to(args['device'])

        spt_deltas.requires_grad = True
        momentum = torch.zeros_like(spt_deltas).to(args['device'])

        pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(args['device'])
        _, output, _ = model(pert_images)

        ###
        Grid_Com = GridComDropout(224, ComG)
        grid_image = Grid_Com.GridAndCom(task_images.data + spt_deltas)
        dropnum = 0     # 就正常 多加一层ComG
        grid_list = [0, 2, 6, 8]
        grid_list += random.sample([1, 3, 4, 5, 7], 5 - dropnum)
        new_image = Grid_Com.merge_imgs(grid_image, dropnum=dropnum, grid_list=grid_list).cuda()
        pert_images_com = clamp_img(ComG(new_image), dataset).to(args['device'])

        _, output_com, _ = model(pert_images_com)
        loss_org, _, _ = compute_loss_sub(output, target_hash)  # 这是一批图像(batch_size)的loss
        loss_com, _, _ = compute_loss_sub(output_com, target_hash)
        loss = 0.3 * loss_org + 0.7 * loss_com
        loss.backward()
        sub_loss_0 += loss.data.cpu()
        spt_deltas.data = spt_deltas.data + 16/255 * spt_deltas.grad.sign()
        spt_deltas.data = clamp_noise(spt_deltas.data, dataset)
        spt_deltas.data = clamp_img(task_images.data + spt_deltas.data, dataset) - task_images.data

        new_pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(args['device'])
        _, new_outputs, _ = model(new_pert_images)
        new_loss_org, new_product_loss_org, new_varient_loss_org = compute_loss(new_outputs, spt_anchors.to(args['device']))  # 这是一批图像(batch_size)的loss

        grid_image_2 = Grid_Com.GridAndCom(task_images.data + spt_deltas)
        new_image_2 = Grid_Com.merge_imgs(grid_image_2, dropnum=dropnum, grid_list=grid_list).cuda()

        new_pert_images_com = clamp_img(ComG(new_image_2), dataset).to(args['device'])

        _, new_outputs_com, _ = model(new_pert_images_com)
        new_loss_com, new_product_loss_com, new_varient_loss_com = compute_loss(new_outputs_com, spt_anchors.to(args['device']))  # 这是一批图像(batch_size)的loss
        new_loss = 0.3 * new_loss_org + 0.7 * new_loss_com
        new_product_loss = - new_loss
        new_varient_loss = - new_loss
        new_loss.backward()
        sub_loss += new_loss.data.cpu()
        sub_prod_loss += new_product_loss.data.cpu()
        sub_varient_loss += new_varient_loss.data.cpu()
        grads += spt_deltas.grad.data.sum(0)
        spt_deltas.grad.data.zero_()

    return grads, sub_loss, sub_prod_loss, sub_varient_loss, -sub_loss_0

def compute_hammingdist(code, com_code):
    q = code.shape[1]  # bits
    num = code.shape[0]  # num
    avg_dist = q - torch.sum(code * com_code, dim=1).sum() / num
    return avg_dist

def generate_universal_noise(args, model, hashcenters, test_loader, train_loader, count, ComG):
    noise = noise_initialization()
    noise = torch.from_numpy(noise).to(args['device'])
    noise.requires_grad = True
    tst_mAP, Best_mAP = 0, 1.0
    momentum = torch.zeros_like(noise).to(args['device'])

    for epoch in range(args['epochs']):
        batch_grads = []

        total_loss = 0
        total_prod_loss = 0
        total_prod_loss_0 = 0
        counter = 0
        for idx, (image, label, _, target_hash) in enumerate(train_loader):
            if idx % 20 == 0:
                print(idx)
            counter = idx
            image = image.to(args['device'])
            inner_grad, sub_loss, sub_prod_loss, _, sub_prod_loss_0 = attack(args, image, noise, hashcenters, model, args['dataset'], target_hash, ComG)
            inner_grad = inner_grad / (torch.mean(torch.abs(inner_grad), (0, 1, 2), keepdim=True) + 1e-12)
            batch_grads.append(inner_grad.cpu())
            total_loss += sub_loss
            total_prod_loss_0 += sub_prod_loss_0
            total_prod_loss += sub_prod_loss
        print('Epoch %d: Avg attack loss: %f, Avg product loss: %f, Avg product loss sub_task: %f' % (
            epoch, total_loss / counter, total_prod_loss / counter, total_prod_loss_0 / counter))
        final_grad = torch.stack(batch_grads).sum(0).to(args['device'])

        # MI
        if args['MI'] == 'True':
            final_grad = momentum * 0.8 + final_grad / (
                        torch.mean(torch.abs(final_grad), (0, 1, 2), keepdim=True) + 1e-12)
            momentum = final_grad
        else:
            final_grad = final_grad

        noise.data = noise.data + 0.02 * final_grad.sign()
        noise.data = clamp_noise(noise.data, args['dataset'])

        tr_mAP = train_mAP(args, train_loader, noise.clone(), model, args['dataset'])
        tst_mAP, save_noise_npy = test_mAP(args, test_loader, noise.clone(), model, count, epoch, epoch, args['dataset'])
        train_pic = compute_ssim_mse_psnr_(train_loader, noise.clone().detach().cpu(), model, args['dataset'])
        print("[train] ssim =", train_pic[0], ", mse =", train_pic[1], ", psnr =", train_pic[2])
        test_pic = compute_ssim_mse_psnr_(test_loader, noise.clone().detach().cpu(), model, args['dataset'])
        print("[test] ssim =", test_pic[0], ", mse =", test_pic[1], ", psnr =", test_pic[2])
        draw_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' \
                    + args['dataset'] + '/' + args['output_subfold'] + str(count) + '/draw'
        if not os.path.exists(draw_path):
            os.mkdir(draw_path)
        save_mAP_quality(draw_path, tr_mAP, tst_mAP, train_pic, test_pic, total_prod_loss / counter, total_prod_loss_0 / counter)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print('[epoch]', epoch, ', [current_time]', current_time)

        if tst_mAP < Best_mAP:
            Best_mAP = tst_mAP
            save_imgs(args, noise, epoch, count, tst_mAP)
            save_noise_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args[
                'dataset'] + '/' + args['output_subfold'] + str(count) + "/best"
            np.save(save_noise_path + '_noise.npy', save_noise_npy)
            # *** best_noise时 训练压缩模拟器(微调) ***
            noise_detach = noise.data.detach()
            for idx, (image, label, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                image = image.to(args['device'])
                adv_image = clamp_img(image + noise_detach, None).cuda()
                comG_adv_image = ComG(adv_image)
                loss_MSE_img = torch.nn.functional.mse_loss(image, comG_adv_image)
                loss_MSE_features = torch.nn.functional.mse_loss(model(image)[2], model(comG_adv_image)[2]) * 10e-3
                loss_hammdist = compute_hammingdist(model(image)[1], model(comG_adv_image)[1]) * 10e-5
                loss_G = loss_MSE_img + loss_MSE_features + loss_hammdist
                loss_G.backward()
                optimizer.step()

            model_dict = {'ComG': ComG}
            torch.save(model_dict, args['ComG_save_path'])

    o_mAP = org_mAP(args, test_loader, model)
    record(o_mAP, tst_mAP, test_pic, draw_path)


def save_imgs(args, noise, epoch, count, mAP):
    now = "epoch_" + str(epoch)
    path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset'] + '/' + args[
        'output_subfold'] + str(
        count)
    noise_name = '/noise_' + now + "_" + str(mAP) + '.JPEG'

    save_image(
        (noise.clone().squeeze(0) + torch.abs(torch.min(noise))) / (torch.max(noise) + torch.abs(torch.min(noise))),
        path + noise_name)


def test_mAP(args, test_loader, noise, model, count, epoch, idx, dataset):
    per_codes, org_codes, org_labels = compute_result(test_loader, noise, model, device=args['device'], dataset=dataset)
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print('test_mAP =', mAP)
    exp_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args[
        'dataset'] + '/' + args['output_subfold'] + str(count) + "/" + str(epoch) + '_' + str(idx)
    np.save(exp_path + '_per_codes.npy', per_codes.numpy())
    np.save(exp_path + '_org_codes.npy', org_codes.numpy())
    np.save(exp_path + '_org_labels.npy', org_labels.numpy())
    np.save(exp_path + '_noise.npy', noise.clone().detach().cpu().numpy())
    return mAP, noise.clone().detach().cpu().numpy()

def compute_result(dataloader, noise, net, device, dataset):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _, _ in tqdm(dataloader):
        img = img.to(device)
        per_images = clamp_img(img + noise, dataset)
        bs.append((net(per_images.to(device))[0]).data.cpu())
        bs_2.append((net(img.to(device))[0]).data.cpu())
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

def train_mAP(args, train_loader, noise, model, dataset):
    per_codes, org_codes, org_labels = compute_result(train_loader, noise, model, device=args['device'],
                                                      dataset=dataset)
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print("train_mAP =", mAP)
    return mAP


def save_mAP_quality(draw_path, tr_mAP, tst_mAP, train_pic, test_pic, loss_1, loss_2):
    train_mAP_path = draw_path + '/train_mAP.txt'
    test_mAP_path = draw_path + '/test_mAP.txt'

    train_ssim_path = draw_path + '/train_ssim.txt'
    train_mse_path = draw_path + '/train_mse.txt'
    train_psnr_path = draw_path + '/train_psnr.txt'
    test_ssim_path = draw_path + '/test_ssim.txt'
    test_mse_path = draw_path + '/test_mse.txt'
    test_psnr_path = draw_path + '/test_psnr.txt'

    loss_path_1 = draw_path + '/loss_overall.txt'
    loss_path_2 = draw_path + '/loss_sub.txt'
    loss_path_3 = draw_path + '/loss_add.txt'
    with open(loss_path_1, "a") as f:
        f.write(str(loss_1.data) + '\n')
    with open(loss_path_2, "a") as f:
        f.write(str(loss_2.data) + '\n')
    with open(loss_path_3, "a") as f:
        f.write(str((loss_1+loss_2).data) + '\n')

    with open(train_mAP_path, "a") as f:
        f.write(',' + str(tr_mAP) + '\n')
    with open(test_mAP_path, "a") as f:
        f.write(',' + str(tst_mAP) + '\n')

    with open(train_ssim_path, "a") as f:
        f.write(',' + str(train_pic[0]) + '\n')
    with open(train_mse_path, "a") as f:
        f.write(',' + str(train_pic[1]) + '\n')
    with open(train_psnr_path, "a") as f:
        f.write(',' + str(train_pic[2]) + '\n')
    with open(test_ssim_path, "a") as f:
        f.write(',' + str(test_pic[0]) + '\n')
    with open(test_mse_path, "a") as f:
        f.write(',' + str(test_pic[1]) + '\n')
    with open(test_psnr_path, "a") as f:
        f.write(',' + str(test_pic[2]) + '\n')


def record(org_mAP, tst_mAP, pic_quality, path):
    with open(path + "/record.txt", "w") as f:
        mAP_record = "org_mAP = " + str(org_mAP) + " --> " + "per_mAP = " + str(tst_mAP)
        ssim_record = "ssim = " + str(pic_quality[0])
        mse_record = "mse = " + str(pic_quality[1])
        psnr_record = "psnr = " + str(pic_quality[2])
        f.write(mAP_record + '\n')
        f.write(ssim_record + '\n')
        f.write(mse_record + '\n')
        f.write(psnr_record)


def input_diversity(img, DI='True'):
    if DI == 'True':
        size = img.size(2)
        resize = int(size / 0.875)

        rnd = torch.randint(size, resize + 1, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
        h_rem = resize - rnd
        w_hem = resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (size, size), mode="nearest")

        p = torch.rand(1).item()
        if p > 0.5:
            return padded
        else:
            return img
    else:
        return img


def org_mAP(args, test_loader, model):
    org_codes, org_labels = compute_result_org(test_loader, model, args['device'])
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, org_codes, db_labels, org_labels, args['topk'])
    return mAP

def copy_current_script(output_dir):
    current_script_path = os.path.abspath(__file__)
    os.makedirs(output_dir, exist_ok=True)

    script_name = os.path.basename(current_script_path)
    output_path = os.path.join(output_dir, script_name)

    # 复制当前脚本到指定目录
    shutil.copy(current_script_path, output_path)
    print(f"Current script copied to {output_path}")

if __name__ == '__main__':
    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print('[current_time]', current_time)
    args = get_args()
    args = args_setting(args)
    model, hashcenters, target_hash = load_model_and_hashcenter(args)
    test_loader = load_data_(args, args['test_txt'], target_hash, data='test')
    train_loader = load_data_(args, args['train_txt'], target_hash, data='train')
    count = exp_count(args)
    exp_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset'] + '/' + args[
        'output_subfold'] + str(count)
    os.makedirs(exp_path, exist_ok=True)
    copy_current_script(exp_path)
    ComG_model = torch.load(args['ComG_path'])
    ComG = ComG_model['ComG'].to(args['device'])
    optimizer = optim.SGD(ComG.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    # GridCom = GridCom(224, ComG)
    generate_universal_noise(args, model, hashcenters, test_loader, train_loader, count, ComG)
