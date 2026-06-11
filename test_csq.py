import torch
from network import *
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.noise_utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from DiffJPEG import DiffJPEG
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda:0'

cpu_num = 5     # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 模型加载
def load_model(hash_bit, specifc_model, model_path):
    if 'ResNet' in specific_model:
        if 'DHD' in model_path:
            model = ResNet_Robust(specific_model)
            fc_dim, N_bits, NB_CLS = 2048, hash_bit, 28
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = ResNet(hash_bit, res_model=specific_model)
    elif 'Vgg' in specific_model:
        if 'DHD' in model_path:
            model = Vgg_Robust(specific_model)
            fc_dim, N_bits, NB_CLS = 4096, hash_bit, 28
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = Vgg(hash_bit, vgg_model=specific_model)
    else:
        raise NotImplementedError("Only ResNet and Vgg are implemented currently.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_data(data_path, list_path, batch_size, resize_size, crop_size, data, dset):
    dataset = ImageList(data_path, open(list_path).readlines(),
                        transform=image_transform(resize_size, crop_size, data, dset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

# 压缩
def image_jpeg(images, quality, dataset):
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality, test=True)
    return jpeg(images)

# resize
def image_resize(images, new_size):
    transform_resize = transforms.Compose([
        transforms.Resize((new_size, new_size)),
        transforms.Resize((224, 224)),
    ])
    return transform_resize(images)

# 模糊
def image_GaussianBlur(images, ker, cir_len):
    transform_blur = transforms.Compose([
        transforms.GaussianBlur(ker, cir_len)
    ])
    return transform_blur(images)

# 高斯噪声
def image_noise(images, noise_v):
    mask = np.random.normal(0, noise_v, (3, 224, 224))
    mask = torch.from_numpy(mask).float().cuda()
    return images + mask

def image_rotate(images, angle):
    transform_rotate = transforms.Compose([
        transforms.RandomRotation((angle, angle))
    ])
    return transform_rotate(images)

# img + noise 计算net(dataloader + noise)结果
def compute_result(dataloader, noise, net, dataset, mode):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset).to(device)
        img = img.to(device)
        # save_image(un_normalize(img[0], None), './save_img/test_.png')
        # org   不进入if else
        if mode == 1:
            perturbated_images = image_jpeg(perturbated_images, 90, dataset)
            img = image_jpeg(img, 90, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 2:
            perturbated_images = image_jpeg(perturbated_images, 80, dataset)
            img = image_jpeg(img, 80, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 3:
            perturbated_images = image_jpeg(perturbated_images, 70, dataset)
            img = image_jpeg(img, 70, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 4:
            perturbated_images = image_jpeg(perturbated_images, 60, dataset)
            img = image_jpeg(img, 60, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 5:
            perturbated_images = image_jpeg(perturbated_images, 50, dataset)
            img = image_jpeg(img, 50, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 6:
            perturbated_images = image_jpeg(perturbated_images, 40, dataset)
            img = image_jpeg(img, 40, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 7:
            perturbated_images = image_jpeg(perturbated_images, 30, dataset)
            img = image_jpeg(img, 30, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)

        # 1. resize
        elif mode == 8:
            perturbated_images = image_resize(perturbated_images, 168)
            img = image_resize(img, 168)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 9:
            perturbated_images = image_resize(perturbated_images, 280)
            img = image_resize(img, 280)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 10:
            perturbated_images = image_resize(perturbated_images, 336)
            img = image_resize(img, 336)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 11:
            perturbated_images = image_resize(perturbated_images, 392)
            img = image_resize(img, 392)
            perturbated_images = clamp_img(perturbated_images, dataset)
        # 2. 模糊
        elif mode == 12:
            perturbated_images = image_GaussianBlur(perturbated_images, 3, 0.8)
            img = image_GaussianBlur(img, 3, 0.8)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 13:
            perturbated_images = image_GaussianBlur(perturbated_images, 3, 3)
            img = image_GaussianBlur(img, 3, 3)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 14:
            perturbated_images = image_GaussianBlur(perturbated_images, 5, 1.1)
            img = image_GaussianBlur(img, 5, 1.1)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 15:
            perturbated_images = image_GaussianBlur(perturbated_images, 5, 3)
            img = image_GaussianBlur(img, 5, 3)
            perturbated_images = clamp_img(perturbated_images, dataset)
        # 3. 高斯噪声
        elif mode == 16:
            perturbated_images = image_rotate(perturbated_images, 10)
            img = image_rotate(img, 10)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 17:
            perturbated_images = image_rotate(perturbated_images, 20)
            img = image_rotate(img, 20)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 18:
            perturbated_images = image_rotate(perturbated_images, 30)
            img = image_rotate(img, 30)
            perturbated_images = clamp_img(perturbated_images, dataset)

        # noise
        elif mode == 19:
            perturbated_images = image_noise(perturbated_images, 0.001 ** 0.5)
            img = image_noise(img, 0.001 ** 0.5)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 20:
            perturbated_images = image_noise(perturbated_images, 0.002 ** 0.5)
            img = image_noise(img, 0.002 ** 0.5)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 21:
            perturbated_images = image_noise(perturbated_images, 0.003 ** 0.5)
            img = image_noise(img, 0.003 ** 0.5)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 22:
            perturbated_images = image_noise(perturbated_images, 0.004 ** 0.5)
            img = image_noise(img, 0.004 ** 0.5)
            perturbated_images = clamp_img(perturbated_images, dataset)

        # save_image(un_normalize(img[0], None), './save_img/test.png')
        # break
        bs.append((net(perturbated_images)[0]).data.cpu())  # 扰动图像
        bs_2.append((net(img)[0]).data.cpu())  # 原图
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

if __name__ == '__main__':
    # ######## 64bit CSQ
    save_path1 = "/data/UTAH_code/UTRAP/save/CSQ/ResNet34/CASIA/0.8841874375307698/"  # CSQ-ResNet34
    save_path2 = "/data/UTAH_code/UTRAP/save/CSQ/ResNet50/CASIA/0.8754336332027824/"  # CSQ-ResNet50
    save_path3 = "/data/UTAH_code/UTRAP/save/CSQ/Vgg16/CASIA/0.8342357348593585/"  # CSQ-Vgg16
    save_path4 = "/data/UTAH_code/UTRAP/save/CSQ/Vgg19/CASIA/0.8207702018196129/"  # CSQ-Vgg19

    # model
    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model2 = load_model(64, 'ResNet50', save_path2 + "model.pt")
    model3 = load_model(64, 'Vgg16', save_path3 + "model.pt")
    model4 = load_model(64, 'Vgg19', save_path4 + "model.pt")

    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)

    # 加载database code + label   (检索算法中计算出来的)
    database_code1 = np.load(save_path1 + "database_code.npy")
    database_label1 = np.load(save_path1 + "database_label.npy")
    database_code2 = np.load(save_path2 + "database_code.npy")
    database_label2 = np.load(save_path2 + "database_label.npy")
    database_code3 = np.load(save_path3 + "database_code.npy")
    database_label3 = np.load(save_path3 + "database_label.npy")
    database_code4 = np.load(save_path4 + "database_code.npy")
    database_label4 = np.load(save_path4 + "database_label.npy")
    #
    # 加载database图像path
    database_txt_path = './data/CASIA/database.txt'
    database_img_path = np.array(open(database_txt_path).readlines())

    # 数据集
    dset = None
    # 加载测试集
    data_path = '/data/UTAH_datasets/CASIA-WebFace/'
    list_path = './data/CASIA/test.txt'
    test_loader = load_data(data_path, list_path, 1, 256, 224, 'test', dset)

    noise_path = '/data/UTAH_code/UTRAP/exp/CSQ/Vgg16/CASIA/UTRAP_165/'
    name = '99_99_noise.npy'
    noise_ = np.load(noise_path + name)
    # noise_ = np.zeros_like(noise_)

    noise = torch.from_numpy(noise_)
    noise = clamp_noise(noise, dset)

    topk = 300
    save_dir = noise_path + 'log'
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir)
    log.info("testing")
    log.info(name)

    # for m in range(23):
    for m in [0, 4, 10, 12, 16, 20]:
    # for m in [0]:
        # if m == 12:
        #     continue
        log.info('============ mode ' + str(m) + ' =============')
        # per_codes1, org_codes1, org_labels1 = compute_result(test_loader, noise, model1, dataset=dset, mode=m)  # tqdm
        # org_mAP = CalcTopMap(database_code1, org_codes1, database_label1, org_labels1, topk)
        # per_mAP = CalcTopMap(database_code1, per_codes1, database_label1, org_labels1, topk)
        # log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        per_codes2, org_codes2, org_labels2 = compute_result(test_loader, noise, model2, dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code2, org_codes2, database_label2, org_labels2, topk)
        per_mAP = CalcTopMap(database_code2, per_codes2, database_label2, org_labels2, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # 对于Vgg16的:
        per_codes3, org_codes3, org_labels3 = compute_result(test_loader, noise, model3, dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code3, org_codes3, database_label3, org_labels3, topk)
        per_mAP = CalcTopMap(database_code3, per_codes3, database_label3, org_labels3, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # # 对于Vgg16的:
        # per_codes3, org_codes3, org_labels3 = compute_result(test_loader, noise, model3, dataset=dset, mode=m)  # tqdm
        # org_mAP = CalcTopMap(database_code3, org_codes3, database_label3, org_labels3, topk)
        # per_mAP = CalcTopMap(database_code3, per_codes3, database_label3, org_labels3, topk)
        # log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # # 对于Vgg19的:
        # per_codes4, org_codes4, org_labels4 = compute_result(test_loader, noise, model4, dataset=dset, mode=m)  # tqdm
        # org_mAP = CalcTopMap(database_code4, org_codes4, database_label4, org_labels4, topk)
        # per_mAP = CalcTopMap(database_code4, per_codes4, database_label4, org_labels4, topk)
        # log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))
        # if m == 0:
        #     np.save('./codes/LO_org_codes_resnet50_wocomg.npy', org_codes2.numpy())
        #     np.save('./codes/LO_org_codes_vgg16_wocomg.npy', org_codes3.numpy())
        #     np.save('./codes/LO_per_codes_resnet50_wocomg.npy', per_codes2.numpy())
        #     np.save('./codes/LO_per_codes_vgg16_wocomg.npy', per_codes3.numpy())