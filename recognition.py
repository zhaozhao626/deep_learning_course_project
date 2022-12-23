import os
print (os.getcwd()) #获取当前工作目录路径
import warnings
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
import math
import pyclipper
import imgaug
import imgaug.augmenters as iaa

from PIL import Image
from shapely.geometry import Polygon
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# 是否使用 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class RecOptions():
    def __init__(self):
        self.height = 32              # 图像尺寸
        self.width = 100         
        self.voc_size = 21            # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5              # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]    # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80
        self.batch_size = 64
        self.num_workers = 8
        self.print_interval = 25
        self.save_interval = 125
        self.train_dir = './temp/rec_datasets/train_imgs'
        self.test_dir = './temp/rec_datasets/test_imgs'
        self.save_dir = './temp/rec_models/'
        self.saved_model_path = './temp/rec_models/checkpoint_final'
        self.rec_res_dir = './temp/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

rec_args = RecOptions()
DEBUG = False  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
# 检测和识别模型需要足够的训练迭代次数，因此DEBUG模式下几乎得不到最终有效结果
if DEBUG:
    rec_args.max_epoch = 1
    rec_args.print_interval = 20
    rec_args.save_interval = 1

    rec_args.batch_size = 10
    rec_args.num_workers = 0

'''
标签处理：定义新字符类处理半字符的情况，比如将'0-1半字符'归到'A'类，减小歧义
识别训练数据构造：从完整图像中裁剪出文本图像作为识别模型输入数据
'''
def PreProcess():
    EXT_CHARS = {
        '01': 'A', '12': 'B', '23': 'C', '34': 'D', '45': 'E',
        '56': 'F', '67': 'G', '78': 'H', '89': 'I', '09': 'J'
    }

    train_dir = './data/train_imgs'
    train_labels_dir = './data/train_gts'
    word_save_dir = './temp/rec_datasets/train_imgs'      # 保存识别训练数据集


    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(train_labels_dir)
    for label_file in tqdm(label_files):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        line = lines[0].strip().split()
        locs = line[:8]
        words = line[8:]
        
        # 标签处理
        if len(words) == 1:
            ext_word = words[0]
        else:
            assert len(words) % 2 == 0
            ext_word = ''
            for i in range(len(words[0])):
                char_i = [word[i] for word in words]
                if len(set(char_i)) == 1:
                    ext_word += char_i[0]
                elif len(set(char_i)) == 2:
                    char_i = list(set(char_i))
                    char_i.sort()
                    char_i = ''.join(char_i)
                    ext_char_i = EXT_CHARS[char_i]
                    ext_word += ext_char_i

        locs = [int(t) for t in line[:8]]
            
        # 将倾斜文字图像调整为水平图像
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2-x1)**2+(y2-y1)**2)**0.5 + ((x4-x3)**2+(y4-y3)**2)**0.5))
        h = int(0.5 * (((x2-x3)**2+(y2-y3)**2)**0.5 + ((x4-x1)**2+(y4-y1)**2)**0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(train_dir, label_file.replace('.txt', '.jpg')))
        word_image = cv2.warpPerspective(image, M, (w, h))
        
        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('.txt', '')+'_'+ext_word+'.jpg'), word_image)



'''
数据集导入方法
'''
# data
class WMRDataset(data.Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train
        
        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in os.listdir(data_dir) if t.endswith('.jpg')]
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.targets)
    
    @staticmethod
    def gen_labelmap(charset='0123456789ABCDEFGHIJ'):
        # 构造字符和数字标签对应字典
        PADDING = 'PADDING'
        char2id = {t:idx for t, idx in zip(charset, range(1, 1+len(charset)))}
        char2id.update({PADDING:0})
        id2char = {v:k for k, v in char2id.items()}
        return PADDING, char2id, id2char

    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.targets[index][0]
        word = self.targets[index][1]
        img = Image.open(img_path)
        
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        word = word[:self.max_len]
        for char in word:
            label_list.append(self.char2id[char])
            
        label_len = len(label_list)
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)
        
        if self.transform is not None and self.is_train:
            img = self.transform(img)
            img.sub_(0.5).div_(0.5)
        
        label_len = np.array(label_len).astype(np.int32)
        label = np.array(label).astype(np.int32)
        
        return img, label, label_len        # 输出图像、文本标签、标签长度, 计算 CTC loss 需要后两者信息


# backbone
class _Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out


class RecBackbone(nn.Module):
    def __init__(self):
        super(RecBackbone, self).__init__()
        
        in_channels = 3
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        
        self.inplanes = 32
        self.layer1 = self._make_layer(32,  3, [2, 2]) # [16, 50]
        self.layer2 = self._make_layer(64,  4, [2, 2]) # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1]) # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1]) # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1]) # [1, 25]
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
            
        layers = []
        layers.append(_Block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_Block(self.inplanes, planes))
            return nn.Sequential(*layers)
        
        
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        cnn_feat = x5.squeeze(2) # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)
        
        return cnn_feat

# decoder
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn=512, nHidden=512, nOut=512):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)    # [T * b, nOut]
        output = output.view(T, b, -1)
        
        return output

# basic
class RecModelBuilder(nn.Module):
    def __init__(self, rec_num_classes, sDim=512):
        super(RecModelBuilder, self).__init__()
        self.rec_num_classes = rec_num_classes
        self.sDim = sDim
        
        self.encoder = RecBackbone()
        self.decoder = nn.Sequential(
        BidirectionalLSTM(sDim, sDim, sDim),
        BidirectionalLSTM(sDim, sDim, rec_num_classes))
        
        self.rec_crit = nn.CTCLoss(zero_infinity=True)
        
    
    def forward(self, inputs):
        x, rec_targets, rec_lengths = inputs
        batch_size = x.shape[0]
        
        encoder_feats = self.encoder(x)   # N, T, C
        encoder_feats = encoder_feats.transpose(0, 1).contiguous() # T, N, C
        rec_pred = self.decoder(encoder_feats)
        
        if self.training:
            rec_pred = rec_pred.log_softmax(dim=2)
            preds_size = torch.IntTensor([rec_pred.size(0)] * batch_size)
            loss_rec = self.rec_crit(rec_pred, rec_targets, preds_size, rec_lengths)
            return loss_rec
        else:
            rec_pred_scores = torch.softmax(rec_pred.transpose(0, 1), dim=2)
            return rec_pred_scores


# train
def rec_train():
    # dataset
    dataset = WMRDataset(rec_args.train_dir, max_len=rec_args.max_len, resize_shape=(rec_args.height, rec_args.width), train=True)
    train_dataloader = data.DataLoader(dataset, batch_size=rec_args.batch_size, num_workers=rec_args.num_workers, shuffle=True, pin_memory=True, drop_last=False)
    
    # model
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model = model.to(device)
    model.train()
    
    # Optimizer
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(param_groups, lr=rec_args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=rec_args.milestones, gamma=0.1)

    os.makedirs(rec_args.save_dir, exist_ok=True)
    # do train
    step = 0
    for epoch in range(rec_args.max_epoch):
        current_lr = optimizer.param_groups[0]['lr']
        
        for i, batch in enumerate(train_dataloader):
            step += 1
            batch = [v.to(device) for v in batch]
            loss = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print
            if step % rec_args.print_interval == 0:
                print('step: {:4d}\tepoch: {:4d}\tloss: {:.4f}'.format(step, epoch, loss.item()))
        scheduler.step()
        
        # save
        # if epoch % rec_args.save_interval == 0:
        save_name = 'checkpoint_' + str(epoch)
        torch.save(model.state_dict(), os.path.join(rec_args.save_dir, save_name))

    torch.save(model.state_dict(), rec_args.saved_model_path)


'''
根据检测结果生成识别模型测试数据
'''
def rec_test_data_gen():
    test_dir = './data/test_imgs'
    det_dir = './temp/det_res'
    word_save_dir = './temp/rec_datasets/test_imgs'

    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        line = lines[0].strip().split(',')
        locs = [float(t) for t in line[:8]]

        # image warp
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2-x1)**2+(y2-y1)**2)**0.5 + ((x4-x3)**2+(y4-y3)**2)**0.5))
        h = int(0.5 * (((x2-x3)**2+(y2-y3)**2)**0.5 + ((x4-x1)**2+(y4-y1)**2)**0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
        word_image = cv2.warpPerspective(image, M, (w, h))
        
        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('det_res_', '')[:-4]+'.jpg'), word_image)


# inference
# 模型输出进行CTC对应解码，去除blank，将连续同字符合并
def rec_decode(rec_prob, labelmap, blank=0):
    raw_str = torch.max(rec_prob, dim=-1)[1].data.cpu().numpy()
    res_str = []
    for b in range(len(raw_str)):
        res_b = []
        prev = -1
        for ch in raw_str[b]:
            if ch == prev or ch == blank:
                prev = ch
                continue
            res_b.append(labelmap[ch])
            prev = ch
        res_str.append(''.join(res_b))
    return res_str
    
    
def rec_load_test_image(image_path, size=(100, 32)):
    img = Image.open(image_path)
    img = img.resize(size, Image.BILINEAR)
    img = torchvision.transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)
    return img.unsqueeze(0)

# 测试
def rec_test():
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model.load_state_dict(torch.load(rec_args.saved_model_path, map_location=device))
    model.eval()
    
    os.makedirs(rec_args.rec_res_dir, exist_ok=True)
    _, _, labelmap = WMRDataset().gen_labelmap()          # labelmap是类别和字符对应的字典
    with torch.no_grad():
        for file in tqdm(os.listdir(rec_args.test_dir)):
            img_path = os.path.join(rec_args.test_dir, file)
            image = rec_load_test_image(img_path)
            batch = [image, None, None]
            pred_prob = model.forward(batch)
            # todo post precess
            rec_str = rec_decode(pred_prob, labelmap)[0]
            # write to file
            with open(os.path.join(rec_args.rec_res_dir, file.replace('.jpg', '.txt')), 'w') as f:
                f.write(rec_str)

# 运行测试代码
rec_test()

'''
识别结果后处理
'''
def final_postProcess():
    SPECIAL_CHARS = {k:v for k, v in zip('ABCDEFGHIJ', '1234567890')}

    test_dir = './data/test_imgs'
    rec_res_dir = './temp/rec_res'
    rec_res_files = os.listdir(rec_res_dir)

    final_res = dict()
    for file in os.listdir(test_dir):
        # res_file = file.replace('.jpg', '.txt')
        res_file = file.replace('.jpg', '.txt')
        if res_file not in rec_res_files:
            final_res[file] = ''
            continue
        
        with open(os.path.join(rec_res_dir, res_file), 'r') as f:
            rec_res = f.readline().strip()
        final_res[file] = ''.join([t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])

    with open('./work/final_res.txt', 'w') as f:
        for key, value in final_res.items():
            f.write(key + '\t' + value + '\n')



if __name__=="__main__":
    # 运行识别训练数据前处理代码 
    PreProcess()

    dataset = WMRDataset(rec_args.train_dir, max_len=5, resize_shape=(rec_args.height, rec_args.width), train=True)
    train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False)
    batch = next(iter(train_dataloader))

    image, label, label_len = batch
    image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    plt.title('image')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
            
    label_digit = label[0].to('cpu').numpy().tolist()
    label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])

    print('label_digit: ', label_digit)
    print('label_str: ', label_str)

    # 运行训练代码
    rec_train()

    # 使用检测模型获取识别测试数据
    rec_test_data_gen()
        
    # 生成最终的测试结果
    final_postProcess()

    # Commented out IPython magic to ensure Python compatibility.
    '''
    最终结果可视化
    '''
    # %matplotlib inline
    import matplotlib
    import matplotlib.pyplot as plt

    with open('./drive/MyDrive/work/final_res.txt', 'r') as f:
        lines = f.readlines()
        
    plt.figure(figsize=(60,60))
    lines = lines[:5]
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            image_name = line.strip()        # 没有识别出来
            word = '###'
        else:
            image_name, word = line.strip().split()
        image = cv2.imread(os.path.join(test_dir, image_name))
        
        plt.subplot(151+i)
        plt.title(word, fontdict={'size': 50})
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
