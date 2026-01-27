import clip
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la
from Utilities.CUDA_Check import GPUorCPU

DEVICE = GPUorCPU.DEVICE

logabs = lambda x: torch.log(torch.abs(x))


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()

    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Block(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        if not w_s.flags['WRITEABLE']:
            w_s.setflags(write=1)

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

        self.actnorm = ActNorm(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=True)

    def forward(self, input):
        input = self.actnorm(input)
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        out = self.coupling(out)
        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        output = self.coupling.reverse(output)
        weight = self.calc_weight()
        out = F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
        out = self.actnorm.reverse(out)
        return out


def symmetric_wct(fea_a, fea_b):

    batch_size, channels, height, width = fea_a.size()

    fea_a_2d = fea_a.view(channels, -1)
    fea_b_2d = fea_b.view(channels, -1)

    a_mean = torch.mean(fea_a_2d, 1)  # channels x 1
    b_mean = torch.mean(fea_b_2d, 1)  # channels x 1

    fea_a_centered = fea_a_2d - a_mean.unsqueeze(1).expand_as(fea_a_2d)
    fea_b_centered = fea_b_2d - b_mean.unsqueeze(1).expand_as(fea_b_2d)

    aConv = torch.mm(fea_a_centered, fea_a_centered.t()).div(fea_a_centered.size(1) - 1)
    bConv = torch.mm(fea_b_centered, fea_b_centered.t()).div(fea_b_centered.size(1) - 1)

    a_u, a_e, a_v = torch.svd(aConv, some=False)
    b_u, b_e, b_v = torch.svd(bConv, some=False)

    k_a = channels
    for i in range(channels - 1, -1, -1):
        if a_e[i] >= 0.00001:
            k_a = i + 1
            break

    k_b = channels
    for i in range(channels - 1, -1, -1):
        if b_e[i] >= 0.00001:
            k_b = i + 1
            break

    a_d = (a_e[0:k_a]).pow(-0.5)
    step1_a = torch.mm(a_v[:, 0:k_a], torch.diag(a_d))
    step2_a = torch.mm(step1_a, a_v[:, 0:k_a].t())
    whiten_aF = torch.mm(step2_a, fea_a_centered)

    b_d = (b_e[0:k_b]).pow(-0.5)
    step1_b = torch.mm(b_v[:, 0:k_b], torch.diag(b_d))
    step2_b = torch.mm(step1_b, b_v[:, 0:k_b].t())
    whiten_bF = torch.mm(step2_b, fea_b_centered)

    fused_whiten = (whiten_aF + whiten_bF) / 2.0

    commonConv = (aConv + bConv) / 2.0
    common_u, common_e, common_v = torch.svd(commonConv, some=False)

    common_d = (common_e[0:k_a]).pow(0.5)
    step1_common = torch.mm(common_v[:, 0:k_a], torch.diag(common_d))
    step2_common = torch.mm(step1_common, common_v[:, 0:k_a].t())
    colored_fused = torch.mm(step2_common, fused_whiten)

    fused_mean = (a_mean + b_mean) / 2.0
    fused_feat_2d = colored_fused + fused_mean.unsqueeze(1).expand_as(colored_fused)

    fused_feat = fused_feat_2d.view(batch_size, channels, height, width)

    return fused_feat


class Jiangwei(nn.Module):
    def __init__(self, input_dim=512, output_dim=64):
        super(Jiangwei, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)

        nn.init.orthogonal_(self.fc.weight)
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, text):

        input_dtype = text.dtype

        text = text.to(self.fc.weight.dtype)
        text_feature = self.fc(text)

        text_feature = text_feature.to(input_dtype)

        return text_feature


class Network(nn.Module):
    def __init__(self, clip_model, resnet_model_18, resnet_model_50, resnet_model_101, in_channel=3, depth=3):
        super().__init__()

        self.clip_model = clip_model
        self.resnet_model_18 = resnet_model_18
        self.resnet_model_50 = resnet_model_50
        self.resnet_model_101 = resnet_model_101

        self.resnet_conv_18 = resnet_model_18.conv1
        self.resnet_conv_18.stride = 1
        self.resnet_conv_18.padding = (0, 0)

        self.resnet_conv_50 = resnet_model_50.conv1
        self.resnet_conv_50.stride = 1
        self.resnet_conv_50.padding = (0, 0)

        self.resnet_conv_101 = resnet_model_101.conv1
        self.resnet_conv_101.stride = 1
        self.resnet_conv_101.padding = (0, 0)

        self.clip_model.eval()
        self.resnet_model_18.eval()
        self.resnet_model_50.eval()
        self.resnet_model_101.eval()

        self.img_text_fusion = FeatureWiseAffine(in_channels=512, out_channels=512)
        self.jiangwei = Jiangwei()
        self.forward_conv = nn.Conv2d(12, 512, 1, 1, 0)
        self.reverse_conv = nn.Conv2d(1024, 12, 1, 1, 0)

        self.DWT = DWT()
        self.IWT = IWT()

        self.fake_layer0 = nn.Conv2d(12, 64, 3, 1, 1)
        self.fake_layer1 = nn.Conv2d(12, 64, 3, 1, 1)
        self.fake_layer2 = nn.Conv2d(12, 64, 3, 1, 1)

        self.blocks = nn.ModuleList()
        n_channel = in_channel * 4
        for i in range(depth):
            self.blocks.append(Block(n_channel))

    def forward(self, input, feature=None, text=None, forward=True):

        if forward:
            return self._forward(input)
        else:
            return self._reverse(input, feature, text)

    def _forward(self, input):
        size = input.shape[0]
        real_img_fea_list = []
        fake_img_fea_list = []
        z = input
        z = self.DWT(z)
        for i, block in enumerate(self.blocks):
            z = block(z)
            real_img_fea_list.append(self.get_real_img_feature(z, i))

            fake_layer_name = f"fake_layer{i}"
            if hasattr(self, fake_layer_name):
                fake_layer = getattr(self, fake_layer_name)
                fake_img_fea_list.append(fake_layer(z))
            else:
                raise ValueError(f"Layer {fake_layer_name} does not exist")

        real_classification, fake_classification, texts1, texts2, texts3 = self.intelligent_agent(real_img_fea_list, fake_img_fea_list, batchsize=input.shape[0])
        text = clip.tokenize(self.generate_description(fake_classification, texts1, texts2, texts3)).to(DEVICE)
        text_features = self.get_text_feature(text.expand(size, -1)).to(input.dtype)
        return z, real_classification, fake_classification, text_features

    def _reverse(self, input, feature, text_feature):
        fea_a = input
        fea_b = feature
        z = symmetric_wct(fea_a, fea_b)
        for i, block in enumerate(self.blocks[::-1]):
            z = self.forward_conv(z)
            z_text = self.img_text_fusion(z, text_feature)
            z = self.reverse_conv(torch.cat([z_text, z], dim=1))
            z = block.reverse(z)
        z = self.IWT(z)
        return z

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.clip_model.encode_text(text)
        return text_feature

    @torch.no_grad()
    def get_real_img_feature(self, img, num):
        img = self.IWT(img)
        img = self.tensor_padding(tensors=img, padding=(3, 3, 3, 3), mode='replicate')
        if num == 0:
            img_feature = self.resnet_conv_18(img)
        elif num == 1:
            img_feature = self.resnet_conv_50(img)
        else:
            img_feature = self.resnet_conv_101(img)
        return img_feature

    @torch.no_grad()
    def generate_description(self, fake_classification, category_1_texts, category_2_texts, category_3_texts):
        result = []

        if len(fake_classification) > 0 and 1.0 in fake_classification[0]:
            index = fake_classification[0].index(1.0)
            result.append(category_1_texts[index])
        if len(fake_classification) > 1 and 1.0 in fake_classification[1]:
            index = fake_classification[1].index(1.0)
            result.append(category_2_texts[index])
        if len(fake_classification) > 2 and 1.0 in fake_classification[2]:
            index = fake_classification[2].index(1.0)
            result.append(category_3_texts[index])

        return " ".join(result[:-1]) + " " + result[-1]

    @torch.no_grad()
    def intelligent_agent(self, real_img_fea, fake_img_fea, batchsize):

        real_similarities = []
        fake_similarities = []

        category_1_texts = [
            "The source image has noise interference with scattered color/black and white dots.",
            "The source image has low-light interference causing blurred edges and unclear texture.",
            "The source image is overexposed with decreased color saturation and dull colors.",
            "The source image has low contrast making foreground and background depth hard to distinguish",
            "The source image is affected by unknown interference."
        ]
        category_2_texts = [
            "The defocusing diffusion is severe needing detailed processing at clear/blurry object intersections.",
            "The defocusing diffusion is slight but attention should be paid to reduced contrast in the generated image."
        ]
        category_3_texts = [
            "The model generates images with severe grid shapes.",
            "The network extracts good features but attention should be paid to contrast loss during reverse recovery."
        ]
        category_texts = [category_1_texts, category_2_texts, category_3_texts]

        i = 0
        for category in category_texts:
            real_similaritie = []
            fake_similaritie = []
            category = clip.tokenize(category).to(DEVICE)
            for text in category:
                text_feature = self.get_text_feature(text.expand(batchsize, -1)).to(DEVICE)
                fea_d = self.jiangwei(text_feature)
                real_similaritie.append(torch.sum(F.cosine_similarity(torch.mean(real_img_fea[i], dim=(2,3)), fea_d)))
                fake_similaritie.append(torch.sum(F.cosine_similarity(torch.mean(fake_img_fea[i], dim=(2,3)), fea_d)))
            i += 1
            real_similarities.append(real_similaritie)
            fake_similarities.append(fake_similaritie)

        for sublist in real_similarities:
            max_value = max(sublist)
            for i in range(len(sublist)):
                sublist[i] = 1.0 if sublist[i] == max_value else 0.0
        for sublist in fake_similarities:
            max_value = max(sublist)
            for i in range(len(sublist)):
                sublist[i] = 1.0 if sublist[i] == max_value else 0.0

        return real_similarities, fake_similarities, category_1_texts, category_2_texts, category_3_texts,

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensor = F.pad(tensors, padding, mode=mode, value=value)
        return out_tensor

    def all_ones_matrix(self, shape_h, shape_w):
        matrix = torch.ones(shape_h, shape_w).to(DEVICE)
        return matrix

if __name__ == '__main__':
    text_line = []
    for i in range(1):
        text_line.append("This is unknown to the multi-focus image fusion task.")
    text = clip.tokenize(text_line).to(DEVICE)
    test_tensor_A = torch.randn(1, 3, 10, 10).to(DEVICE)
    test_tensor_B = torch.randn(1, 3, 10, 10).to(DEVICE)

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    resnet_18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_101 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    model = Network(clip_model, resnet_18, resnet_50, resnet_101).to(DEVICE)
    for module in [model.clip_model, model.resnet_model_18, model.resnet_model_50, model.resnet_model_101]:
        for param in module.parameters():
            param.requires_grad = False
    exclude_modules = ['clip_model', 'resnet_model_18', 'resnet_model_50', 'resnet_model_101']
    num_params = 0
    for name, p in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in exclude_modules:
            num_params += p.numel()
    print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    # Forward
    Pre_A, real_similarities_A, fake_similarities_A, text_A = model(test_tensor_A, text, forward=True)
    Pre_B, real_similarities_B, fake_similarities_B, text_B = model(test_tensor_B, text, forward=True)
    # Reverse
    Pre = model(Pre_A, Pre_B, text_A, forward=False)


    print(Pre.shape)
