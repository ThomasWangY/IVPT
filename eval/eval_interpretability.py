import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage
from pathlib import Path
from utils.data_utils.transform_utils import inverse_normalize_w_resize
from utils.misc_utils import factors
import matplotlib.patches as patches
import colorcet as cc

# colors = cc.glasbey_category10
import matplotlib.cm as cm

from eval.datasets import Cub2011Eval
from eval.preprocess import mean, std
from eval.local_parts import id_to_path, id_to_part_loc, id_to_bbox, part_num, in_bbox


def perturb_img(norm_img, std=0.2, eps=0.25):
    noise = torch.zeros(norm_img.shape).normal_(mean=0, std=std).cuda()
    noise = torch.clip(noise, min=-eps, max=eps)    # Constrain the maximum absolute value, ensuring that the noise is imperceptible by humans
    perturb_img = norm_img + noise
    return perturb_img


class VisualizeAttentionMaps:
    def __init__(self, snapshot_dir="", save_resolution=(224, 224), alpha=0.5, sub_path_test="",
                 bg_label=10, batch_size=6, num_parts=11, plot_ims_separately=False,
                 plot_landmark_amaps=False):
        """
        Plot attention maps and optionally landmark centroids on images.
        :param snapshot_dir: Directory to save the visualization results
        :param save_resolution: Size of the images to save
        :param alpha: The transparency of the attention maps
        :param sub_path_test: The sub-path of the test dataset
        :param dataset_name: The name of the dataset
        :param bg_label: The background label index in the attention maps
        :param batch_size: The batch size
        :param num_parts: The number of parts in the attention maps
        :param plot_ims_separately: Whether to plot the images separately
        :param plot_landmark_amaps: Whether to plot the landmark attention maps
        """
        self.save_resolution = save_resolution
        self.alpha = alpha
        self.sub_path_test = sub_path_test
        self.dataset_name = "cub"
        self.bg_label = bg_label
        self.snapshot_dir = snapshot_dir
        if self.snapshot_dir == "":
            matplotlib.use('Qt5Agg')
        self.resize_unnorm = inverse_normalize_w_resize(resize_resolution=self.save_resolution)
        self.batch_size = batch_size
        self.nrows = factors(self.batch_size)[-1]
        self.ncols = factors(self.batch_size)[-2]
        self.num_parts = num_parts
        self.req_colors = plt.cm.Reds(np.linspace(0.2, 1, num_parts)) # colors[:num_parts]
        self.plot_ims_separately = plot_ims_separately
        self.plot_landmark_amaps = plot_landmark_amaps
        if self.nrows == 1 and self.ncols == 1:
            self.figs_size = (10, 10)
        else:
            self.figs_size = (self.ncols * 2, self.nrows * 2)

    def recalculate_nrows_ncols(self):
        self.nrows = factors(self.batch_size)[-1]
        self.ncols = factors(self.batch_size)[-2]
        if self.nrows == 1 and self.ncols == 1:
            self.figs_size = (10, 10)
        else:
            self.figs_size = (self.ncols * 2, self.nrows * 2)

    @torch.no_grad()
    def show_all_maps(self, ims_, maps_, scores_, test_label):
        """
        Plot images, attention maps and landmark centroids.
        Parameters
        ----------
        ims: Tensor, [batch_size, 3, width_im, height_im]
            Input images on which to show the attention maps
        maps: Tensor, [batch_size, number of parts + 1, width_map, height_map]
            The attention maps to display
        epoch: int
            The epoch number
        curr_iter: int
            The current iteration number
        extra_info: str
            Any extra information to add to the file name
        """
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))

        for idx in range(len(ims_)):
            xx, yy = idx // 2, idx % 2
            axs = ax[xx, yy]
            ims, maps, scores = ims_[idx], maps_[idx], scores_[idx][test_label]
            sorted_indices = np.argsort(scores)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(len(scores))

            color_rank = [self.req_colors[i] for i in ranks]

            ims = torch.from_numpy(ims).float() / 255.0
            ims = ims.unsqueeze(0).permute(0, 3, 1, 2)
            
            ims = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            maps = torch.from_numpy(maps).unsqueeze(0)
            map_argmax = torch.nn.functional.interpolate(maps.clone().detach(), size=self.save_resolution,
                                                        mode='bilinear',
                                                        align_corners=True).argmax(dim=1).cpu().numpy()

            curr_map = skimage.color.label2rgb(label=map_argmax[0], image=ims[0], colors=color_rank,
                                            bg_label=self.bg_label, alpha=0.6)
            axs.imshow(curr_map)
            
            regions = skimage.measure.regionprops(map_argmax)

            for region in regions:
                cy, cx = region.centroid[1], region.centroid[2]
                axs.text(cx, cy, str(region.label), color='white', fontsize=6, ha='center', va='center', weight='bold')
            axs.axis('off')


        save_dir = Path(os.path.join(self.snapshot_dir, 'results_all' + self.sub_path_test))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.dataset_name}_c{test_label}.png')
        fig.tight_layout()
        if self.snapshot_dir != "":
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close('all')


    @torch.no_grad()
    def show_maps(self, ims_, part_label_, region_pred_, proto_idx, test_label, img_idx_, heatmap_):
        """
        Plot images, attention maps and landmark centroids.
        Parameters
        ----------
        ims: Tensor, [batch_size, 3, width_im, height_im]
            Input images on which to show the attention maps
        maps: Tensor, [batch_size, number of parts + 1, width_map, height_map]
            The attention maps to display
        epoch: int
            The epoch number
        curr_iter: int
            The current iteration number
        extra_info: str
            Any extra information to add to the file name
        """
        
        # 开始绘制
        fig, ax = plt.subplots(3, 4, figsize=(20, 15))

        for idx in range(len(ims_)):
            cmap = cm.get_cmap('tab20', 15)
            xx, yy = idx // 2, 2*(idx % 2)
            ims, part_label, region_pred, img_idx, heatmap = ims_[idx], part_label_[idx], region_pred_[idx], img_idx_[idx], heatmap_[idx]
            ims = torch.from_numpy(ims).float() / 255.0
            img = (ims.unsqueeze(0).cpu().numpy() * 255).astype(np.uint8).squeeze(0)
            # 显示图像
            ax[xx, 0+yy].imshow(img)

            # 绘制边界框 (region_pred)
            y_min, y_max, x_min, x_max = region_pred
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, 223)
            y_max = min(y_max, 223)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax[xx, 0+yy].add_patch(rect)

            # 绘制 part_label 中的点，不同类别用不同的颜色
            for label, x, y in part_label:
                color = cmap(label - 1)  # 根据类别选择颜色，label 从 1 到 15
                ax[xx, 0+yy].scatter(x, y, color=color, s=50)
                ax[xx, 0+yy].text(x, y, f'{label}', color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='none', pad=1))

            # 隐藏坐标轴
            ax[xx, 0+yy].axis('off')

            cmap = cm.get_cmap('jet')
            heatmap_color = cmap(heatmap)[:, :, :3]
            # 在第二个子图中叠加图像和热力图
            ax[xx, 1+yy].imshow(img)  # 先显示背景图像
            ax[xx, 1+yy].imshow(heatmap_color, alpha=0.6)  # 叠加热力图，透明度为 0.6
            ax[xx, 1+yy].axis('off')  # 隐藏坐标轴


        save_dir = Path(os.path.join(self.snapshot_dir, 'results_single_proto' + self.sub_path_test))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.dataset_name}_p{proto_idx}_c{test_label}.png')
        fig.tight_layout()
        if self.snapshot_dir != "":
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close('all')


@torch.no_grad()
def get_corresponding_object_parts(ppnet, args, half_size, visual=False, vis_att_maps=None, use_noise=False):
    ppnet.eval()
    ppnet_without_ddp = ppnet.module if hasattr(ppnet, 'module') else ppnet
    img_size = args.image_size # ppnet_without_ddp.img_size
    proto_per_class = int(args.n_pro.split(",")[-3]) - 1 # NPRO
    # proto_per_class = int(args.n_pro) - 1

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval(args.data_path, train=False, transform=transform)    # CUB test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=10, pin_memory=True, drop_last=False, shuffle=False)
    num_classes = args.nb_classes

    # Infer on the whole test dataset
    all_proto_acts, all_targets, all_img_ids, all_scores = [], [], [], []
    for j, (data, targets, img_ids) in enumerate(test_loader):
        data = data.cuda()
        targets = targets.cuda()
        if use_noise:   # This is used when calculating stability score
            data = perturb_img(data)

        _, proto_acts, scores, _, _ = ppnet_without_ddp(data) # NPRO

        # Select the prototypes belonging to the ground-truth class of each image
        fea_size = proto_acts.shape[-1]
        proto_indices = (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).cuda()   # The indexes of prototypes belonging to the ground-truth class of each image
        proto_indices = proto_indices[:, :, None, None].repeat(1, 1, fea_size, fea_size)
        
        all_proto_acts.append(proto_acts.cpu().detach())
        all_targets.append(targets.cpu())
        all_img_ids.append(img_ids)
        all_scores.append(scores.cpu().detach())
    all_proto_acts = torch.cat(all_proto_acts, dim=0).numpy()   # The activation maps of all test images
    all_targets = torch.cat(all_targets, dim=0).numpy() # The categories of all test images
    all_img_ids = torch.cat(all_img_ids, dim=0).numpy() # The image ids of all test images
    all_scores = torch.cat(all_scores, dim=0).numpy()
    # print(all_scores.shape, all_scores.mean(axis=0))

    # Enumerate all the classes, thus enumerate all the prototypes
    all_proto_to_part, all_proto_part_mask = [], []
    '''
    The length of `all_proto_to_part` is 2000, each element indicates the corresponding object parts of a prototype on the images.
    The length of `all_proto_part_mask` is 2000, each element indicates the existing (non-masked) object parts on the images of a prototype.
    '''
    img_size = 224 # TODO
    for test_image_label in range(num_classes):
        arr_ids = np.nonzero(all_targets == test_image_label)[0]
        class_proto_acts = all_proto_acts[arr_ids] # Get the activation maps of all the images of current class
        img_ids = all_img_ids[arr_ids]  # Get the image ids of all the images of current class
        scores = all_scores[arr_ids]

        # Get part annotations on all the images of current class
        class_part_labels, class_part_masks = [], []
        '''
        `class_part_labels` save the part labels of images in this class.
        `class_part_masks` save the part masks of images in this class.
        '''
        imgs_,  maps_, scores_ = [], [], []
        for img_idx, img_id in enumerate(img_ids):
            test_image_path = os.path.join(args.data_path, 'test_cropped', id_to_path[img_id][0], id_to_path[img_id][1])
            # Read the image
            original_img = cv2.imread(test_image_path)
            original_img = cv2.resize(original_img, (img_size, img_size))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Get part annotations
            part_labels, part_mask = [], np.zeros(part_num,)
            bbox = id_to_bbox[img_id]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            part_locs = id_to_part_loc[img_id]
            for part_loc in part_locs:
                part_id = part_loc[0] - 1   # The id of current object part (begin from 0)
                part_mask[part_id] = 1  # The current object part exists in current image
                loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
                ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (bbox_y2 - bbox_y1) # Fit the bounding boxes' coordinates to the cropped images
                re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
                part_labels.append([part_id, re_loc_x, re_loc_y])
                # print([part_id, re_loc_x, re_loc_y]);exit(0)
            class_part_labels.append(part_labels)
            class_part_masks.append(part_mask)

            if test_image_label % 20 == 0 and img_idx <= 10 and img_idx % 2 == 0 and visual:
                imgs_.append(original_img)
                maps_.append(class_proto_acts[img_idx])
                scores_.append(scores[img_idx])

        if test_image_label % 20 == 0 and visual:
            vis_att_maps.show_all_maps(imgs_, maps_, scores_, test_image_label)

        # Enumerate the prototypes of current class
        for proto_idx in range(proto_per_class):
            img_num = len(img_ids)
            proto_to_part = np.zeros((img_num, part_num))   # Element = 1 -> the prototype corresponds to this object part on this image, element = 0 otherwise
            original_img_, part_labels_, region_pred_, img_idx_, heatmap_ = [], [], [], [], []
            for img_idx in range(img_num):
                part_labels = class_part_labels[img_idx]    # Get the part labels of current image
                activation_map = class_proto_acts[img_idx, proto_idx]   # Get the activation map of current prototype on current image
                upsampled_activation_map = cv2.resize(activation_map, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                # print(upsampled_activation_map.shape)

                max_indice = np.where(upsampled_activation_map==upsampled_activation_map.max())
                # print(activation_map, activation_map.shape, activation_map.max(), upsampled_activation_map.shape, upsampled_activation_map.max(), max_indice)
                max_indice = (max_indice[0][0], max_indice[1][0])
                region_pred = (max(0, max_indice[0] - half_size), min(img_size, max_indice[0] + half_size), max(0, max_indice[1] - half_size), min(img_size, max_indice[1] + half_size)) # Get the corresponding region of current prototype, (y1, y2, x1, x2)
                
                # Get the corresponding object parts of current prototype
                for part_label in part_labels:
                    part_id, loc_x_gt, loc_y_gt = part_label[0], part_label[1], part_label[2]
                    # print((loc_y_gt, loc_x_gt), region_pred)
                    if in_bbox((loc_y_gt, loc_x_gt), region_pred):
                        proto_to_part[img_idx, part_id] = 1

                test_image_path = os.path.join(args.data_path, 'test_cropped', id_to_path[img_ids[img_idx]][0], id_to_path[img_ids[img_idx]][1])
                # Read the image
                original_img = cv2.imread(test_image_path)
                original_img = cv2.resize(original_img, (img_size, img_size))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                if test_image_label % 20 == 0 and img_idx <= 10 and img_idx % 2 == 0 and visual:
                    original_img_.append(original_img)
                    part_labels_.append(part_labels)
                    region_pred_.append(region_pred)
                    img_idx_.append(img_idx)
                    heatmap_.append(upsampled_activation_map)
                    
            if test_image_label % 20 == 0 and visual:
                vis_att_maps.show_maps(original_img_, part_labels_, region_pred_, proto_idx, test_image_label, img_idx_, heatmap_)

            proto_to_part = np.stack(proto_to_part, axis=0)
            class_part_masks = np.stack(class_part_masks, axis=0)
            all_proto_to_part.append(proto_to_part)
            all_proto_part_mask.append(class_part_masks)

    return all_proto_to_part, all_proto_part_mask


def evaluate_consistency(ppnet, args, half_size=36, part_thresh=0.8):
    num_parts = int(args.n_pro.split(',')[-3]) - 1 # NPRO
    # num_parts = int(args.n_pro) - 1

    vis_att_maps = VisualizeAttentionMaps(num_parts=num_parts, snapshot_dir=args.output_dir)


    all_proto_to_part, all_proto_part_mask = get_corresponding_object_parts(ppnet, args, half_size, visual=True, vis_att_maps=vis_att_maps)
    all_proto_consis = []
    '''
    The length of `all_proto_consis` is 2000, each element indicates the consistency of a prototype, 1 -> consistent, 0 -> non-consistent.
    '''
    # Enumerate all the prototypes to calculate consistency score
    for proto_idx in range(len(all_proto_to_part)):
        proto_to_part = all_proto_to_part[proto_idx]
        proto_part_mask = all_proto_part_mask[proto_idx]
        assert ((1. - proto_part_mask) * proto_to_part).sum() == 0  # Assert that the prototype does not correspond to an object part that cannot be visualized (not in the part annotations)
        proto_to_part_sum = proto_to_part.sum(axis=0)
        proto_part_mask_sum = proto_part_mask.sum(axis=0)
        proto_part_mask_sum = np.where(proto_part_mask_sum == 0, proto_part_mask_sum + 1, proto_part_mask_sum)  # Eliminate the 0 elements in all_part_mask_sum~(prevent 0 from being denominator), it doesn't affect the evaluation result
        mean_part_float = proto_to_part_sum / proto_part_mask_sum
        mean_part = (mean_part_float >= part_thresh).astype(np.int32)   # The prototope is determined to be non-consistent if  no element in the averaged corresponding object parts exceeds `part_thresh`

        if mean_part.sum() == 0:
            all_proto_consis.append(0)
        else:
            all_proto_consis.append(1)

    all_proto_consis = np.array(all_proto_consis).reshape(200, num_parts)
    consistency_score = all_proto_consis.mean() * 100

    return consistency_score


def evaluate_stability(ppnet, args, half_size=36):
    all_proto_to_part, _ = get_corresponding_object_parts(ppnet, args, half_size, use_noise=False)
    all_proto_to_part_noise, _ = get_corresponding_object_parts(ppnet, args, half_size, use_noise=True)

    all_proto_stability = []
    for proto_idx in range(len(all_proto_to_part)):
        proto_to_part = all_proto_to_part[proto_idx]
        proto_to_part_noise = all_proto_to_part_noise[proto_idx]

        is_equal = (np.abs(proto_to_part - proto_to_part_noise).sum(axis=-1) == 0).astype(np.float32)   # Determine whether the elements in `proto_to_part` and `proto_to_part_perturb` are equal
        proto_stability = is_equal.mean()   # The ratio of elements that keep unchanged under perturbation
        all_proto_stability.append(proto_stability)

    all_proto_stability = np.array(all_proto_stability)
    stability_score = all_proto_stability.mean() * 100

    return stability_score