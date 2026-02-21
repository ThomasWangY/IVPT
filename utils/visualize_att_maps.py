"""Attention map visualisation utilities for IVPT.

Provides :class:`VisualizeAttentionMaps` which renders per-image argmax
part overlays (with centroid labels), extracts cropped patches for each
prototype, and generates hierarchical multi-layer attention figures.
"""

import copy
import math
import os
from pathlib import Path

import cv2
import colorcet as cc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
import numpy as np
import skimage
import torch
from PIL import Image

from utils.data_utils.transform_utils import inverse_normalize_w_resize
from utils.misc_utils import factors

# Define the colors to use for the attention maps
colors = cc.glasbey_category10

def extract_and_resize_patch(image, mask, target_size=(64, 64), min_patches=3):
    H, W, C = image.shape
    h_mask, w_mask = mask.shape
    patch_h, patch_w = H // h_mask, W // w_mask
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Mask does not contain any 1s.")
    
    top_row, bottom_row = rows.min(), rows.max()
    left_col, right_col = cols.min(), cols.max()
    
    patch_size = min_patches
    if bottom_row - top_row + 1 < patch_size:
        diff = patch_size - (bottom_row - top_row + 1)
        top_row = max(0, top_row - diff // 2)
        bottom_row = min(h_mask - 1, bottom_row + (diff + 1) // 2)
    
    if right_col - left_col + 1 < patch_size:
        diff = patch_size - (right_col - left_col + 1)
        left_col = max(0, left_col - diff // 2)
        right_col = min(w_mask - 1, right_col + (diff + 1) // 2)
    
    y1, y2 = top_row * patch_h, (bottom_row + 1) * patch_h
    x1, x2 = left_col * patch_w, (right_col + 1) * patch_w
    
    cropped_image = image[y1:y2, x1:x2]
    resized_image = cv2.resize(cropped_image, target_size)

    return resized_image


class VisualizeAttentionMaps:
    def __init__(self, snapshot_dir="", save_resolution=(256, 256), alpha=0.5, sub_path_test="",
                 dataset_name="", bg_label=0, batch_size=32, num_parts=10, n_pro=[], plot_ims_separately=False,
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
        self.dataset_name = dataset_name
        self.bg_label = bg_label
        self.snapshot_dir = snapshot_dir
        if self.snapshot_dir == "":
            matplotlib.use('Qt5Agg')
        self.resize_unnorm = inverse_normalize_w_resize(resize_resolution=self.save_resolution)
        self.batch_size = batch_size
        self.nrows = factors(self.batch_size)[-1]
        self.ncols = factors(self.batch_size)[-2]
        self.n_pro = n_pro
        self.num_parts = self.n_pro[0]+1
        self.req_colors = colors[:self.num_parts]
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
    def show_maps(self, ims, maps, epoch=0, curr_iter=0, extra_info=""):
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
        self.req_colors = colors
        save_dir = Path(os.path.join(self.snapshot_dir, 'results_vis_' + self.sub_path_test))
        save_dir.mkdir(parents=True, exist_ok=True)

        ims = self.resize_unnorm(ims)
        if ims.shape[0] != self.batch_size:
            self.batch_size = ims.shape[0]
            self.recalculate_nrows_ncols()
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, squeeze=False, figsize=self.figs_size)
        ims = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        map_argmax = torch.nn.functional.interpolate(maps.clone().detach(), size=self.save_resolution,
                                                     mode='bilinear',
                                                     align_corners=True).argmax(dim=1).cpu().numpy()
        # avg_pool = nn.AvgPool2d(kernel_size=5, stride=1)
        for i in range(map_argmax.max()):
            block_arrays = []
            for j in range(len(maps)):
                image_tensor = ims[j]
                (h, w, _) = image_tensor.shape
                if i in map_argmax[j]:
                    mask = (map_argmax[j] == i).astype(int)
                    final_image = extract_and_resize_patch(image_tensor, mask)
                    block_arrays.append(final_image)


            n_col = math.isqrt(len(block_arrays))
            if n_col == 0:
                merged_image = np.zeros((64, 64, 3)).astype(np.uint8)
            else:
                merged_image = np.zeros((64*n_col, 64*n_col, 3)).astype(np.uint8)
                for ii in range(n_col):
                    for jj in range(n_col):
                        merged_image[ii * 64:(ii + 1) * 64, jj * 64:(jj + 1) * 64, :] = block_arrays[ii * n_col + jj]
            block_image = Image.fromarray(merged_image, 'RGB')
            map_dir = Path(os.path.join(self.snapshot_dir, 'results_hie_' + self.sub_path_test, "middle", str(extra_info)), str(i))
            map_dir.mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(map_dir, f'{epoch}_{curr_iter}.png')
            block_image.save(save_path)
        
        for i, ax in enumerate(axs.ravel()):
            self.bg_label = self.n_pro[extra_info]-1
            values, counts = np.unique(map_argmax[i], return_counts=True)
            curr_map = skimage.color.label2rgb(label=map_argmax[i], image=ims[i], colors=self.req_colors,
                                               bg_label=self.bg_label, alpha=self.alpha) # 
            ax.imshow(curr_map) # 
            
            regions = skimage.measure.regionprops(map_argmax[i]+1)
            for region in regions:
                cy, cx = region.centroid
                if region.label-1 != self.bg_label:
                    ax.text(cx, cy, str(region.label-1), color='white', fontsize=6, ha='center', va='center', weight='bold')
            ax.axis('off')

        save_path = os.path.join(save_dir, f'{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.png')
        fig.tight_layout()
        if self.snapshot_dir != "":
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close('all')

        if self.plot_ims_separately:
            fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, squeeze=False, figsize=self.figs_size)
            for i, ax in enumerate(axs.ravel()):
                ax.imshow(ims[i])
                ax.axis('off')
            save_path = os.path.join(save_dir, f'image_{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.jpg')
            fig.tight_layout()
            if self.snapshot_dir != "":
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()
        plt.close('all')

        if self.plot_landmark_amaps:
            if self.batch_size > 1:
                raise ValueError('Not implemented for batch size > 1')
            for i in range(self.num_parts):
                fig, ax = plt.subplots(1, 1, figsize=self.figs_size)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(maps[0, i, ...].detach().cpu().numpy(), cmap='cet_gouldian')
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.axis('off')
                save_path = os.path.join(save_dir,
                                         f'landmark_{i}_{epoch}_{curr_iter}_{self.dataset_name}{extra_info}.png')
                fig.tight_layout()
                if self.snapshot_dir != "":
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                else:
                    plt.show()
                plt.close()

        plt.close('all')


    @torch.no_grad()
    def show_maps_hie(self, ims, maps_loss, epoch=0, curr_iter=0, extra_info="", vis_flag=[]):
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
        self.figs_size = (10*2, (len(vis_flag)+1)*2)
        save_dir = Path(os.path.join(self.snapshot_dir, 'results_vis_' + self.sub_path_test))
        save_dir.mkdir(parents=True, exist_ok=True)

        ims = self.resize_unnorm(ims)

        ims = (ims.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for img_idx in range(len(ims)):
            fig, axs = plt.subplots(nrows=self.figs_size[1]//2, ncols=self.figs_size[0]//2, squeeze=False, figsize=self.figs_size)
            maps_inter = []
            for maps in maps_loss:
                map_argmax = torch.nn.functional.interpolate(maps[img_idx:img_idx+1].clone().detach(), size=self.save_resolution,
                                                            mode='bilinear', align_corners=True).argmax(dim=1).cpu().numpy()
                maps_inter.append(map_argmax)

            for i, ax in enumerate(axs.ravel()):
                x, y = i // (self.figs_size[0]//2), i % (self.figs_size[0]//2) # layer, proto
                if x < len(vis_flag):
                    relation = vis_flag[x]
                    ll = []
                    for r in relation:
                        if r[0] == y:
                            ll.append(r[1])
                else:
                    ll = [y]

                map_ = copy.deepcopy(maps_inter[x][0])
                mask = np.isin(map_, ll)
                map_[~mask] = self.n_pro[x]-1
                curr_map = skimage.color.label2rgb(label=map_, image=ims[img_idx], colors=self.req_colors,
                                                bg_label=self.n_pro[x]-1, alpha=self.alpha) # self.bg_label
                ax.imshow(curr_map)
                
                regions = skimage.measure.regionprops(map_+1)
                for region in regions:
                    cy, cx = region.centroid
                    if region.label != self.n_pro[x]:
                        ax.text(cx, cy, str(region.label-1), color='white', fontsize=6, ha='center', va='center', weight='bold')
                ax.axis('off')

            save_path = os.path.join(save_dir, f'new_{curr_iter}_{self.dataset_name}{img_idx}.png')
            print("done")
            fig.tight_layout()
            if self.snapshot_dir != "":
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()
            plt.close('all')
