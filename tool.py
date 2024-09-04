import re
import math
import heapq
import cv2
import random
import matplotlib
import colorsys
import torch
import torch.nn.functional as F
import warnings
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
from tqdm.auto import tqdm
from skimage import segmentation
from scipy import stats
from skimage import  exposure
from stardist_pkg.matching  import _check_label_array
from skimage.draw import polygon
from shapely.geometry import Point, LineString
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
)

np.set_printoptions(threshold=5, edgeitems=2)
cfg = ConfigParser()
cfg.read(".\params.ini")
n_rays = int(cfg.get("globe", "n_rays"))


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def select_top_elements_and_indices(scores,topn):

    if topn == "all":
        new_scores3 = [sublist[:] for sublist in scores]
        indices_list = [[j for j in range(len(sublist))] for sublist in scores]
        return new_scores3, indices_list

    all_elements = []
    for i, sublist in enumerate(scores):
        for j, value in enumerate(sublist):
            all_elements.append((value, i, j))
    top_elements = heapq.nlargest(topn, all_elements, key=lambda x: x[0])
    new_scores3 = [[] for _ in scores]
    indices_list = [[] for _ in scores]
    for value, i, j in top_elements:
        new_scores3[i].append(value)
        indices_list[i].append(j)
    return new_scores3, indices_list

def add_nested_lists(lst1, lst2):
    return [[elem1 + elem2 for elem1, elem2 in zip(sublist1, sublist2)]
            for sublist1, sublist2 in zip(lst1, lst2)]

def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:

    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    total_slices = num_win * batch_size

    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore

    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True

    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]
        ).to(sw_device)
        seg_prob_out = predictor(window_data, *args, **kwargs)

        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)

            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:

                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]

                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            for idx, original_idx in zip(slice_range, unravel_slice):

                original_idx_zoom = list(original_idx)
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)

                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )


    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)


    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    final_output = final_output[0] if is_tensor_output else final_output

    if isinstance(inputs, MetaTensor):
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]  # type: ignore
    return final_output


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def calculate_endpoint_coordinates(origin, distances):
    x0, y0 = origin
    angle_increment = 2 * math.pi / int(n_rays)
    endpoints = []
    for i, distance in enumerate(distances):
        angle = i * angle_increment
        x = x0 + distance * math.cos(angle)
        y = y0 + distance * math.sin(angle)
        endpoints.append([x, y])
    return endpoints

def calculate_region_mean_in_difference(image, large_contour_points, small_contour_points):
    image = np.array(image)
    large_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    small_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(large_mask, [large_contour_points], 255)
    cv2.fillPoly(small_mask, [small_contour_points], 255)
    new_mask = cv2.subtract(large_mask, small_mask)
    region_pixels = image[np.where(new_mask == 255)]
    average_green_value = np.mean(region_pixels)
    return new_mask, average_green_value

def calculate_radial_distances(centroid, binary_image, n_rays=32):
    assert n_rays > 0, "Number of rays must be positive"

    distances = np.zeros(n_rays, dtype=np.float32)
    st_rays = 2 * np.pi / n_rays

    for k in range(n_rays):
        phi = k * st_rays
        dy, dx = np.sin(phi), np.cos(phi)

        for r in range(1, max(binary_image.shape)):
            i = int(round(centroid[1] + r * dy))
            j = int(round(centroid[0] + r * dx))

            if (i < 0 or i >= binary_image.shape[0] or
                    j < 0 or j >= binary_image.shape[1] or
                    binary_image[i, j] == 0):

                t_corr = 1 - 0.5 / max(abs(dy), abs(dx))
                r -= t_corr
                distances[k] = r
                break

    return distances

def draw_dashed_contour_yellow(image, contour_points):
    image = np.array(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], 255)
    new_mask = mask.copy()
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a = contour_points[:, 0]
    b = contour_points[:, 1]
    a = np.append(a, a[0])
    b = np.append(b, b[0])
    plt.plot(a, b, '--', alpha=1, linewidth=2, zorder=1, color='yellow')
    plt.imshow(image)
    plt.title("Image with Dashed Contour")
    plt.axis('off')

def draw_dashed_contour(image, contour_points, color):
    image = np.array(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], 255)
    new_mask = mask.copy()
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a = contour_points[:, 0]
    b = contour_points[:, 1]
    a = np.append(a, a[0])
    b = np.append(b, b[0])
    plt.plot(a, b, '-', alpha=1, linewidth=2, zorder=1, color=color)
    plt.imshow(image)
    plt.title("Image with Dashed Contour")
    plt.axis('off')
    # plt.show()

def plot_polygon(x,y,color):
    a,b = list(x),list(y)
    a += a[:1]
    b += b[:1]
    plt.plot(a,b,'--', alpha=1, linewidth=2, zorder=1, color=color)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def int_point(all_coords):
    processed_points=[]
    for x, y in all_coords[:int(n_rays)//8]:
        processed_points.append([math.floor(x), math.floor(y)])
    for x, y in all_coords[int(n_rays)//8:int(n_rays)//4]:
        processed_points.append([math.floor(x), math.floor(y)])
    for x, y in all_coords[int(n_rays)//4: int(n_rays)*3//8]:
        processed_points.append([math.ceil(x), math.ceil(y)])
    for x, y in all_coords[int(n_rays)*3//8:int(n_rays)//2]:
        processed_points.append([math.ceil(x), math.ceil(y)])
    for x, y in all_coords[int(n_rays)//2: int(n_rays)*5//8]:
        processed_points.append([math.ceil(x), math.ceil(y)])
    for x, y in all_coords[int(n_rays)*5//8:int(n_rays)*3//4]:
        processed_points.append([math.ceil(x), math.ceil(y)])
    for x, y in all_coords[int(n_rays)*3//4: int(n_rays)*7//8]:
        processed_points.append([math.floor(x), math.floor(y)])
    for x, y in all_coords[int(n_rays)*7//8:]:
        processed_points.append([math.floor(x), math.floor(y)])
    processed_points = np.array(processed_points)
    return processed_points

def boundary_iou(gt, dt, dilation_ratio=0.005):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def mask_to_boundary(mask, dilation_ratio=0.005):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def refine(labels, polys, thr=0.05, progress=False):

    thr = float(thr)
    assert 0 <= thr <= 1, f"required: 0 <= {thr} <= 1"
    if thr == 1:
        thr -= np.finfo(float).epsF
    nms = polys["nms"]

    obj_ind = np.flatnonzero(nms["suppressed"] == -1)

    assert np.allclose(nms["scores"][obj_ind], sorted(nms["scores"][obj_ind])[::-1])
    mask = np.zeros_like(labels)
    for k, i in tqdm(zip(range(len(obj_ind), 0, -1), reversed(obj_ind)), total=len(obj_ind), disable=(not progress)):
        polys_i = nms["coord"][i : i + 1]
        polys_i_suppressed = nms["coord"][nms["suppressed"] == i]
        polys_i = np.concatenate([polys_i, polys_i_suppressed], axis=0)
        ss = tuple(
            slice(max(int(np.floor(start)), 0), min(int(np.ceil(stop)), w))
            for start, stop, w in zip(
                np.min(polys_i, axis=(0, 2)), np.max(polys_i, axis=(0, 2)), labels.shape
            )
        )
        shape_i = tuple(s.stop - s.start for s in ss)
        offset = np.array([s.start for s in ss]).reshape(2, 1)
        n_i = len(polys_i)
        weight_winner = n_i - 1

        polys_i_weights = np.ones(n_i)
        polys_i_weights[0] = weight_winner
        polys_i_weights = polys_i_weights / np.sum(polys_i_weights)
        mask_i = np.zeros(shape_i, float)

        for p, w in zip(polys_i, polys_i_weights):
            ind = polygon(*(p - offset), shape=shape_i)
            mask_i[ind] += w

        mask[ss][mask_i > thr] = k
    return mask

def calculate_boundary_iou_for_predictions(S, G):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))

    biou_list = []
    for iLabelS in listLabelS:
        Si = (S == iLabelS)
        Si_boundary = mask_to_boundary(Si)
        intersectlist = G[Si]

        if intersectlist.any():

            indexGi = stats.mode(intersectlist).mode[0]
            Gi = (G == indexGi)
        else:
            continue

        biou = boundary_iou(Si, Gi)
        iou = np.sum((Gi & Si)) / np.sum((Gi | Si))
        biou_list.append(min(biou,iou))

    return np.mean(biou_list)