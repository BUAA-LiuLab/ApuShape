import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageEnhance
from scipy.spatial import distance
from tifffile import imread

from model.flexible_unet_convnext import FlexibleUNet_star
from stardist_pkg import non_maximum_suppression, polygons_to_label
from stardist.geometry.geom2d import dist_to_coord
from tool import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=np.inf)

cfg = ConfigParser()
cfg.read(".\params.ini")

folder_path_images = cfg.get('refine', "folder_path_images")
folder_path_masks = cfg.get('refine', "folder_path_masks")
folder_path_pred = cfg.get('refine', "folder_path_pred")
n_rays = int(cfg.get('globe', "n_rays"))
er1 = int(cfg.get('refine', "er1"))
er2 = int(cfg.get('refine', "er2"))
fl1 = int(cfg.get('refine', "fl1"))
fl2 = int(cfg.get('refine', "fl2"))


BAP_refined = []
BAP_initial = []

sorted_images_filenames = sorted(
    [f for f in os.listdir(folder_path_images) if f.endswith(".tif")],
    key=extract_number
)
sorted_masks_filenames = sorted(
    [f for f in os.listdir(folder_path_masks) if f.endswith(".tif")],
    key=extract_number
)
sorted_pred_filenames = sorted(
    [f for f in os.listdir(folder_path_pred) if f.endswith(".tif")],
    key=extract_number
)


batch_coords = []
batch_distances = []
batch_points = []

for num, (image_filename, mask_filename, pred_filename) in enumerate(tqdm(zip(sorted_images_filenames, sorted_masks_filenames, sorted_pred_filenames), total=len(sorted_images_filenames))):

    image_path = os.path.join(folder_path_images, image_filename)
    mask_path = os.path.join(folder_path_masks, mask_filename)
    pred_path = os.path.join(folder_path_pred, pred_filename)

    img = imread(image_path)
    img_mask = imread(mask_path)
    pred = imread(pred_path)
    h, w = img.shape[:2]

    unique_values = np.unique(pred)
    unique_values = unique_values[unique_values != 0]

    img_distances = []
    img_points = []
    img_coords= []

    for value in unique_values:
        binary_mask = (pred == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                point = [cX, cY]
                dist = calculate_radial_distances(point, binary_mask, n_rays)
                coords = calculate_endpoint_coordinates(point, dist)
                img_coords.append(coords)
                img_distances.append(dist)
                img_points.append(point)

    batch_coords.append(img_coords)
    batch_distances.append(img_distances)
    batch_points.append(img_points)


for num, (images_filename, img_distances, img_points, img_coords) in enumerate(tqdm(zip(sorted_images_filenames, batch_distances,batch_points, batch_coords), total=len(sorted_images_filenames))):
    if img_distances is not None:

        refined_coords = []

        image_path = os.path.join(folder_path_images, images_filename)
        mask_path = os.path.join(folder_path_masks, images_filename)

        img = imread(image_path)
        img_mask = imread(mask_path)

        for index, (coords, distances, point) in enumerate(zip(img_coords, img_distances, img_points)):
            coords = int_point(coords)
            distances_copy = distances.copy()

            all_inter_annular_values = []
            all_outer_annular_values = []

            for dist_vary in range(1, er1):
                adjusted_distances = [dist - dist_vary for dist in distances_copy]
                end_coords = calculate_endpoint_coordinates(point, adjusted_distances)
                _, inter_annular_value = calculate_region_mean_in_difference(img, coords, int_point(end_coords))
                all_inter_annular_values.append(inter_annular_value)
            max_inter_annular_value = max(all_inter_annular_values)

            saved_value = [0] * 1000 + [distances_copy, distances_copy]
            for _ in range(1, er2):
                for i in range(n_rays):
                    adjusted_distances = saved_value[-1].copy()
                    adjusted_distances[i] -= 1
                    end_coords1 = calculate_endpoint_coordinates(point, adjusted_distances)
                    end_coords2 = calculate_endpoint_coordinates(point, saved_value[-2])
                    _, inter_mask_value = calculate_region_mean_in_difference(img, int_point(end_coords2), int_point(end_coords1))
                    if inter_mask_value < max_inter_annular_value:
                        saved_value.append(adjusted_distances)

            distances_copy = saved_value[-1].copy()
            for _ in range(1, fl1):
                for i in range(n_rays):
                    adjusted_distances0 = [dist + dist_vary for dist in distances_copy]
                    end_coords1 = int_point(calculate_endpoint_coordinates(point, adjusted_distances0))
                    end_coords2 = calculate_endpoint_coordinates(point, distances_copy)
                    new, outer_annular_value = calculate_region_mean_in_difference(img, end_coords1, int_point(end_coords2))
                    all_outer_annular_values.append(outer_annular_value)
            min_outer_annular_value = min(all_outer_annular_values)

            saved_value = [0] * 1000 + [adjusted_distances, adjusted_distances]
            skip_indices = set()
            for _ in range(1, fl2):
                for i in range(n_rays):
                    if i not in skip_indices:
                        distances_adjusted = saved_value[-1].copy()
                        distances_adjusted[i] += 1
                        end_coords1 = int_point(calculate_endpoint_coordinates(point, distances_adjusted))
                        end_coords2 = calculate_endpoint_coordinates(point, saved_value[-2])
                        _, outer_annular_value = calculate_region_mean_in_difference(img, end_coords1, int_point(end_coords2))
                        if outer_annular_value > min_outer_annular_value:
                            saved_value.append(distances_adjusted)
                        else:
                            skip_indices.add(i)

            refined_coord = calculate_endpoint_coordinates(point, saved_value[-1])
            draw_dashed_contour(img, int_point(coords), "yellow")
            draw_dashed_contour(img, int_point(refined_coord), "red")
            refined_coords.append(int_point(refined_coord))
        # plt.show()

        mask_pred = np.zeros((h, w), dtype=np.uint8)
        mask_initial = np.zeros((h, w), dtype=np.uint8)

        for pixel_value, contour in enumerate(refined_coords, start=1):
            cv2.drawContours(mask_pred, [contour], -1, pixel_value, thickness=cv2.FILLED)

        for pixel_value, contour in enumerate(img_coords, start=1):
            cv2.drawContours(mask_initial, [int_point(contour)], -1, pixel_value, thickness=cv2.FILLED)

        groundtruth = remap_label(img_mask)

        mask_pred = remap_label(mask_pred)
        mask_raw = remap_label(mask_initial)

        bap_refined = calculate_boundary_iou_for_predictions(mask_pred, groundtruth)
        BAP_refined.append(bap_refined)

        bap_refined = calculate_boundary_iou_for_predictions(mask_raw, groundtruth)
        BAP_initial.append(bap_refined)

average_BAP_refined = np.mean(BAP_refined)
average_BAP_initial = np.mean(BAP_initial)

print(f"Average BAP_refined : {average_BAP_refined}")
print(f"Average BAP_initial: {average_BAP_initial}")




