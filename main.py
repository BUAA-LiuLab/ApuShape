import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

folder_path_images = cfg.get('main', "folder_path_images")
folder_path_masks = cfg.get('main', "folder_path_masks")
model_file = cfg.get("main", "model_file")
n_rays = int(cfg.get("globe", "n_rays"))
er1 = int(cfg.get("main", "er1"))
er2 = int(cfg.get("main", "er2"))
fl1 = int(cfg.get("main", "fl1"))
fl2 = int(cfg.get("main", "fl2"))

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

model = FlexibleUNet_star(in_channels=3, out_channels=n_rays + 1, backbone='convnext_small', pretrained=False, n_rays=n_rays, prob_out_channels=1).to(device)
checkpoint = torch.load(model_file, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

batch_coords = []
batch_distances = []
batch_points = []
batch_feasiblecontours = []

for num, (image_filename, mask_filename) in enumerate(tqdm(zip(sorted_images_filenames, sorted_masks_filenames), total=len(sorted_images_filenames))):

    image_path = os.path.join(folder_path_images, image_filename)
    mask_path = os.path.join(folder_path_masks, mask_filename)

    img = imread(image_path)
    img_mask = imread(mask_path)
    h, w = img.shape[:2]

    with torch.no_grad():
        test_tensor = torch.from_numpy(np.expand_dims(img, 0)).permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
        output_dist, output_prob = sliding_window_inference(test_tensor, (h), 8, model)

        prob = output_prob[0][0].cpu().numpy()
        dist = output_dist[0].cpu().numpy()
        dist = np.transpose(dist, (1, 2, 0))
        dist = np.maximum(1e-3, dist)

        pointsi, probi, disti, nms = non_maximum_suppression(dist, prob, prob_thresh=0.4, nms_thresh=0.35)
        coords = dist_to_coord(disti, pointsi)

        if nms is not None:
            nms['coord'] = dist_to_coord(nms['dist'], nms['points'])

        res = dict(coord=coords, points=pointsi, prob=probi, nms=nms)
        labels = polygons_to_label(disti, pointsi, shape=img.shape)
        refine_shapes = dict()
        feasiblecontours = refine(labels, res, **refine_shapes)
        img_coords = [[[x, y] for x, y in zip(item[1], item[0])] for item in coords]
        points = [[y, x] for x, y in pointsi]

        batch_points.append(points)
        batch_coords.append(img_coords)
        batch_distances.append(disti)
        batch_feasiblecontours.append(feasiblecontours[:, :, 0])

print("Refine image num : %d"%(len(sorted_images_filenames)))

for num, (images_filename, img_distances, img_points, img_coords, img_fc) in enumerate(tqdm(zip(sorted_images_filenames, batch_distances, batch_points, batch_coords, batch_feasiblecontours), total=len(sorted_images_filenames))):
    if img_distances is not None:

        refined_coords = []

        image_path = os.path.join(folder_path_images, images_filename)
        mask_path = os.path.join(folder_path_masks, images_filename)
        img = imread(image_path)
        img_mask = imread(mask_path)

        for index, (coords, distances, point) in enumerate(zip(img_coords, img_distances, img_points)):

            all_inter_annular_values = []
            all_outer_annular_values = []

            coords = int_point(coords)
            distances_copy = distances.copy()
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
                        adjusted_distances = saved_value[-1].copy()
                        adjusted_distances[i] += 1
                        end_coords1 = int_point(calculate_endpoint_coordinates(point, adjusted_distances))
                        if (0 <= end_coords1[i][0] < img_fc.shape[0]) and (0 <= end_coords1[i][1] < img_fc.shape[1]):
                            if img_fc[end_coords1[i][1], end_coords1[i][0]] == img_fc[point[1], point[0]]:
                                end_coords2 = calculate_endpoint_coordinates(point, saved_value[-2])
                                _, outer_annular_value = calculate_region_mean_in_difference(img, end_coords1, int_point(end_coords2))
                                if outer_annular_value > min_outer_annular_value:
                                    saved_value.append(adjusted_distances)
                                else:
                                    skip_indices.add(i)

            refined_coord = calculate_endpoint_coordinates(point, saved_value[-1])
            draw_dashed_contour(img, int_point(coords), "yellow")
            draw_dashed_contour(img, int_point(refined_coord), "red")
            refined_coords.append(int_point(refined_coord))
        plt.show()

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

print(f"Average BAP_refined: {average_BAP_refined}")
print(f"Average BAP_initial: {average_BAP_initial}")