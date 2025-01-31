import os
import open3d as o3d
import numpy as np
import torch
import time
import pandas as pd
import cv2

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

# Function to load the intrinsic matrix
def load_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    intrinsic_matrix = []
    for line in lines:
        numbers = [float(num) for num in line.strip().split()]
        intrinsic_matrix.append(numbers)
    intrinsic_matrix = np.array(intrinsic_matrix)
    return intrinsic_matrix

# Function to project 3D points to 2D using intrinsic matrix
def project_points_to_image_plane(points_3d, intrinsic_matrix, transformation_matrix):
    # Transform points with the transformation matrix
    points_3d_hom = np.hstack(
        (points_3d, np.ones((points_3d.shape[0], 1)))
    )
    transformed_points = (
        (transformation_matrix @ points_3d_hom.T).T[:, :3]
    )

    # Project transformed points to 2D
    projected_points = intrinsic_matrix @ transformed_points.T
    projected_points = (projected_points[:2] / projected_points[2]).T

    return projected_points

# Function to overlay point cloud on image
def overlay_point_cloud_on_image(
    source, image, transformation_matrix, intrinsic_matrix, output_path
):
    points_3d = np.asarray(source.points)

    # Project points to 2D
    projected_points = project_points_to_image_plane(
        points_3d, intrinsic_matrix, transformation_matrix
    )

    # Create an overlay layer
    overlay = image.copy()

    # Draw points on the overlay
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(
                overlay, (x, y), radius=2, color=(0, 0, 255), thickness=-1
            )

    # Blend the overlay with the original image
    alpha = 0.50
    blended_image = cv2.addWeighted(
        overlay, alpha, image, 1 - alpha, 0
    )

    # Save the blended image
    cv2.imwrite(output_path, blended_image)

# Function to compute evaluation metrics
def compute_metrics(estimated_matrix, ground_truth_matrix):
    translation_est = estimated_matrix[:3, 3]
    translation_gt = ground_truth_matrix[:3, 3]
    # Convert translation vectors from meters to millimeters
    translation_error_m = np.linalg.norm(translation_est - translation_gt)
    translation_error_mm = translation_error_m * 1000  # Convert to mm

    rotation_est = estimated_matrix[:3, :3]
    rotation_gt = ground_truth_matrix[:3, :3]
    rotation_diff = np.dot(rotation_est, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rotation_error = np.arccos(trace)
    return translation_error_mm, np.degrees(rotation_error)


def add_metric(model_points, R_gt, t_gt, R_pred, t_pred):
    """
    Calculates the Average Distance of Model Points (ADD) metric.
    """
    # Transform points using ground truth
    transformed_gt = (R_gt @ model_points.T).T + t_gt

    # Transform points using predictions
    transformed_pred = (R_pred @ model_points.T).T + t_pred

    # Compute average distance
    add = np.mean(np.linalg.norm(transformed_gt - transformed_pred, axis=1))
    return add

def main():
    # Paths and configurations
    scenes_root_dir = "/home/martyn/Thesis/pose-estimation/data/scenes/"
    output_root_dir = "/home/martyn/Thesis/pose-estimation/results/methods/dgr/"
    model_weights_path = "/home/martyn/Thesis/pose-estimation/methods/DeepGlobalRegistration/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
    num_runs = 5  # Number of runs per scene

    # Ensure output root directory exists
    os.makedirs(output_root_dir, exist_ok=True)

    # Load intrinsic matrix (assuming it's the same for all scenes)
    intrinsic_matrix = load_intrinsic_matrix("/home/martyn/Thesis/pose-estimation/data/cam_K.txt")
    source_path = '/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply'
    source = o3d.io.read_point_cloud(source_path)
    source.estimate_normals()

    # Error handling for source point cloud
    if source.is_empty():
        print("Error: Source point cloud is empty or failed to load.")
        return

    # Initialize DeepGlobalRegistration
    print("Initializing DeepGlobalRegistration...")
    config = get_config()
    config.weights = model_weights_path
    dgr = DeepGlobalRegistration(config)

    # Prepare lists to collect metrics over all scenes
    all_metrics = []

    for scene_num in range(1, 11):  # scenes from 1 to 10
        scene_name = f"scene_{scene_num:02d}"
        scene_dir = os.path.join(scenes_root_dir, scene_name)
        output_dir = os.path.join(output_root_dir, scene_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load scene-specific data
        target_path = os.path.join(scene_dir, "point_cloud_cropped.ply")
        ground_truth_path = os.path.join(scene_dir, "tf_ground_truth.txt")
        rgb_image_path = os.path.join(scene_dir, "rgb.png")

        target = o3d.io.read_point_cloud(target_path)
        target.estimate_normals()
        ground_truth = np.loadtxt(ground_truth_path)
        image = cv2.imread(rgb_image_path)

        # Error handling
        if target.is_empty():
            print(f"Error: Target point cloud is empty or failed to load for {scene_name}.")
            continue
        if ground_truth.size == 0:
            print(f"Error: Ground truth transformation matrix failed to load for {scene_name}.")
            continue
        if image is None:
            print(f"Error: Failed to load image for {scene_name}.")
            continue

        # Initialize lists to store metrics and runtimes for this scene
        registration_times = []
        total_runtimes = []
        translation_errors = []
        rotation_errors = []
        descriptor_times = []
        add_metrics = []

        for i in range(num_runs):
            print(f"\n{scene_name} - Run {i+1}/{num_runs}")

            # Create a directory for this run
            run_dir = os.path.join(output_dir, f"run_{i+1:02d}")
            os.makedirs(run_dir, exist_ok=True)

            # Start timing the total process for this run
            total_start_time = time.time()

            # Registration using DeepGlobalRegistration
            print("Performing registration using DeepGlobalRegistration...")
            registration_start_time = time.time()
            try:
                T_estimated, descriptor_time = dgr.register(source, target)
                descriptor_times.append(descriptor_time)
                registration_time = time.time() - registration_start_time
                registration_times.append(registration_time)
                print(f"Descriptor computation time: {descriptor_time:.2f} seconds")
                print(f"Registration time: {registration_time:.2f} seconds")
            except Exception as e:
                print(f"Error during registration: {e}")
                T_estimated = np.identity(4)
                registration_time = time.time() - registration_start_time
                registration_times.append(registration_time)

            # End timing the total process for this run
            total_runtime = time.time() - total_start_time
            total_runtimes.append(total_runtime)

            # Compute metrics
            translation_error, rotation_error = compute_metrics(T_estimated, ground_truth)

            # Compute ADD metrics
            R_pred = T_estimated[:3, :3]
            t_pred = T_estimated[:3, 3]
            R_gt = ground_truth[:3, :3]
            t_gt = ground_truth[:3, 3]
            add_error = add_metric(np.asarray(source.points), R_gt, t_gt, R_pred, t_pred)

            translation_errors.append(translation_error)
            rotation_errors.append(rotation_error)
            add_metrics.append(add_error)

            # Save transformation matrix
            np.savetxt(os.path.join(run_dir, "transformation.txt"), T_estimated)

            # Save runtime information
            with open(os.path.join(run_dir, "runtime.txt"), "w") as f:
                f.write(f"Total Runtime: {total_runtime:.4f} seconds\n")
                f.write(f"Descriptor Computation Time: {descriptor_time:.4f} seconds\n")
                f.write(f"Registration Runtime: {registration_time:.4f} seconds\n")

            # Save metrics
            with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
                f.write(f"Translation Error: {translation_error:.6f} mm\n")
                f.write(f"Rotation Error: {rotation_error:.6f} deg\n")
                f.write(f"ADD: {add_error:.6f} deg\n")

            # Overlay 3D points on 2D image and save
            print("Creating overlay image...")
            overlay_point_cloud_on_image(
                source, image, T_estimated, intrinsic_matrix,
                os.path.join(run_dir, "overlay.png")
            )

            print(f"{scene_name} - Run {i+1} results saved in {run_dir}")

        # Save overall metrics to CSV for this scene
        metrics = {
            "Run": list(range(1, num_runs+1)),
            "Translation Error (mm)": translation_errors,
            "Rotation Error (deg)": rotation_errors,
            "ADD (mm)": add_metrics,
            "Descriptor Computation Time (s)": descriptor_times,
            "Registration Runtime (s)": registration_times,
            "Total Runtime (s)": total_runtimes
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, "metrics_over_runs.csv"), index=False)
        print(f"{scene_name} - Metrics saved to:", os.path.join(output_dir, "metrics_over_runs.csv"))

        # -------------------------------------------------
        # Compute means (averages) and standard deviations
        # -------------------------------------------------
        avg_total_runtime = np.mean(total_runtimes)
        sd_total_runtime = np.std(total_runtimes)

        avg_desc_time = np.mean(descriptor_times)
        sd_desc_time = np.std(descriptor_times)

        avg_reg_time = np.mean(registration_times)
        sd_reg_time = np.std(registration_times)

        avg_trans_error = np.mean(translation_errors)
        sd_trans_error = np.std(translation_errors)

        avg_rot_error = np.mean(rotation_errors)
        sd_rot_error = np.std(rotation_errors)

        avg_add = np.mean(add_metrics)
        sd_add = np.std(add_metrics)

        # Print scene-level summary
        print(f"\n{scene_name} - Averages over runs:")
        print(f"Average Total Runtime (s): {avg_total_runtime:.6f}")
        print(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}")
        print(f"Average Registration Runtime (s): {avg_reg_time:.6f}")
        print(f"Average Translation Error (mm): {avg_trans_error:.6f}")
        print(f"Average Rotation Error (deg): {avg_rot_error:.6f}")
        print(f"Average ADD (mm): {avg_add:.6f}")

        print(f"\n{scene_name} - Standard Deviation over runs:")
        print(f"Total Runtime (s) SD: {sd_total_runtime:.6f}")
        print(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}")
        print(f"Registration Runtime (s) SD: {sd_reg_time:.6f}")
        print(f"Translation Error (mm) SD: {sd_trans_error:.6f}")
        print(f"Rotation Error (deg) SD: {sd_rot_error:.6f}")
        print(f"ADD (mm) SD: {sd_add:.6f}")

        # -------------------------------------------------
        # Save average & SD metrics to average_metrics.txt
        # -------------------------------------------------
        avg_metrics_file = os.path.join(output_dir, "average_metrics.txt")
        with open(avg_metrics_file, "w") as f:
            f.write(f"Average Metrics over runs for {scene_name}:\n")
            f.write(f"Average Total Runtime (s): {avg_total_runtime:.6f}\n")
            f.write(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}\n")
            f.write(f"Average Registration Runtime (s): {avg_reg_time:.6f}\n")
            f.write(f"Average Translation Error (mm): {avg_trans_error:.6f}\n")
            f.write(f"Average Rotation Error (deg): {avg_rot_error:.6f}\n")
            f.write(f"Average ADD (mm): {avg_add:.6f}\n")

            f.write("\nStandard Deviation over runs:\n")
            f.write(f"Total Runtime (s) SD: {sd_total_runtime:.6f}\n")
            f.write(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}\n")
            f.write(f"Registration Runtime (s) SD: {sd_reg_time:.6f}\n")
            f.write(f"Translation Error (mm) SD: {sd_trans_error:.6f}\n")
            f.write(f"Rotation Error (deg) SD: {sd_rot_error:.6f}\n")
            f.write(f"ADD (mm) SD: {sd_add:.6f}\n")

        print(f"{scene_name} - Averages and SD saved to: {avg_metrics_file}")

        # -------------------------------------------------
        # Create a dictionary for final CSV (avg + SD)
        # -------------------------------------------------
        scene_metrics = {
            "Scene": scene_name,

            "Total Runtime (s) Mean": avg_total_runtime,
            "Total Runtime (s) SD": sd_total_runtime,

            "Descriptor Computation Time (s) Mean": avg_desc_time,
            "Descriptor Computation Time (s) SD": sd_desc_time,

            "Registration Runtime (s) Mean": avg_reg_time,
            "Registration Runtime (s) SD": sd_reg_time,

            "Registration Translation Error (mm) Mean": avg_trans_error,
            "Registration Translation Error (mm) SD": sd_trans_error,

            "Registration Rotation Error (deg) Mean": avg_rot_error,
            "Registration Rotation Error (deg) SD": sd_rot_error,

            "ADD (mm) Mean": avg_add,
            "ADD (mm) SD": sd_add
        }

        # Append to all_metrics list
        all_metrics.append(scene_metrics)

    # After processing all scenes, save all average metrics to a CSV
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(os.path.join(output_root_dir, "all_scenes_average_metrics.csv"), index=False)
    print("All scenes average metrics saved to:", os.path.join(output_root_dir, "all_scenes_average_metrics.csv"))

if __name__ == '__main__':
    main()
