import os
import cv2
import pandas as pd
import numpy as np
from numpy import logical_xor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import label, maximum_filter, binary_erosion


# DataFrame
cell_data_list = []
image_data_list = []

def compute_perimeter(region_mask):
    """Approximate the perimeter of the region using edge detection."""
    # Binary corrosion of regions using predefined structural elements
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    eroded_image = binary_erosion(region_mask, structure)
    # User logical XOR because direct subtraction is not applicable in Boolean types
    boundary = logical_xor(region_mask, eroded_image)
    return np.sum(boundary)

def compute_cell_centroids(watershed_labels, num_cells):
    """Compute centroid (X, Y) for each cell."""
    centroids = []
    for cell_label in range(1, num_cells + 1):
        region_mask = (watershed_labels == cell_label).astype(np.uint8)
        moments = cv2.moments(region_mask)
        if moments["m00"] != 0:  # Avoid division by zero
            cx = int(moments["m10"] / moments["m00"])  # X coordinate
            cy = int(moments["m01"] / moments["m00"])  # Y coordinate
            centroids.append({"label": cell_label, "x": cx, "y": cy})
    return centroids

def compute_local_density(centroids, radius=50):
    """Compute local cell density (number of neighbors within a given radius)."""
    if not centroids:  # Check if centroids list is empty
        return  # If empty, skip density calculation

    positions = np.array([[cell["x"], cell["y"]] for cell in centroids])

    if positions.shape[0] < 2:
        for cell in centroids:
            cell["local_density"] = 0  # No neighbors, density = 0
        return

    distances = cdist(positions, positions)  # Compute pairwise distances
    local_densities = np.sum(distances < radius, axis=1) - 1  # Count neighbors (excluding self)

    for i, cell in enumerate(centroids):
        cell["local_density"] = local_densities[i]

def distance_tranform(num_labels, stats, filtered_binary, labels, area_t):
    # Preserve small connected domains
    # Retain only small connected components
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_t:  # set maximum area threshold
            filtered_binary[labels == i] = 255

    plt.figure()
    plt.imshow(filtered_binary, cmap='gray')
    plt.title('Filtered Binary Image')
    plt.show()

    # 3. Distance Transform
    dist_transform = cv2.distanceTransform(filtered_binary, cv2.DIST_L2, 5)
    plt.figure()
    plt.imshow(dist_transform, cmap='gray')
    plt.title('Distance Transform')
    plt.show()

    # 4. Detect Local Maxima as Seeds
    local_max = maximum_filter(dist_transform, size=30)  # adjust window size
    maxima = (dist_transform == local_max) & (dist_transform > 0.3 * dist_transform.max())  # Enhanced threshold filtering
    labeled_maxima, num_cells = label(maxima)  # Marked seed point

    return labeled_maxima, num_cells


def count_cells_from_distance_transform(image_path):
    """Main function to process image and extract cell features."""
    # Load Image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name = image_path.split('/')[-1]
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # 1. Preprocessing
    # Noise Reduction
    image_filtered = cv2.medianBlur(image, 5)

    # Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_filtered)
    plt.figure()
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Contrast')
    plt.show()

    # Threshold Segmentation
    _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Morphological Operations - Noise Removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Connected domain analysis to remove large areas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    filtered_binary = np.zeros_like(binary_image)
    labeled_maxima, num_cells = distance_tranform(num_labels, stats, filtered_binary, labels, 120000)

    area_t = 50000
    while num_cells < 10 and area_t >= 10000:
      print(f"num_cells < 10: Reapplying area filtering with stricter threshold {area_t}")

      # Creates a new binary image to store the filtered region
      filtered_binary = np.zeros_like(binary_image)
      labeled_maxima, num_cells = distance_tranform(num_labels, stats, filtered_binary, labels, area_t)

      area_t -= 10000

    # Watershed Algorithm for Segmentation
    markers = np.zeros_like(filtered_binary, dtype=np.int32)

    # Iterate through each seed point, assigning unique tags
    seed_indices = np.argwhere(labeled_maxima)  # Get the seed point coordinates
    for idx, (y, x) in enumerate(seed_indices, start=1):
        markers[y, x] = idx

    watershed_labels = cv2.watershed(cv2.cvtColor(filtered_binary, cv2.COLOR_GRAY2BGR), markers)

    # 2. Compute Features for Each Cell
    cell_features = []
    centroids = compute_cell_centroids(watershed_labels, num_cells)
    compute_local_density(centroids, radius=50)

    for centroid in centroids:
        region_mask = (watershed_labels == centroid["label"]).astype(np.uint8)
        area = np.sum(region_mask)  # area
        perimeter = compute_perimeter(region_mask)  # perimeter

        cell_features.append({
            "Image Name": image_name,
            "Cell ID": centroid["label"],
            "X Position": centroid["x"],
            "Y Position": centroid["y"],
            "Area": area,
            "Perimeter": perimeter,
            "Local density": centroid["local_density"]
        })

    # Append cell data to list
    cell_data_list.extend(cell_features)


    # 3. Compute Image-Level Summary Statistics
    avg_area = np.mean([cell["Area"] for cell in cell_features])
    avg_perimeter = np.mean([cell["Perimeter"] for cell in cell_features])
    avg_local_density = np.mean([cell["Local density"] for cell in cell_features])


    image_summary = {
        "Image Name": image_name,
        "Total Cells": num_cells,
        "Average Area": avg_area,
        "Average Perimeter": avg_perimeter,
        "Average Local Density": avg_local_density
    }

    image_data_list.append(image_summary)

    # 4. Draw final image with labels
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for cell_label in range(1, num_cells + 1):
        coords = np.argwhere(labeled_maxima == cell_label)
        if len(coords) == 0:  # Prevent invalid seed points
            continue
        y, x = coords[0]  # Select a coordinate for the seed point
        cv2.circle(image_colored, (x, y), 5, (0, 255, 0), -1)  # draw seed point
        cv2.putText(image_colored, str(cell_label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_colored, cv2.COLOR_BGR2RGB))
    plt.title(f'Total Cells Detected: {num_cells}')
    plt.show()


    # Print each cell's features
    for cell in cell_features:
      print(f"Cell {cell['Cell ID']}: Area = {cell['Area']}, Perimeter = {cell['Perimeter']}, "
                    f"Position = ({cell['X Position']}, {cell['Y Position']}), Local Density = {cell['Local density']}")
    return num_cells, cell_features

"""Support natural language or SQL-like queries:"""
def load_data(choice):
    if choice == "cell":
        return pd.read_csv(cell_data_path)
    elif choice == "summary":
        return pd.read_csv(image_summary_path)
    else:
        print("N/A")
        return None

def run_query(query_string, df):
    try:
        result = df.query(query_string)
        print(f"Query executed: {query_string}")
        display(result.head())
        return result
    except Exception as e:
        print(f"Query error: {e}")

def plot_histogram(column_name, df, data_type="cell"):
    print("\n")
    plt.figure(figsize=(8, 6))
    plt.hist(df[column_name], bins=20, color="blue", alpha=0.7, edgecolor="black")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name} ({data_type.capitalize()} Level Data)")
    plt.grid(True)
    plt.show()

def plot_scatter(x_col, y_col, df, data_type="cell"):
    print("\n")
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], color="green", alpha=0.6)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scatter Plot of {x_col} vs {y_col} ({data_type.capitalize()} Level Data)")
    plt.grid(True)
    plt.show()

def plot_heatmap(column_name, df, data_type="cell"):
    print("\n")
    plt.figure(figsize=(8, 6))

    heatmap, xedges, yedges = np.histogram2d(df["X Position"], df["Y Position"], bins=[50, 50], weights=df[column_name])

    plt.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto")
    plt.colorbar(label=column_name)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Heatmap of {column_name} ({data_type.capitalize()} Level Data)")
    plt.show()


"""Test and store all data; Organized data into well-structured tabular format:"""
# Test
image_dir = "/content/images/"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

for img in image_paths:
    count_cells_from_distance_transform(img)

# Convert lists to DataFrames
df_cells = pd.DataFrame(cell_data_list)
df_images = pd.DataFrame(image_data_list)

# Save DataFrames to CSV
df_cells.to_csv("cell_data.csv", index=False)
df_images.to_csv("image_summary.csv", index=False)

# Display DataFrames
print("\n Processing Completed. Saved as 'cell_data.csv' and 'image_summary.csv'.")
print("\n Cell Data Preview:")
print(df_cells.head())

print("\n Image Summary Preview:")
print(df_images.head())


"""Output tabular-format examples and show distribusions:"""
cell_data_path = "cell_data.csv"
image_summary_path = "image_summary.csv"

data_choice = "cell"
df = load_data(data_choice)

# Example query: Find cells with Area > 1000 and Local Density > 2
query_result1 = run_query("Area > 1000 and `Local density` > 2", df)
print(query_result1)

# Example query: Find Cells in a Specific Image
query_result2 = run_query("`Image Name` == '26_01_2024_11.png'", df)
print(query_result2)

# Example query: Find Cells in the Left Half of an Image (X Position < 250)
query_result3 = run_query("`X Position` < 250", df)
print(query_result3)

# Example plot: Area Distribution
plot_histogram("Area", df, "cell")

df_sum = load_data("summary")
plot_histogram("Total Cells", df_sum, "summary")