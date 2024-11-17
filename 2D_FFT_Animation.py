# fourier_synthesis.py

import numpy as np
import matplotlib.pyplot as plt

image_filename = "16x16.jpg"


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt((coords[0] - centre) ** 2 + (coords[1] - centre) ** 2)


def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]), centre + (centre - coords[1]))


def display_plots(part_ft, acc_ft, individual_grating, reconstruction, idx):
    plt.subplot(334)
    plt.title(f"Part ft step={idx}")
    plt.imshow(part_ft)

    plt.subplot(335)
    plt.title(f"Acc ft step={idx}")
    plt.imshow(acc_ft)

    plt.subplot(336)
    plt.title(f"Ind step={idx}")
    plt.imshow(individual_grating)

    plt.subplot(337)
    plt.title(f"Rec step={idx}")
    plt.imshow(reconstruction)

    plt.pause(0.01)


# Read and process image
image = plt.imread(image_filename)
image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

# Array dimensions (array is square) and centre pixel
# Use smallest of the dimensions and ensure it's odd
array_size = min(image.shape) - 1 + min(image.shape) % 2

# Crop image so it's a square image
image = image[:array_size, :array_size]
centre = int((array_size - 1) / 2)

# Get all coordinate pairs in the left half of the array,
# including the column at the centre of the array (which
# includes the centre pixel)
coords_left_half = ((x, y) for x in range(array_size) for y in range(centre + 1))

# Sort points based on distance from centre
coords_left_half = sorted(
    coords_left_half, key=lambda x: calculate_distance_from_centre(x, centre)
)

plt.set_cmap("gray")
plt.subplots_adjust(wspace=0, hspace=1.0)


ft = calculate_2dft(image)

# Show grayscale image and its Fourier transform
plt.subplot(331)
plt.title("Original")
plt.imshow(image)

plt.subplot(332)
plt.title("Freq domain")
plt.imshow(abs(ft))

plt.subplot(333)
plt.title("log freq domain")
plt.imshow(np.log(abs(ft)))

# Step 1
# Set up empty arrays for final image and
# individual gratings
rec_image = np.zeros(image.shape)
individual_grating = np.zeros(image.shape, dtype="complex")

# Count of layers added to the final image
idx = 0

# All steps are displayed until display_all_until value
display_all_until = 200
# After this, skip which steps to display using the
# display_step value
display_step = 10
# Work out index of next step to display
next_display = display_all_until + display_step
# Step 1: Initialize the partial Fourier transform array
partial_ft = np.zeros_like(ft)  # To store the accumulated frequency components
acc_ft = np.zeros_like(ft)  # To store the accumulated frequency components

# Step 2: Modify the loop
for coords in coords_left_half:
    # Central column: only include if points in top half of the central column
    if coords[1] == centre and coords[0] > centre:
        continue

    idx += 1
    symm_coords = find_symmetric_coordinates(coords, centre)

    # Step 3: Add the frequency components to partial_ft
    partial_ft[coords] = ft[coords]
    partial_ft[symm_coords] = ft[symm_coords]
    acc_ft[coords] = ft[coords]
    acc_ft[symm_coords] = ft[symm_coords]

    # Step 4: Copy values from partial_ft into individual_grating for reconstruction
    individual_grating[coords] = ft[coords]
    individual_grating[symm_coords] = ft[symm_coords]

    # Step 5: Calculate the inverse Fourier transform to get the reconstructed grating
    rec_grating = calculate_2dift(individual_grating)
    rec_image += rec_grating

    # Step 6: Display the partial Fourier Transform and reconstruction
    display_plots(
        np.log(abs(partial_ft) + 1),  # Partial Fourier Transform (log-scaled)
        np.log(abs(acc_ft) + 1),  # Accumulated Fourier Transform
        rec_grating,  # Current grating
        rec_image,  # Reconstructed image
        idx,
    )

    # Clear individual_grating array for the next iteration
    individual_grating[coords] = 0
    individual_grating[symm_coords] = 0
    partial_ft[coords] = 0
    partial_ft[symm_coords] = 0


plt.show()
