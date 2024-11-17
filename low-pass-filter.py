import numpy as np
import matplotlib.pyplot as plt

cutoff_frequency = 20
image_filename = "100x100.jpg"


def calculate_2dft(input):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input)))


def calculate_2dift(input):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input))).real


def calculate_distance_from_centre(coords, centre):
    return np.sqrt((coords[0] - centre) ** 2 + (coords[1] - centre) ** 2)


def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]), centre + (centre - coords[1]))


image = plt.imread(image_filename)
image = image[:, :, :3].mean(axis=2)

array_size = min(image.shape) - 1 + min(image.shape) % 2

image = image[:array_size, :array_size]
centre = int((array_size - 1) / 2)

coords_left_half = sorted(
    ((x, y) for x in range(array_size) for y in range(centre + 1)),
    key=lambda x: calculate_distance_from_centre(x, centre),
)

plt.set_cmap("gray")
plt.subplots_adjust(wspace=0, hspace=1.0)

ft = calculate_2dft(image)

low_pass_filter = np.fromfunction(
    lambda x, y: calculate_distance_from_centre((x, y), centre) <= cutoff_frequency,
    ft.shape,
    dtype=int,
)

ft_filtered = ft * low_pass_filter

plt.subplot(121)
plt.title("Original")
plt.axis("off")
plt.imshow(image)

rec_image = np.zeros(image.shape)
individual_grating = np.zeros(image.shape, dtype="complex")

for coords in coords_left_half:
    if coords[1] == centre and coords[0] > centre:
        continue

    symm_coords = find_symmetric_coordinates(coords, centre)

    individual_grating[coords] = ft_filtered[coords]
    individual_grating[symm_coords] = ft_filtered[symm_coords]

    rec_image += calculate_2dift(individual_grating)

    individual_grating[coords] = 0
    individual_grating[symm_coords] = 0

plt.subplot(122)
plt.title("Result")
plt.axis("off")
plt.imshow(rec_image)
plt.show()
