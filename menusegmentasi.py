import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class MenuSegmentasi:

    outputPath = ".output"
    outputFile = rf"{outputPath}\output.png"

    def plot_images(original_image, result_image):
        plt.figure(figsize=(10, 5))

        # Plot gambar asli
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('on')

        # Plot gambar hasil segmentasi
        plt.subplot(1, 2, 2)
        plt.title("Segmented Image")
        # plt.imshow(result_image, cmap='gray')
        plt.imshow(result_image)
        plt.axis('on')

        plt.tight_layout()
        plt.show()

    def region_growing(image_path, seed_point, threshold):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        segmented[seed_point] = 255
        region_intensity = image[seed_point]

        to_check = [seed_point]

        while to_check:
            current_point = to_check.pop(0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = current_point[0] + dx, current_point[1] + dy
                    if 0 <= x < h and 0 <= y < w and segmented[x, y] == 0:
                        if abs(int(image[x, y]) - int(region_intensity)) <= threshold:
                            segmented[x, y] = 255
                            to_check.append((x, y))

        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)
        
        # Konversi array numpy menjadi gambar PIL
        segmented_image_pil = Image.fromarray(segmented)
        
        # Simpan gambar dengan nama file yang benar
        # segmented_image_pil.save(MenuSegmentasi.outputFile)

        # Tampilkan gambar asli dan hasil segmentasi
        # image_ori = cv2.imread(image_path)
        # MenuSegmentasi.plot_images(image_ori, segmented)

        return segmented_image_pil
    
    def kmeans_clustering(image_path, k):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        pixel_values = image.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)

        # Definisikan kriteria untuk algoritma K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Ubah label menjadi citra hasil klaster
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)

        # Konversi array numpy menjadi gambar PIL
        kmeans_image_pil = Image.fromarray(segmented_image)
        
        # Simpan gambar dengan nama file yang benar
        # kmeans_image_pil.save(MenuSegmentasi.outputFile)

        # Tampilkan gambar asli dan hasil klasterisasi
        # image_ori = cv2.imread(image_path)
        # MenuSegmentasi.plot_images(image_ori, segmented_image)

        return kmeans_image_pil
    
    def watershed_segmentation(image_path):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape

        # Menggunakan thresholding untuk binerisasi
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Menggunakan morfologi untuk menghilangkan noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Tentukan sure background dan sure foreground
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Label unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label marker
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        image[markers == -1] = [255]  # Garis watershed

        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)

        # Konversi array numpy menjadi gambar PIL
        watershed_image_pil = Image.fromarray(image)

        # # Simpan gambar dengan nama file yang benar
        # watershed_image_pil.save(MenuSegmentasi.outputFile)

        # # Tampilkan gambar asli dan hasil watershed
        # image_ori = cv2.imread(image_path)
        # MenuSegmentasi.plot_images(image_ori, image)

        return watershed_image_pil
    
    def global_thresholding(image_path, threshold_value):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Terapkan global thresholding
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)

        # Konversi array numpy menjadi gambar PIL
        binary_image_pil = Image.fromarray(binary_image)
        
        # # Simpan gambar dengan nama file yang benar
        # binary_image_pil.save(MenuSegmentasi.outputFile)

        # # Tampilkan gambar asli dan hasil global thresholding
        # image_ori = cv2.imread(image_path)
        # MenuSegmentasi.plot_images(image_ori, binary_image)

        return binary_image_pil
    
    def adaptive_thresh_mean(image_path):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Terapkan thresholding adaptif mean
        adaptive_thresh_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY, 11, 2)

        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)

        # Konversi array numpy menjadi gambar PIL
        adaptive_mean_image_pil = Image.fromarray(adaptive_thresh_mean)
        
        # Simpan gambar dengan nama file yang benar
        # adaptive_mean_image_pil.save(MenuSegmentasi.outputFile)

        # Tampilkan gambar asli dan hasil thresholding adaptif mean
        # image_ori = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca citra grayscale untuk kesesuaian plot
        # MenuSegmentasi.plot_images(image_ori, adaptive_thresh_mean)

        return adaptive_mean_image_pil
    
    def adaptive_thresh_gaussian(image_path):
        # Membaca citra grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Terapkan thresholding adaptif Gaussian
        adaptive_thresh_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 11, 2)
        # Buat direktori jika belum ada
        # if not os.path.exists(MenuSegmentasi.outputPath):
        #     os.makedirs(MenuSegmentasi.outputPath)

        # Konversi array numpy menjadi gambar PIL
        adaptive_gaussian_image_pil = Image.fromarray(adaptive_thresh_gaussian)
        
        # Simpan gambar dengan nama file yang benar
        # adaptive_gaussian_image_pil.save(MenuSegmentasi.outputFile)

        # Tampilkan gambar asli dan hasil thresholding adaptif Gaussian
        # image_ori = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca citra grayscale untuk kesesuaian plot
        # MenuSegmentasi.plot_images(image_ori, adaptive_thresh_gaussian)

        return adaptive_gaussian_image_pil

# Contoh penggunaan
image_path = "sample/image.jpeg"
seed = (10, 10)  # Koordinat seed point
threshold_value = 20  # Threshold
# MenuSegmentasi.region_growing(image_path, seed, threshold_value)
# MenuSegmentasi.kmeans_clustering(image_path, 2)
# MenuSegmentasi.watershed_segmentation(image_path)
# MenuSegmentasi.global_thresholding(image_path, 100)
# MenuSegmentasi.adaptive_thresh_mean(image_path)
# MenuSegmentasi.adaptive_thresh_gaussian(image_path)