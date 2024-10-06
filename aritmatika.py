import cv2
import numpy as np
from PIL import Image

class Aritmatika:
    def __init__(self, image1_path, image2_path):
        # Membaca citra dari path
        self.image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        # Menyamakan ukuran citra
        self.image2 = cv2.resize(self.image2, (self.image1.shape[1], self.image1.shape[0]))

    def penjumlahan(self):
        result = cv2.add(self.image1, self.image2)
        return Image.fromarray(result)

    def pengurangan(self):
        result = cv2.subtract(self.image1, self.image2)
        return Image.fromarray(result)

    def perkalian(self):
        result = cv2.multiply(self.image1, self.image2)
        return Image.fromarray(result)

    def pembagian(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = cv2.divide(self.image1.astype('float'), self.image2.astype('float'))
            result = np.nan_to_num(result).astype('uint8')
        return Image.fromarray(result)

    def or_operation(self):
        result = cv2.bitwise_or(self.image1, self.image2)
        return Image.fromarray(result)

    def and_operation(self):
        result = cv2.bitwise_and(self.image1, self.image2)
        return Image.fromarray(result)

    def xor_operation(self):
        result = cv2.bitwise_xor(self.image1, self.image2)
        return Image.fromarray(result)

# Contoh penggunaan kelas Aritmatika
# image1_path = 'sample/apple.jpg'  # Ganti dengan path citra pertama
# image2_path = 'sample/image.jpeg'  # Ganti dengan path citra kedua

# aritmatika = Aritmatika(image1_path, image2_path)

# Mendapatkan hasil operasi
# penjumlahan_result = aritmatika.penjumlahan()
# pengurangan_result = aritmatika.pengurangan()
# perkalian_result = aritmatika.perkalian()
# pembagian_result = aritmatika.pembagian()
# or_result = aritmatika.or_operation()
# and_result = aritmatika.and_operation()
# xor_result = aritmatika.xor_operation()

# Menampilkan hasil menggunakan PIL
# penjumlahan_result.show(title="Hasil Penjumlahan")
# pengurangan_result.show(title="Hasil Pengurangan")
# perkalian_result.show(title="Hasil Perkalian")
# pembagian_result.show(title="Hasil Pembagian")
# or_result.show(title="Hasil OR")
# and_result.show(title="Hasil AND")
# xor_result.show(title="Hasil XOR")
