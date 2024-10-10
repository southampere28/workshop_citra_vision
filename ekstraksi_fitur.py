import cv2
import os
import pandas as pd
import mahotas as mt

class extraction_feature:
    def ekstraksi_warna(pathImg):
        # Tentukan path file gambar
        file_path = pathImg
        filename = os.path.basename(file_path)

        # Cek apakah file adalah gambar (berformat .png, .jpg, atau .jpeg)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Baca gambar menggunakan OpenCV
            img = cv2.imread(file_path)

            # Hitung nilai rata-rata warna RGB untuk seluruh gambar
            avg_color_per_row = cv2.mean(img)[:3]  # Ambil 3 komponen pertama (R, G, B)
            R, G, B = avg_color_per_row

            # Tentukan path output
            output_folder = 'output_feature'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)  # Buat folder jika belum ada

            output_path = os.path.join(output_folder, 'rgb_average.xlsx')

            # Cek apakah file Excel sudah ada
            if os.path.exists(output_path):
                # Baca file Excel yang sudah ada
                df_existing = pd.read_excel(output_path)

                # Dapatkan nomor urut terakhir
                last_no = df_existing['No'].max()
            else:
                # Jika belum ada, buat dataframe baru
                df_existing = pd.DataFrame(columns=['No', 'Nama', 'R', 'G', 'B'])
                last_no = 0

            # Tambahkan data baru ke dataframe
            new_data = pd.DataFrame([[last_no + 1, filename, R, G, B]], columns=['No', 'Nama', 'R', 'G', 'B'])
            df_updated = pd.concat([df_existing, new_data], ignore_index=True)

            # Simpan hasil ke file Excel (menimpa file lama dengan data baru)
            df_updated.to_excel(output_path, index=False)

            # Mengembalikan hasil path file yang tersimpan
            return f'Hasil ekstraksi telah disimpan di {output_path}'
        else:
            return "File yang dimasukkan bukan gambar yang valid (hanya .png, .jpg, .jpeg)."


    def ekstraksi_texture(pathImg):
        # Fungsi untuk menghitung fitur GLCM dengan mahotas
        def extract_glcm_features(image_gray):
            glcm = mt.features.haralick(image_gray).mean(axis=0)
            contrast = glcm[1]   # Contrast
            homogeneity = glcm[4]  # Homogeneity
            energy = glcm[8]   # Energy
            correlation = glcm[2]  # Correlation
            return contrast, homogeneity, energy, correlation

        # Path file gambar
        file_path = pathImg
        filename = os.path.basename(file_path)

        # Cek apakah file adalah gambar (berformat .png, .jpg, .jpeg)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Baca gambar menggunakan OpenCV
            img = cv2.imread(file_path)

            # Konversi gambar ke grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Ekstraksi fitur GLCM
            contrast, homogeneity, energy, correlation = extract_glcm_features(img_gray)

            # Tentukan path output
            output_folder = 'output_feature'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)  # Buat folder jika belum ada

            output_path = os.path.join(output_folder, 'glcm_features_output.xlsx')

            # Cek apakah file Excel sudah ada
            if os.path.exists(output_path):
                # Baca file Excel yang sudah ada
                df_existing = pd.read_excel(output_path)

                # Dapatkan nomor urut terakhir
                last_no = df_existing['No'].max()
            else:
                # Jika belum ada, buat dataframe baru
                df_existing = pd.DataFrame(columns=['No', 'Nama', 'Kontras', 'Homogenitas', 'Energi', 'Korelasi'])
                last_no = 0

            # Tambahkan data baru ke dataframe
            new_data = pd.DataFrame([[last_no + 1, filename, contrast, homogeneity, energy, correlation]], 
                                    columns=['No', 'Nama', 'Kontras', 'Homogenitas', 'Energi', 'Korelasi'])
            df_updated = pd.concat([df_existing, new_data], ignore_index=True)

            # Simpan hasil ke file Excel (menimpa file lama dengan data baru)
            df_updated.to_excel(output_path, index=False)

            # Mengembalikan hasil path file yang tersimpan
            return f'Hasil ekstraksi fitur GLCM telah disimpan di {output_path}'
        else:
            return "File yang dimasukkan bukan gambar yang valid (hanya .png, .jpg, .jpeg)."