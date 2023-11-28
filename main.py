import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.image_path = None
        self.kernel_size = 5
        self.gamma = 1.5  # Giá trị gamma mặc định
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("Median Filter")  # Giá trị mặc định

        self.create_widgets()

    def apply_noise_reduction(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (self.kernel_size, self.kernel_size), 0)

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Processed Image (Noise Reduction)', blurred_image)
        cv2.resizeWindow('Original Image', 800, 600)  # Thay đổi kích thước cửa sổ hiển thị
        cv2.resizeWindow('Processed Image (Noise Reduction)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_contrast(self):
        # Đọc ảnh
        image = cv2.imread(self.image_path)

        # Chuyển đổi sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng phương pháp CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Hiển thị ảnh gốc và ảnh được tăng cường độ tương phản
        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Processed Image (Contrast)', enhanced_image)
        cv2.resizeWindow('Original Image', 800, 600)
        cv2.resizeWindow('Processed Image (Contrast)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_stretch(self):
        image = cv2.imread(self.image_path)

        # Chuyển đổi sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng phương pháp power-law transformation (gamma correction)
        stretched_image = np.power(gray_image / float(np.max(gray_image)), self.gamma) * 255.0
        stretched_image = np.uint8(stretched_image)

        # Hiển thị ảnh gốc và ảnh được dãn

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Processed Image (Contrast)', stretched_image)
        cv2.resizeWindow('Original Image', 800, 600)
        cv2.resizeWindow('Processed Image (Contrast)', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_max_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        max_filtered_image = cv2.dilate(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Max Filtered Image', max_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def apply_min_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_filtered_image = cv2.erode(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Min Filtered Image', min_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_midpoint_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        max_filtered_image = cv2.dilate(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        min_filtered_image = cv2.erode(gray_image, np.ones((self.kernel_size, self.kernel_size), np.uint8))
        midpoint_filtered_image = (max_filtered_image + min_filtered_image) // 2

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Midpoint Filtered Image', midpoint_filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_Mean_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.blur(gray_image, self.kernel_size)

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Filtered Image (Median Filter)', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_widgets(self):
        # Nút để chọn ảnh
        select_image_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        select_image_button.pack(pady=10)

        # Dropdown để chọn thuật toán
        algorithms = ["Median Filter", "Noise Reduction", "Contrast", "Stretch", "Max and Min Filter", "Midpoint Filter"]
        algorithm_menu = tk.OptionMenu(self.root, self.algorithm_var, *algorithms)
        algorithm_menu.pack(pady=10)

        # Nút để thực hiện xử lý ảnh
        process_button = tk.Button(self.root, text="Process Image", command=self.process_image)
        process_button.pack(pady=10)

    def select_image(self):
        # Hiển thị hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path

    def process_image(self):
        if self.image_path:
            algorithm = self.algorithm_var.get()

            if algorithm == "Lọc trung vị":
                self.apply_median_filter()
            elif algorithm == "Noise Reduction":
                self.apply_noise_reduction()
            elif algorithm == "Co ảnh":
                self.apply_contrast()
            elif algorithm == "Giãn ảnh":
                self.apply_stretch()
            elif algorithm == "Max Filter":
                self.apply_max_filter()
            elif algorithm == "Min Filter":
                self.apply_min_filter()
            elif algorithm == "Lọc điểm giữa":
                self.apply_midpoint_filter()
            elif algorithm == "Lọc trung bình":
                self.apply_Mean_filter()

    def apply_median_filter(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.medianBlur(gray_image, self.kernel_size)

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Filtered Image (Median Filter)', filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
