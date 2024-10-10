from PIL import Image
import numpy as np

class Rgb_filter:
    
    def __init__(self, imgFile):
        self.imagefile = imgFile

    def filter_kuning(self):
        image_np = np.array(self.imagefile)

        # Apply yellow filter (increase red and green, remove blue)
        image_np[:, :, 2] = 0  # Remove blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        return image_out

    def filter_orange(self):
        image_np = np.array(self.imagefile)

        # Apply orange filter (reduce blue component)
        image_np[:, :, 2] = image_np[:, :, 2] // 2  # Reduce blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        return image_out

    def filter_cyan(self):
        image_np = np.array(self.imagefile)

        # Apply cyan filter (remove red component)
        image_np[:, :, 0] = 0  # Remove red component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        return image_out

    def filter_purple(self):
        image_np = np.array(self.imagefile)

        # Apply purple filter (remove green component)
        image_np[:, :, 1] = 0  # Remove green component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        return image_out

    def filter_grey(self):
        image_np = np.array(self.imagefile)

        # Convert to greyscale
        grey_image = np.mean(image_np, axis=2).astype(np.uint8)

        image_out = Image.fromarray(grey_image)
        return image_out

    def filter_coklat(self):
        image_np = np.array(self.imagefile)

        # Adjust red, green, and blue components to produce a brownish hue
        brown_filter = image_np.copy()
        brown_filter[:, :, 0] = brown_filter[:, :, 0] // 1.5  # Slightly reduce red
        brown_filter[:, :, 1] = brown_filter[:, :, 1] // 2.5  # Further reduce green
        brown_filter[:, :, 2] = brown_filter[:, :, 2] // 3  # Reduce blue even more

        image_out = Image.fromarray(brown_filter.astype(np.uint8))
        return image_out

    def filter_merah(self):
        image_np = np.array(self.imagefile)

        # Apply red filter (remove green and blue components)
        image_np[:, :, 1] = 0  # Remove green component
        image_np[:, :, 2] = 0  # Remove blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        return image_out
