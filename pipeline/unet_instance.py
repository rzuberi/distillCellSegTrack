import numpy as np
import torch
from unet_architecture import UNet
import torch
import numpy as np
import cv2
from scipy.ndimage import label

class UNetModel:
    def __init__(self, encChannels=(1, 32, 64, 128, 256), decChannels=(256, 128, 64, 32), nbClasses=1):
        self.encChannels = encChannels
        self.decChannels = decChannels
        self.nbClasses = nbClasses
        self.device = torch.device('cpu')  # Default device is CPU
        self.model = None

    def set_device(self, device):
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            print('CUDA is not available. Using CPU.')
            self.device = torch.device('cpu')

    def load_weights(self, weights_path):
        self.model = UNet(self.encChannels, self.decChannels, self.nbClasses)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images):
        if isinstance(images, np.ndarray):
            images = [images]  # Convert single image to list

        predictions = []
        for image in images:
            instance_segmentation = self._process_image(image)
            predictions.append(instance_segmentation)

        return predictions

    def _process_image(self, image):
        image = self._normalize_image(image)
        tiles, height, width, num_tiles_vertical, num_tiles_horizontal, pad_height, pad_width, tile_size, overlap = self._cut_image_into_tiles(image)
        predictions = self._predict_tiles(tiles)
        reassembled_image = self._reassemble_tiles(predictions, height, width, num_tiles_vertical, num_tiles_horizontal, pad_height, pad_width, tile_size, overlap)
        reassembled_image = 1 / (1 + np.exp(-reassembled_image))
        reassembled_image = reassembled_image > 0.5
        instance_segmentation = self._binary_to_instance(reassembled_image)
        return instance_segmentation

    def _normalize_image(self, image):
        if np.max(image) > 1:
            min_val = np.min(image)
            max_val = np.max(image)
            image = (image - min_val) / (max_val - min_val)
        return image

    def _cut_image_into_tiles(self, image):
        height, width = image.shape[:2]
        tile_size = 256
        overlap = tile_size // 2

        # Calculate the number of tiles needed in each dimension
        num_tiles_vertical = int(np.ceil(height / (tile_size - overlap)))
        num_tiles_horizontal = int(np.ceil(width / (tile_size - overlap)))

        # Calculate the padding required for the image
        pad_height = num_tiles_vertical * (tile_size - overlap) + overlap
        pad_width = num_tiles_horizontal * (tile_size - overlap) + overlap

        # Pad the image
        padded_image = np.pad(image, ((0, pad_height - height), (0, pad_width - width)), mode='constant')

        # Create an empty array to store the tiles
        tiles = np.empty((num_tiles_vertical, num_tiles_horizontal, tile_size, tile_size), dtype=image.dtype)

        # Generate the tiles
        for i in range(num_tiles_vertical):
            for j in range(num_tiles_horizontal):
                y_start = i * (tile_size - overlap)
                y_end = y_start + tile_size
                x_start = j * (tile_size - overlap)
                x_end = x_start + tile_size
                tiles[i, j] = padded_image[y_start:y_end, x_start:x_end]

        return tiles, height, width, num_tiles_vertical, num_tiles_horizontal, pad_height, pad_width, tile_size, overlap
    
    def _predict_tiles(self, tiles):
        predictions = []
        with torch.no_grad():
            for col in tiles:
                col_preds = []
                for tile in col:
                    tile = torch.from_numpy(np.float32(tile)).unsqueeze(0).unsqueeze(0).float()
                    tile = tile.to(self.device)
                    _, _, pred = self.model(tile)
                    pred = pred.cpu().detach().numpy().squeeze()
                    col_preds.append(pred)
                col_preds = np.array(col_preds)
                predictions.append(col_preds)
        predictions = np.array(predictions)
        return predictions

    def _reassemble_tiles(self, tiles, original_img_height, original_img_width, num_tiles_vertical, num_tiles_horizontal, pad_height, pad_width, tile_size, overlap):
        # Create an empty array to store the reassembled image
        reassembled_image = np.empty((pad_height, pad_width))

        # Generate the reassembled image by averaging overlapping tiles
        for i in range(num_tiles_vertical):
            for j in range(num_tiles_horizontal):
                y_start = i * (tile_size - overlap)
                y_end = y_start + tile_size
                x_start = j * (tile_size - overlap)
                x_end = x_start + tile_size
                reassembled_image[y_start:y_end, x_start:x_end] = tiles[i, j]

        # Crop the padded regions from the reassembled image
        cropped_image = reassembled_image[:original_img_height, :original_img_width]

        return cropped_image

    def _binary_to_instance(self, img):
        image = img.astype(np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((1, 1), dtype=int))

        a = image
        img = image

        border = cv2.dilate(img, None, iterations=1)
        border = border - cv2.erode(border, None)

        dt = cv2.distanceTransform(img, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, 130, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        # Completing the markers now.
        lbl[border == 255] = 255

        lbl = lbl.astype(np.int32)
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        cv2.watershed(a, lbl)

        lbl[lbl == -1] = 0
        lbl = lbl.astype(np.uint8)

        result = 255 - lbl

        # Find the most frequent value in the array and set it to zero
        unique_values, counts = np.unique(result, return_counts=True)
        sorted_indices = np.argsort(counts)
        most_frequent_values = unique_values[sorted_indices[-2:]]
        result[np.isin(result, most_frequent_values)] = 0

        unique_values = np.unique(result)  # Find the unique values in the array
        sorted_values = np.sort(unique_values)  # Sort the unique values
        value_index_map = {value: index for index, value in enumerate(sorted_values)}
        ordered_array = np.vectorize(lambda x: value_index_map[x])(result)

        return ordered_array
