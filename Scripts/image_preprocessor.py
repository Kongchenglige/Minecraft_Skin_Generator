# Scripts/image_preprocessor.py

from PIL import Image
from rembg import remove


class ImagePreprocessor:
    """圖片預處理：背景移除 + 標準化"""

    def remove_background(self, image_path: str) -> Image.Image:
        """移除背景，返回 RGBA 圖片"""
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        return output_image

    def preprocess(self, image: Image.Image, target_size: int = 512) -> Image.Image:
        """標準化圖片：去背景色 → resize → 居中到正方形畫布"""
        # 轉換為 RGB（IP-Adapter 需要 RGB 輸入）
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 保持寬高比縮放
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        # 居中到正方形畫布
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - image.width) // 2, (target_size - image.height) // 2)
        canvas.paste(image, offset)

        return canvas

    def process(self, image_path: str, target_size: int = 512) -> Image.Image:
        """完整預處理管道：背景移除 → 標準化 → 返回 PIL Image"""
        image = self.remove_background(image_path)
        image = self.preprocess(image, target_size)
        return image
