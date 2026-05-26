"""OCR wrapper supporting EasyOCR and PaddleOCR backends for recruitment poster text extraction."""
import os
import yaml
import logging

logging.getLogger("easyocr").setLevel(logging.WARNING)


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class OCRExtractor:
    """Extract text from recruitment poster images.

    Supports two backends:
    - easyocr (default): Reliable Chinese + English OCR, simpler setup.
    - paddleocr: Higher accuracy Chinese OCR but heavier dependencies.
    """

    name = "ocr"

    def __init__(self):
        config = _load_config()
        ocr_cfg = config.get("ocr", {})
        self.engine = ocr_cfg.get("engine", "easyocr")
        self.lang = ocr_cfg.get("lang", "ch")
        self.use_gpu = ocr_cfg.get("use_gpu", False)
        self._reader = None

    @property
    def _engine(self):
        if self._reader is None:
            if self.engine == "paddleocr":
                self._init_paddleocr()
            else:
                self._init_easyocr()
        return self._reader

    def _init_easyocr(self):
        import easyocr
        langs = ["ch_sim", "en"] if self.lang == "ch" else ["en"]
        self._reader = easyocr.Reader(langs, gpu=self.use_gpu)

    def _init_paddleocr(self):
        from paddleocr import PaddleOCR
        self._reader = PaddleOCR(lang=self.lang)

    def extract(self, image_path: str, language: str = "zh") -> str:
        """Run OCR on an image and return concatenated text.

        Args:
            image_path: Path to the recruitment poster image.
            language: Hint for language (used by EasyOCR to select model).

        Returns:
            Concatenated text from all detected lines, joined by newline.
            Returns empty string if OCR fails or detects no text.
        """
        if not os.path.isfile(image_path):
            return ""

        try:
            engine = self._engine
        except Exception:
            return ""

        try:
            if self.engine == "paddleocr":
                raw = engine.ocr(image_path)
                if not raw or not isinstance(raw, list) or len(raw) == 0:
                    return ""
                page = raw[0]
                if not page:
                    return ""
                lines = [item[1][0] for item in page]
            else:
                results = engine.readtext(image_path)
                lines = [text for (_, text, _) in results]

            return "\n".join(lines)
        except Exception:
            return ""
