import requests
from tqdm import tqdm
import logging
import sys

class ColoredLogger:
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'RESET': '\033[0m'
    }

    LEVEL_COLORS = {
        'DEBUG': COLORS['BLUE'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['MAGENTA']
    }

    def __init__(self, name="MY-APP"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.app_name = name
        self.logger.propagate = False
        self.logger.handlers = []
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                level_color = ColoredLogger.LEVEL_COLORS.get(record.levelname, '')
                colored_levelname = f"{level_color}{record.levelname}{ColoredLogger.COLORS['RESET']}"
                colored_name = f"{ColoredLogger.COLORS['BLUE']}{record.name}{ColoredLogger.COLORS['RESET']}"
                record.levelname = colored_levelname
                record.name = colored_name
                return super().format(record)
        
        formatter = ColoredFormatter('[%(name)s|%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(f"{self.COLORS['GREEN']}{message}{self.COLORS['RESET']}")

    def warning(self, message):
        self.logger.warning(f"{self.COLORS['YELLOW']}{message}{self.COLORS['RESET']}")

    def error(self, message):
        self.logger.error(f"{self.COLORS['RED']}{message}{self.COLORS['RESET']}")

def download_file(url, save_path):
    GREEN = '\033[92m'
    RESET = '\033[0m'
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm(desc=save_path, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, colour='green', bar_format=f'{GREEN}{{l_bar}}{{bar}}{RESET}{GREEN}{{r_bar}}{RESET}') as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

# WICHTIG: Hier wird 'model_scale' übergeben
def get_final_resolutions(width, height, resize_to, model_scale=4):
    final_width = None
    final_height = None
    aspect_ratio = float(width/height)

    match resize_to:
        case "HD":
            final_width = 1280
            final_height = 720
        case "FHD":
            final_width = 1920
            final_height = 1080
        case "2k":
            final_width = 2560
            final_height = 1440
        case "4k":
            final_width = 3840
            final_height = 2160
        case "none":
            final_width = width * model_scale
            final_height = height * model_scale
        case _:
            resize_factor = float(resize_to.split('x')[0])
            final_width = width*resize_factor
            final_height = height*resize_factor

    if aspect_ratio == 1.0:
        final_width = final_height
    if aspect_ratio < 1.0 and resize_to not in ("none", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"):
        temp = final_width
        final_width = final_height
        final_height = temp

    return (int(final_width), int(final_height))
