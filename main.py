from src.visualization import visualize_slices
from src.data import MedicalDataset
from src.utils import *
from src.model.unet_model import UNet
from src.settings import Settings, Paths

settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()

