import sys, importlib

vendored = importlib.import_module(__name__ + ".diffusers")
sys.modules["diffusers"] = vendored

for sub in (
    "pipelines", "models", "schedulers", "utils", "configuration_utils",
    "image_processor", "image_processing_utils", "loaders", "quantizers"
):
    try:
        sys.modules[f"diffusers.{sub}"] = importlib.import_module(f"{__name__}.diffusers.{sub}")
    except ModuleNotFoundError:
        pass