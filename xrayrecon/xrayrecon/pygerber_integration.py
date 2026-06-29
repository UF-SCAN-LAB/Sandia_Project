
import numpy as np
from typing import Optional
import sys
from PIL import Image

class PyGerberParser:
    def __init__(self):
        pass

class PyGerberRasterizer:
    def __init__(self, dpi=1000):
        self.dpi = dpi

    def rasterize(self, content: str) -> np.ndarray:
        try:
            from pygerber.gerberx3.api import Layer, ColorScheme, LayerParams
            from pygerber.backend import Rasterized2DBackend
            from pygerber.backend.rasterized_2d.backend_cls import Rasterized2DBackendOptions
            
            class CustomLayer(Layer):
                def __init__(self, options: LayerParams, dpi: int):
                    self.custom_dpi = dpi
                    super().__init__(options)
                def _create_backend(self):
                    return Rasterized2DBackend(
                        options=Rasterized2DBackendOptions(
                            dpi=self.custom_dpi,
                            color_scheme=ColorScheme.COPPER_ALPHA, 
                        )
                    )
            
            layer = CustomLayer(options=LayerParams(source_code=content), dpi=self.dpi)
            result = layer.render()
            
            # Access internal handle
            if hasattr(result, '_result_handle'):
                handle = result._result_handle
                # print(f"DEBUG Internal Handle Dir: {dir(handle)}")
                
                # Check for image access
                if hasattr(handle, 'result'):
                    # Sometimes handle.result is the image
                    img = handle.result
                elif hasattr(handle, 'get_image'):
                    img = handle.get_image()
                else:
                    # Check backend specific handle
                    # Step 209 said Rasterized2DBackend has get_result_handle
                    # The handle is Rasterized2DResultHandle
                    # I'll print to find out
                    print(f"DEBUG Handle Dir: {dir(handle)}")
                    raise AttributeError("Cannot find image in internal handle")
            else:
                 raise AttributeError("Cannot find _result_handle")

            # Verify img is PIL Image
            if not isinstance(img, Image.Image):
                 print(f"DEBUG Retrieved object type: {type(img)}")
                 # If it's not image, maybe it's something else
                 raise AttributeError("Retrieved object is not PIL Image")

            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                mask = np.array(a)
            else:
                mask = np.array(img.convert("L")) 
            return mask
            
        except ImportError as e:
            print(f"PyGerber Import Error: {e}")
            return np.zeros((10,10), dtype=np.uint8)
        except Exception as e:
            print(f"PyGerber Rasterization Failed: {e}")
            return np.zeros((10,10), dtype=np.uint8)
