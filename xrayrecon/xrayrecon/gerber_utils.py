
import re
import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Aperture:
    code: int
    type: str # 'C', 'R', 'O'
    params: List[float]

@dataclass
class GraphicObject:
    type: str # 'LINE', 'FLASH'
    aperture: Aperture
    start: Tuple[float, float]
    end: Tuple[float, float]

class GerberLayer:
    def __init__(self, name="Unknown"):
        self.name = name
        self.objects: List[GraphicObject] = []
        self.apertures: Dict[int, Aperture] = {}
        self.bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
        
        # Raster support
        self.raster_mask: Optional[np.ndarray] = None
        self.raster_resolution_xy: Optional[float] = None 

    def update_bounds(self, x, y):
        self.bounds[0] = min(self.bounds[0], x)
        self.bounds[1] = min(self.bounds[1], y)
        self.bounds[2] = max(self.bounds[2], x)
        self.bounds[3] = max(self.bounds[3], y)

class SimpleGerberParser:
    def __init__(self):
        self.layer = GerberLayer()
        self.current_aperture: Optional[Aperture] = None
        self.current_pos = (0.0, 0.0)
        self.unit_scale = 1.0
        
    def parse(self, content: str) -> GerberLayer:
        commands = re.split(r'[*]', content)
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd: continue
            
            clean_cmd = cmd.strip('% \n\r\t')
            # Use clean_cmd for routing to handle residue chars from split
            
            if clean_cmd.startswith('ADD') or clean_cmd.startswith('MO'):
                self._parse_macro(clean_cmd)
            elif 'X' in clean_cmd or 'Y' in clean_cmd or clean_cmd.startswith('D'):
                self._parse_coordinate(clean_cmd) # Pass clean cmd
                
        return self.layer

    def _parse_macro(self, cmd):
        if cmd.startswith('MOIN'):
            self.unit_scale = 25.4
        elif cmd.startswith('MOMM'):
            self.unit_scale = 1.0
        
        match = re.match(r'ADD(\d+)([A-Za-z])[,](.*)', cmd)
        if match:
            d_code = int(match.group(1))
            shape_type = match.group(2)
            try:
                params_str = match.group(3).strip('%') 
                params = [float(x) * self.unit_scale for x in params_str.split('X')]
                ap = Aperture(d_code, shape_type, params)
                self.layer.apertures[d_code] = ap
            except ValueError:
                pass 

    def _parse_coordinate(self, cmd):
        x_match = re.search(r'X([\d.-]+)', cmd)
        y_match = re.search(r'Y([\d.-]+)', cmd)
        d_match = re.search(r'D(\d+)', cmd)
        
        target_x = float(x_match.group(1)) * self.unit_scale if x_match else self.current_pos[0]
        target_y = float(y_match.group(1)) * self.unit_scale if y_match else self.current_pos[1]
        
        operation = 0 
        if d_match:
            d_val = int(d_match.group(1))
            if d_val == 1: operation = 1 
            elif d_val == 2: operation = 2 
            elif d_val == 3: operation = 3 
            elif d_val >= 10: 
                self.current_aperture = self.layer.apertures.get(d_val)
                return 

        if operation == 1: 
             if self.current_aperture:
                 self.layer.objects.append(GraphicObject(
                     type='LINE',
                     aperture=self.current_aperture,
                     start=self.current_pos,
                     end=(target_x, target_y)
                 ))
                 self.layer.update_bounds(self.current_pos[0], self.current_pos[1])
                 self.layer.update_bounds(target_x, target_y)

        elif operation == 3: 
            if self.current_aperture:
                 self.layer.objects.append(GraphicObject(
                     type='FLASH',
                     aperture=self.current_aperture,
                     start=(target_x, target_y),
                     end=(target_x, target_y)
                 ))
                 self.layer.update_bounds(target_x, target_y)

        self.current_pos = (target_x, target_y)
