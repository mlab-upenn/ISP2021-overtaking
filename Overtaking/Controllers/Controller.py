class Controller:
    def __init__(self, f1map, local_grid_size, resolution):
        self.map = f1map
        self.resolution = resolution
        self.local_grid_size = local_grid_size