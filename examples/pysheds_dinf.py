# import numpy as np
from pysheds.grid import Grid

# read and process DEM
dem_path = "data/OC1mTest_filled.tif"
grid = Grid.from_raster(dem_path)
dem = grid.read_raster(dem_path)
channels_path = "data/OC1mTest_channel_network.tif"
channels = grid.read_raster(channels_path)
# calculate and write flow direction rasters
fdir_dinf = grid.flowdir(dem, routing="dinf")
hand_dinf = grid.compute_hand(fdir_dinf, dem, channels == 1, routing="dinf")
grid.to_raster(hand_dinf, "data/pysheds_hand_dinf.tif", blockxsize=16, blockysize=16)
# grid.to_raster(fdir_dnf, "data/pysheds_dinf.tif", blockxsize=16, blockysize=16)
