import sys
import os

# While running tests, run on this source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# import packages
import pytest
from pygeoflood import pyGeoFlood

# set input data paths
data_path = os.path.join(os.path.dirname(__file__), "test_data")
test_dem_path = os.path.join(data_path, "OC1mTest.tif")
test_flowline_path = os.path.join(data_path, "Flowline.shp")
test_catchment_path = os.path.join(data_path, "Catchment.shp")


# test class for PyGeoFlood functionality
class TestPyGeoFlood:
    @pytest.fixture(scope="class")
    def pgf(self):
        # Initialize PyGeoFlood instance
        pgf = pyGeoFlood(dem_path=test_dem_path)
        pgf.flowline_path = test_flowline_path
        pgf.catchment_path = test_catchment_path
        return pgf

    def test_initialization(self, pgf):
        # Test if the instance is initialized correctly
        assert isinstance(pgf, pyGeoFlood)
        

