import sys
import os
import numpy as np
import rasterio as rio
import geopandas as gpd
import inspect
import time

# While running tests, run on this source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# import packages
import pytest
from pygeoflood import pyGeoFlood
import shutil


# set input data paths
data_path = os.path.join(os.path.dirname(__file__), "test_data", "OC1mTest")
data_path_inputs =  os.path.join(os.path.dirname(__file__), "test_data", "OC1mTest_inputs")
# Copy the input datasets
# Copy all files from data_path_inputs to data_path
for fname in os.listdir(data_path_inputs):
    src = os.path.join(data_path_inputs, fname)
    dst = os.path.join(data_path, fname)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
test_dem_path = os.path.join(data_path, "OC1mTest.tif")
test_flowline_path = os.path.join(data_path, "Flowline.shp")
test_catchment_path = os.path.join(data_path, "Catchment.shp")
c_hand_dem_path = os.path.join(os.path.dirname(__file__), "test_data", 'HoustonTest', "houston_example_DEM_30m.tif")  # Example path for C_HAND DEM
ocean_E = 317540
ocean_N = 3272260
ocean_pixel = (ocean_E, ocean_N)  # Example pixel coordinates for ocean
ike_gage = 3.8

# Base testing class
class baseTestPyGeoFlood:
    """
    Base class for testing PyGeoFlood functionality.
    """
    def validate_raster_file(self, pgf, file_path, file_desc):
        """
        Validate a single raster file for quality and formatting.
        """ 
        if file_path is None:
            file_path = getattr(pgf, file_desc)

        # get reference information for file checking
        with rio.open(pgf.dem_path) as src:
            dem_data = src.read(1)
            dem_profile = src.profile
            reference_shape = dem_data.shape
            reference_transform = dem_profile['transform']
            reference_crs = dem_profile['crs']

        # Check if file exists
        assert os.path.exists(file_path), f"{file_desc} file does not exist: {file_path}"
        
        with rio.open(file_path) as src:
            data = src.read(1)
            profile = src.profile
            
        # Check 1: File has same dimensions as reference DEM
        assert data.shape == reference_shape, \
            f"{file_desc} has incorrect dimensions: {data.shape} vs expected {reference_shape}"
        
        # Check 2: File has same geospatial reference as input DEM
        assert profile['transform'] == reference_transform, \
            f"{file_desc} has incorrect transform"
        assert profile['crs'] == reference_crs, \
            f"{file_desc} has incorrect CRS"
            
        # Check 3: File is not entirely NaN
        if "float" in str(data.dtype).lower():
            valid_data = data[~np.isnan(data)]
            assert len(valid_data) > 0, \
                f"{file_desc} contains only NaN values"
            
            # Check 4: File is not entirely zero (for non-binary outputs)
            if file_desc not in ["binary HAND", "outlets", "basins", "channel network raster"]:
                non_zero_data = valid_data[valid_data != 0]
                assert len(non_zero_data) > 0, \
                    f"{file_desc} contains only zero values"
                    
        else:
            # For integer data types, check for nodata values
            if 'nodata' in profile and profile['nodata'] is not None:
                valid_data = data[data != profile['nodata']]
            else:
                valid_data = data
            assert len(valid_data) > 0, \
                f"{file_desc} contains only nodata values"
                
        # Check 5: Data type is reasonable
        assert str(data.dtype) in ['float32', 'float64', 'int16', 'int32', 'uint8', 'uint16'], \
            f"{file_desc} has unexpected data type: {data.dtype}"

    def validate_vector_file(self, file_path, file_desc):
        """
        Validate a single vector file 
        """
        # Ensure that vector file can be opened and has a minimum of one feature
        vector = gpd.read_file(file_path)
        assert vector.shape[0] > 0, f"{file_desc} has no features"

    def validate_csv(self, file_path, file_desc):
        """
        Validate a single CSV file for quality and formatting.
        """
        # Check if the file exists
        assert os.path.exists(file_path), f"{file_desc} file does not exist: {file_path}"
        
        # Read the CSV file
        try:
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            assert data.size > 0, f"{file_desc} is empty"
        except Exception as e:
            raise AssertionError(f"Error reading {file_desc}: {e}")
        
    def validate_method(self, func, desc, kwargs=None):
        """
        Validate a method by checking if it runs without errors.
        """
        # try:
        #     func(**(kwargs or {}))
        try:
            if kwargs:
                # Get the function signature to filter valid parameters
                sig = inspect.signature(func)
                valid_params = sig.parameters.keys()
                
                # Filter kwargs to only include valid parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                
                func(**filtered_kwargs)
            else:
                func()
            function_ran_successfully = True
        except Exception as e:
            pytest.fail(f"{desc}() failed with error: {e}")

        # Assert that the function completed successfully
        assert function_ran_successfully, f"{desc} did not complete successfully"


    def standard_test(self, func, desc, kwargs):
        expected_error = kwargs.pop('expected_error', None)
        if expected_error:
            with pytest.raises(expected_error):
                func(**kwargs)
        else:
            self.validate_method(func, desc, kwargs=kwargs)


# test class for PyGeoFlood functionality
class TestPyGeoFlood(baseTestPyGeoFlood):
    @pytest.fixture(scope="class")
    def pgf(self):
        # Initialize PyGeoFlood instance
        pgf = pyGeoFlood(dem_path=test_dem_path)
        pgf.flowline_path = test_flowline_path
        pgf.catchment_path = test_catchment_path
        # Need to set custom_flowline_path for testing rasterize_custom_flowline
        pgf.custom_flowline_path = f"{data_path}/custom_flowline"
        pgf.streamflow_forecast_path = f"{data_path}/nwm.t2025070718z.analysis_assim.channel_rt.tm00.conus.nc"
        return pgf

    def test_initialization(self, pgf):
        # Test if the instance is initialized correctly
        assert isinstance(pgf, pyGeoFlood)
        
        # Test validation of the input DEM file
        self.validate_raster_file(pgf, test_dem_path, "input DEM")

    def test_to_paths(self, pgf):
        # Test conversion to paths
        self.validate_method(pgf.to_paths, "to_paths", kwargs={'file_path': f'{data_path}/paths'})


    @pytest.mark.parametrize('custom_dem, custom_path, method, expected_error', [
        (None, None, 'PeronaMalik2', False),  
        (test_dem_path, f"{data_path}/filtered_custom.tif", 'PeronaMalik1', False),
        (test_dem_path, None, 'PeronaMalik2', ValueError)])
    def test_apply_nonlinear_filter(self, 
                                    pgf, 
                                    custom_dem, 
                                    custom_path, 
                                    method, 
                                    expected_error):
        
        self.standard_test(pgf.apply_nonlinear_filter, 
                          "apply_nonlinear_filter",
                          kwargs={'custom_dem': custom_dem,
                                  'custom_path': custom_path,
                                  'method': method,
                                  'expected_error': expected_error})

    @pytest.mark.parametrize('custom_filtered_dem, custom_path, expected_error', [
        (None, None, False),  # Use default filtered DEM path
        (test_dem_path, f"{data_path}/slope_custom.tif", False),
        (test_dem_path, None, ValueError)])
    def test_calculate_slope(self,
                             pgf,
                             custom_filtered_dem,
                             custom_path,
                             expected_error):
        
        self.standard_test(pgf.calculate_slope,
                            "calculate_slope",
                            kwargs={'custom_filtered_dem': custom_filtered_dem,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_filtered_dem, custom_path, method, expected_error', [
        (None, None, 'geometric', False),  # Use default filtered DEM path
        (test_dem_path, f"{data_path}/curvature_custom.tif", 'laplacian', False),
        (test_dem_path, None, 'geometric', ValueError)])
    def test_calculate_curvature(self,
                                 pgf,
                                 custom_filtered_dem,
                                 custom_path,
                                 method,
                                 expected_error):
        
        self.standard_test(pgf.calculate_curvature,
                            "calculate_curvature",
                            kwargs={'custom_filtered_dem': custom_filtered_dem,
                                    'custom_path': custom_path,
                                    'method': method,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_filtered_dem, custom_path, expected_error', [
        (None, None, False),  # Use default filtered DEM path
        (test_dem_path, f"{data_path}/filled_custom.tif", False),
        (test_dem_path, None, ValueError)])
    def test_fill_depressions(self,
                              pgf,
                              custom_filtered_dem,
                              custom_path,
                              expected_error):
        
        self.standard_test(pgf.fill_depressions,
                            "fill_depressions",
                            kwargs={'custom_filtered_dem': custom_filtered_dem,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_filled_dem, custom_path, expected_error', [
        (None, None, False),  # Use default filled DEM path
        (test_dem_path, f"{data_path}/mfd_fac_custom.tif", False),
        (test_dem_path, None, ValueError)])
    def test_calculate_mfd_flow_accumulation(self, 
                                             pgf,
                                             custom_filled_dem,
                                             custom_path,
                                             expected_error):
        
        self.standard_test(pgf.calculate_mfd_flow_accumulation,
                            "calculate_mfd_flow_accumulation",
                            kwargs={'custom_filled_dem': custom_filled_dem,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_filled_dem, custom_filtered_dem, custom_path, expected_error', [
        (None, None, None, False),  # Use default filled and filtered DEM paths
        (f"{data_path}/filled_custom.tif", f"{data_path}/filtered_custom.tif", f"{data_path}/d8_fdr_custom.tif", False),
        (test_dem_path, None, None, ValueError),
        (None, f"{data_path}/filtered_custom.tif", None, ValueError)])
    def test_calculate_d8_flow_direction(self,
                                         pgf,
                                         custom_filled_dem,
                                         custom_filtered_dem,
                                         custom_path,
                                         expected_error):
        
        self.standard_test(pgf.calculate_d8_flow_direction,
                            "calculate_d8_flow_direction",
                            kwargs={'custom_filled_dem': custom_filled_dem,
                                    'custom_filtered_dem': custom_filtered_dem,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_d8_fdr, custom_path, expected_error', [
        (None, None, False), 
        (f"{data_path}/d8_fdr_custom.tif", f"{data_path}/outlets_custom.tif", False),
        (f"{data_path}/d8_fdr_custom.tif", None, ValueError)])
    def test_find_outlets(self,
                        pgf,
                        custom_d8_fdr,
                        custom_path,
                        expected_error):
        
        self.standard_test(pgf.find_outlets,
                            "find_outlets",
                            kwargs={'custom_d8_fdr': custom_d8_fdr,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_d8_fdr, custom_path, expected_error', [
        (None, None, False),  # Use default D8 flow direction path
        (f"{data_path}/d8_fdr_custom.tif", f"{data_path}/basins_custom.tif", False),
        (f"{data_path}/d8_fdr_custom.tif", None, ValueError)])
    def test_delineate_basins(self,
                              pgf,
                              custom_d8_fdr,
                              custom_path,
                              expected_error):
        
        self.standard_test(pgf.delineate_basins,
                            "delineate_basins",
                            kwargs={'custom_d8_fdr': custom_d8_fdr,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_curvature, custom_mfd_fac, custom_path, fac_threshold, write_flow_skeleton, write_curvature_skeleton, expected_error', [
        (None, None, None, 3000, False, False, False),  
        (f"{data_path}/curvature_custom.tif", f"{data_path}/mfd_fac_custom.tif", f"{data_path}/skeleton_custom.tif", 3000, True, True, False),
        (f"{data_path}/curvature_custom.tif", None, None, 3000, False, False, ValueError),
        (None, f"{data_path}/mfd_fac_custom.tif", None, 3000, True, True, ValueError)])
    def test_define_skeleton(self,
                             pgf,
                             custom_curvature,
                             custom_mfd_fac,
                             custom_path,
                             fac_threshold,
                             write_flow_skeleton,
                             write_curvature_skeleton,
                             expected_error):

        self.standard_test(pgf.define_skeleton,
                            "define_skeleton",
                            kwargs={'custom_curvature': custom_curvature,
                                    'custom_mfd_fac': custom_mfd_fac,
                                    'custom_path': custom_path,
                                    'fac_threshold': fac_threshold,
                                    'write_flow_skeleton': write_flow_skeleton,
                                    'write_curvature_skeleton': write_curvature_skeleton,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_curvature, custom_mfd_fac, custom_outlets, custom_basins, custom_combined_skeleton, custom_filtered_dem, custom_path, write_cost_function, basin_elements, area_threshold, normalize_curvature, local_cost_min, expected_error', [
        (None, None, None, None, None, None, None, False, 2, 0.1, True, None, False),
        (f"{data_path}/curvature_custom.tif", f"{data_path}/mfd_fac_custom.tif", f"{data_path}/outlets_custom.tif", f"{data_path}/basins_custom.tif", f"{data_path}/OC1mTest_combined_skeleton.tif", f"{data_path}/filtered_custom.tif", f"{data_path}/geodesic_distance_custom.tif", True, 2, 0.1, True, None, False),
        (f"{data_path}/curvature_custom.tif", None, None, None, None, None, None, False, 2, 0.1, True, None,  ValueError),
        (None, f"{data_path}/mfd_fac_custom.tif", None, None, None, None, None, False, 2, 0.1, True, None, ValueError),
        (None, None, f"{data_path}/outlets_custom.tif", None, None, None, None, False, 2, 0.1, True, None, ValueError),
        (None, None, None, f"{data_path}/basins_custom.tif", None, None, None, False, 2, 0.1, True, None, ValueError),
        (None, None, None, None, f"{data_path}/OC1mTest_combined_skeleton.tif", None, None, False, 2, 0.1, True, None, ValueError),
        (None, None, None, None, None, f"{data_path}/filtered_custom.tif", None, False, 2, 0.1, True, None, ValueError)])
    def test_calculate_geodesic_distance(self,
                                         pgf,
                                         custom_curvature,
                                         custom_mfd_fac,
                                         custom_outlets,
                                         custom_basins,
                                         custom_combined_skeleton,
                                         custom_filtered_dem,
                                         custom_path,
                                         write_cost_function,
                                         basin_elements,
                                         area_threshold,
                                         normalize_curvature,
                                         local_cost_min,
                                         expected_error):
        
        self.standard_test(pgf.calculate_geodesic_distance,
                            "calculate_geodesic_distance",
                            kwargs={'custom_curvature': custom_curvature,
                                    'custom_mfd_fac': custom_mfd_fac,
                                    'custom_outlets': custom_outlets,
                                    'custom_basins': custom_basins,
                                    'custom_combined_skeleton': custom_combined_skeleton,
                                    'custom_filtered_dem': custom_filtered_dem,
                                    'custom_path': custom_path,
                                    'write_cost_function': write_cost_function,
                                    'basin_elements': basin_elements,
                                    'area_threshold': area_threshold,
                                    'normalize_curvature': normalize_curvature,
                                    'local_cost_min': local_cost_min,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_combined_skeleton, custom_geodesic_distance, custom_path, channel_head_median_dist, vector_extension, max_channel_heads, expected_error', [
        (None, None, None, 30, 'shp', 10000, False),  
        (f"{data_path}/OC1mTest_combined_skeleton.tif", f"{data_path}/OC1mTest_geodesic_distance.tif", f"{data_path}/channel_heads_custom", 30, 'shp', 10000, False),
        (f"{data_path}/OC1mTest_combined_skeleton.tif", None, None, 30, 'shp', 10000, ValueError),
        (None, f"{data_path}/geodesic_distance_custom.tif", None, 30, 'shp', 10000, ValueError),
        ])
    def test_identify_channel_heads(self,
                                    pgf,
                                    custom_combined_skeleton,
                                    custom_geodesic_distance,
                                    custom_path,
                                    channel_head_median_dist,
                                    vector_extension,
                                    max_channel_heads,
                                    expected_error):
        
        self.standard_test(pgf.identify_channel_heads,
                            "identify_channel_heads",
                            kwargs={'custom_combined_skeleton': custom_combined_skeleton,
                                    'custom_geodesic_distance': custom_geodesic_distance,
                                    'custom_path': custom_path,
                                    'channel_head_median_dist': channel_head_median_dist,
                                    'vector_extension': vector_extension,
                                    'max_channel_heads': max_channel_heads,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_flowline, custom_path, expected_error', [
        (None, None, False),
        (test_flowline_path, f"{data_path}/endpoints_custom", False),
        (test_flowline_path, None, ValueError),  
    ])
    def test_find_endpoints(self,
                            pgf,
                            custom_flowline,
                            custom_path,
                            expected_error):
        
        self.standard_test(pgf.find_endpoints,
                            "find_endpoints",
                            kwargs={'custom_flowline': custom_flowline,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_dem, custom_flowline, custom_path, expected_error', [
        (None, None, None, False),  
        (test_dem_path, test_flowline_path, f"{data_path}/binary_hand_custom.tif", False),
        (test_dem_path, None, None, ValueError),
        (None, test_flowline_path, None, ValueError),
    ])
    def test_calculate_binary_hand(self,
                                   pgf,
                                   custom_dem,
                                   custom_flowline,
                                   custom_path,
                                   expected_error):
        
        self.standard_test(pgf.calculate_binary_hand,
                            "calculate_binary_hand",
                            kwargs={'custom_dem': custom_dem,
                                    'custom_flowline': custom_flowline,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_flowline, custom_path, expected_error', [
        (test_flowline_path, f"{data_path}/custom_flowline", False),
        (test_flowline_path, None, ValueError)])
    def test_rasterize_custom_flowline(self,pgf, custom_flowline, custom_path, expected_error):
        self.standard_test(pgf.rasterize_custom_flowline,
                            "rasterize_custom_flowline",
                            kwargs={'custom_flowline': custom_flowline,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_flowline, ' \
                            'custom_curvature, ' \
                            'custom_mfd_fac, ' \
                            'custom_endpoints, ' \
                            'custom_binary_hand, ' \
                            'custom_path, ' \
                            'retrace_flowline, ' \
                            'vector_extension, ' \
                            'write_cost_function, ' \
                            'use_custom_flowline, ' \
                            'no_flowline, ' \
                            'custom_weight_curvature, ' \
                            'custom_weight_mfd_fac, ' \
                            'custom_weight_binary_hand, ' \
                            'custom_weight_custom_flowline, ' \
                            'expected_error', [
    (None, None, None, None, None, None, True, 'shp', False, False, False, None, None, None, None, False),
    (test_flowline_path, f"{data_path}/curvature_custom.tif",f"{data_path}/mfd_fac_custom.tif", f"{data_path}/endpoints_custom.csv", f"{data_path}/binary_hand_custom.tif", f"{data_path}/channel_network_custom", False, 'shp', True, True, True, 1, 1, 1, 1,False),
    (test_flowline_path, None, None, None, None, None, False, 'shp', False, False, False, None, None, None, None, ValueError),
    (None, f"{data_path}/curvature_custom.tif", None, None, None, None, True, 'shp', False, False, False, None, None, None, None, ValueError),
    (None, None, f"{data_path}/mfd_fac_custom.tif", None, None, None, True, 'shp', False, False, False, None, None, None, None, ValueError),
    (None, None, None, f"{data_path}/endpoints_custom.csv", None, None, True, 'shp', False, False, False, None, None, None, None, ValueError),
    (None, None, None, None, f"{data_path}/binary_hand_custom.tif", None, True, 'shp', False, False, False, None, None, None, None, ValueError)
    ])
    def test_extract_channel_network(self,
                                      pgf,
                                      custom_flowline,
                                      custom_curvature,
                                      custom_mfd_fac,
                                      custom_endpoints,
                                      custom_binary_hand,
                                      custom_path,
                                      retrace_flowline,
                                      vector_extension,
                                      write_cost_function,
                                      use_custom_flowline,
                                      no_flowline,
                                      custom_weight_curvature,
                                      custom_weight_mfd_fac,
                                      custom_weight_binary_hand,
                                      custom_weight_custom_flowline,
                                      expected_error):
        
        self.standard_test(pgf.extract_channel_network,
                            "extract_channel_network",
                            kwargs={'custom_flowline': custom_flowline,
                                    'custom_curvature': custom_curvature,
                                    'custom_mfd_fac': custom_mfd_fac,
                                    'custom_endpoints': custom_endpoints,
                                    'custom_binary_hand': custom_binary_hand,
                                    'custom_path': custom_path,
                                    'retrace_flowline': retrace_flowline,
                                    'vector_extension': vector_extension,
                                    'write_cost_function': write_cost_function,
                                    'use_custom_flowline': use_custom_flowline,
                                    'no_flowline': no_flowline,
                                    'custom_weight_curvature': custom_weight_curvature,
                                    'custom_weight_mfd_fac': custom_weight_mfd_fac,
                                    'custom_weight_binary_hand': custom_weight_binary_hand,
                                    'custom_weight_custom_flowline': custom_weight_custom_flowline,
                                    'expected_error': expected_error})
        
    @pytest.mark.parametrize('custom_filled_dem, custom_channel_network_raster, custom_path, expected_error', [
        (None, None, None, False),
        (test_dem_path, f"{data_path}/channel_network_custom.tif", f"{data_path}/hand_custom.tif", False),
        (test_dem_path, None, None, ValueError),
        (None, f"{data_path}/channel_network_custom.tif", None, ValueError)])  
    def test_calculate_hand(self,
                            pgf,
                            custom_filled_dem,
                            custom_channel_network_raster,
                            custom_path,
                            expected_error):
        
        self.standard_test(pgf.calculate_hand,
                            "calculate_hand",
                            kwargs={'custom_filled_dem': custom_filled_dem,
                                    'custom_channel_network_raster': custom_channel_network_raster,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_channel_network_vector, custom_path, vector_extension, segment_length, expected_error', [
        (None, None, 'shp', 1000, False), 
        (f"{data_path}/channel_network_custom.shp", f"{data_path}/segmented_channel_network_custom", 'shp', 1000, False),
        (f"{data_path}/channel_network_custom.shp", None, 'shp', 1000, ValueError),
    ])
    def test_segment_channel_network(self,
                                pgf,
                                custom_channel_network_vector,
                                custom_path,
                                vector_extension,
                                segment_length,
                                expected_error):

        self.standard_test(pgf.segment_channel_network,
                           "segment_channel_network",
                           kwargs={'custom_channel_network_vector': custom_channel_network_vector,
                                   'custom_path': custom_path,
                                   'vector_extension': vector_extension,
                                   'segment_length': segment_length,
                                   'expected_error': expected_error})

    @pytest.mark.parametrize('custom_segmented_channel_network, custom_d8_fdr, custom_path, expected_error', [
        (None, None, None, False),  
        (f"{data_path}/segmented_channel_network_custom.shp", f"{data_path}/d8_fdr_custom.tif", f"{data_path}/segment_catchments_custom", False),
        (f"{data_path}/segmented_channel_network_custom.shp", None, None, ValueError),
        (None, f"{data_path}/d8_fdr_custom.tif", None, ValueError)
    ])
    def test_delineate_segment_catchments(self,
                                          pgf,
                                          custom_segmented_channel_network,
                                          custom_d8_fdr,
                                          custom_path,
                                          expected_error):
        
        self.standard_test(pgf.delineate_segment_catchments,
                            "delineate_segment_catchments",
                            kwargs={'custom_segmented_channel_network': custom_segmented_channel_network,
                                    'custom_d8_fdr': custom_d8_fdr,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_segment_catchments_raster, custom_path, expected_error', [
        (None, None, False),  # Use default segment catchments raster path  
        (f"{data_path}/segment_catchments_custom.tif", f"{data_path}/segment_catchments_vector_custom.shp", False),
        (f"{data_path}/segment_catchments_custom.tif", None, ValueError),])
    def test_vectorize_segment_catchments(self,
                                          pgf,
                                          custom_segment_catchments_raster,
                                          custom_path,
                                          expected_error):
        
        self.standard_test(pgf.vectorize_segment_catchments,
                            "vectorize_segment_catchments",
                            kwargs={'custom_segment_catchments_raster': custom_segment_catchments_raster,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_dem, custom_segmented_channel_network, custom_segment_catchments_raster, custom_catchments, custom_hand, custom_path, min_slope, max_stage, incr_stage, custom_roughness_path, expected_error', [
        (None, None, None, None, None, None, 0.000001, 20, 0.1, None, False),  
        (test_dem_path, f"{data_path}/segmented_channel_network_custom.shp", f"{data_path}/segment_catchments_custom.tif", test_catchment_path, f"{data_path}/hand_custom.tif", f"{data_path}/src_custom", 0.000001, 20, 0.1, None, False),
        (test_dem_path, None, None, None, None, None, 0.000001, 20, 0.1, None, ValueError),
        (None, f"{data_path}/segmented_channel_network_custom.shp", None, None, None, None, 0.000001, 20, 0.1, None, ValueError),
        (None, None, f"{data_path}/segment_catchments_custom.tif", None, None, None, 0.000001, 20, 0.1, None, ValueError),
        (None, None, None, f"{data_path}/catchments_custom.shp", None, None, 0.000001, 20, 0.1, None, ValueError),
        (None, None, None, None, f"{data_path}/hand_custom.tif", None, 0.000001, 20, 0.1, None, ValueError)])
    def test_calculate_src(self,
                           pgf,
                           custom_dem,
                           custom_segmented_channel_network,
                           custom_segment_catchments_raster,
                           custom_catchments,
                           custom_hand,
                           custom_path,
                           min_slope,
                           max_stage,
                           incr_stage,
                           custom_roughness_path,
                           expected_error):
        
        self.standard_test(pgf.calculate_src,
                            "calculate_src",
                            kwargs={'custom_dem': custom_dem,
                                    'custom_segmented_channel_network': custom_segmented_channel_network,
                                    'custom_segment_catchments_raster': custom_segment_catchments_raster,
                                    'custom_catchments': custom_catchments,
                                    'custom_hand': custom_hand,
                                    'custom_path': custom_path,
                                    'min_slope': min_slope,
                                    'max_stage': max_stage,
                                    'incr_stage': incr_stage,
                                    'custom_roughness_path': custom_roughness_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_src, custom_streamflow_forecast_path, custom_path, custom_Q, expected_error', [  
        (f"{data_path}/src_custom.csv", None, f"{data_path}/flood_stage_500_custom", 500, False),
        (f"{data_path}/src_custom.csv", f"{data_path}/nwm.t2025070718z.analysis_assim.channel_rt.tm00.conus.nc", f"{data_path}/flood_stage_NWM_custom", 0, False),
        (f"{data_path}/src_custom.csv", None, None, 500, ValueError),
        (None, f'{data_path}/nwm.t2025070718z.analysis_assim.channel_rt.tm00.conus.nc', None, None, ValueError),
        (None, None, None, None, False),  
        ])
    def test_calculate_flood_stage(self,
                                   pgf,
                                   custom_src,
                                   custom_streamflow_forecast_path,
                                   custom_path,
                                   custom_Q,
                                   expected_error):
        
        self.standard_test(pgf.calculate_flood_stage,
                            "calculate_flood_stage",
                            kwargs={'custom_src': custom_src,
                                    'custom_streamflow_forecast_path': custom_streamflow_forecast_path,
                                    'custom_path': custom_path,
                                    'custom_Q': custom_Q,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('custom_hand, custom_flood_stage, custom_segment_catchments_raster, custom_path, expected_error', [
        (None, None, None, None, False), 
        (f"{data_path}/hand_custom.tif", f"{data_path}/flood_stage_500_custom", f"{data_path}/segment_catchments_custom.tif", f"{data_path}/inundation_custom", False),
        (f"{data_path}/hand_custom.tif", None, None, None, ValueError),
        (None, f"{data_path}/flood_stage_500_custom.tif", None, None, ValueError),
        (None, None, f"{data_path}/segment_catchments_custom.tif", None, ValueError),
        ])
    def test_inundate(self,
                      pgf,
                      custom_hand,
                      custom_flood_stage,
                      custom_segment_catchments_raster,
                      custom_path,
                      expected_error):
        
        self.standard_test(pgf.inundate,
                            "inundate",
                            kwargs={'custom_hand': custom_hand,
                                    'custom_flood_stage': custom_flood_stage,
                                    'custom_segment_catchments_raster': custom_segment_catchments_raster,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    @pytest.mark.parametrize('uniform_depth, gridded_depth, custom_dem, overwrite_dephier, custom_path, expected_error', [
        (0.5, None, None, False, None, False),
        (0.5, test_dem_path, test_dem_path, True, f"{data_path}/fsm_inundation_custom.tif", False),
        (0.5, None, test_dem_path, False, None, ValueError),])
    def test_fill_spill_merge(self,
                              pgf,
                              uniform_depth,
                              gridded_depth,
                              custom_dem,
                              overwrite_dephier,
                              custom_path,
                              expected_error):
        self.standard_test(pgf.fill_spill_merge,
                            "fill_spill_merge",
                            kwargs={'uniform_depth': uniform_depth,
                                    'gridded_depth': gridded_depth,
                                    'custom_dem': custom_dem,
                                    'overwrite_dephier': overwrite_dephier,
                                    'custom_path': custom_path,
                                    'expected_error': expected_error})

    def test_fim_workflow(self,pgf):
        self.validate_method(pgf.run_fim_workflow, "run_fim_workflow")

    def test_all_data_files(self, pgf):
        for fname in os.listdir(data_path):
            fpath = os.path.join(data_path, fname)
            if fname.lower().endswith('.tif'):
                self.validate_raster_file(pgf, fpath, fname)
            elif fname.lower().endswith('.csv'):
                self.validate_csv(fpath, fname)
            elif fname.lower().endswith('.shp'):
                self.validate_vector_file(fpath, fname)



  

# Test class for C_HAND calculations
class TestPyGeoFlood_c_hand(baseTestPyGeoFlood):
    @pytest.fixture(scope="class")
    def pgf(self):
        # Initialize PyGeoFlood instance
        pgf = pyGeoFlood(dem_path=c_hand_dem_path)
        # pgf.flowline_path = test_flowline_path
        # pgf.catchment_path = test_catchment_path
        return pgf

    def test_c_hand(self, pgf):
        # Test calculating C_HAND
        self.validate_method(pgf.c_hand, 
                             "calculate_c_hand", 
                             kwargs={'ocean_coords': ocean_pixel, 
                                     'gage_el': ike_gage})
        
        # Validate the C_HAND file
        self.validate_raster_file(pgf, pgf.coastal_inundation_path, "C_HAND")
        
        
# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_data():
#     yield  # Run tests first
#     # Remove created data files to reduce the number of files being stored in the testing suite
#     for fname in os.listdir(data_path):
#         fpath = os.path.join(data_path, fname)
#         if os.path.isfile(fpath):
#             os.remove(fpath)





