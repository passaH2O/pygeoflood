import geopandas as gpd
import fiona
import inspect
import json
import numpy as np
import pandas as pd
import rasterio as rio
import sys
import os
from . import tools as t
from os import PathLike
from pathlib import Path
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling


class PyGeoFlood(object):
    """A class to implement the height above nearest drainage method"""

    @property
    def dem_path(self) -> str | PathLike:
        return getattr(self, "_dem_path")

    @dem_path.setter
    def dem_path(self, value: str | PathLike):
        # convert to Path object unless None
        if value is None:
            setattr(self, "_dem_path", value)
        else:
            if isinstance(value, (str, PathLike)):
                setattr(self, "_dem_path", Path(value))
                # if a project directory is not provided, it will be set to
                # directory containing DEM when dem_path is set
                if self.project_dir is None:
                    self.project_dir = Path(value).parent
            else:
                raise TypeError(
                    f"dem_path must be a string or os.PathLike object"
                )

    # make these attributes properties with getters and setters
    # t.path_property() ensures attribute is a pathlib.Path object
    project_dir = t.path_property("project_dir")
    
    # below paths are set to default values in __init__
    filtered_dem_path = t.path_property("filtered_dem_path")
    slope_path = t.path_property("slope_path")
    curvature_path = t.path_property("curvature_path")
    filled_path = t.path_property("filled_path")
    mfd_fac_path = t.path_property("mfd_fac_path")
    d8_fdr_path = t.path_property("d8_fdr_path")
    basins_path = t.path_property("basins_path")
    outlets_path = t.path_property("outlets_path")
    flow_skeleton_path = t.path_property("flow_skeleton_path")
    curvature_skeleton_path = t.path_property("curvature_skeleton_path")
    combined_skeleton_path = t.path_property("combined_skeleton_path")
    cost_function_geodesic_path = t.path_property("cost_function_geodesic_path")
    geodesic_distance_path = t.path_property("geodesic_distance_path")
    channel_heads_path = t.path_property("channel_heads_path")
    flowline_path = t.path_property("flowline_path")
    endpoints_path = t.path_property("endpoints_path")
    binary_hand_path = t.path_property("binary_hand_path")
    custom_flowline_path = t.path_property("custom_flowline_path")
    custom_flowline_raster_path = t.path_property("custom_flowline_raster_path")
    channel_network_path = t.path_property("channel_network_path")
    channel_network_raster_path = t.path_property("channel_network_raster_path")
    cost_function_channel_path = t.path_property("cost_function_channel_path")
    hand_path = t.path_property("hand_path")
    segmented_channel_network_path = t.path_property("segmented_channel_network_path")
    segmented_channel_network_raster_path = t.path_property("segmented_channel_network_raster_path")
    segment_catchments_raster_path = t.path_property("segment_catchments_raster_path")
    segment_catchments_vector_path = t.path_property("segment_catchments_vector_path")
    river_attributes_path = t.path_property("river_attributes_path")
    catchment_path = t.path_property("catchment_path")
    src_path = t.path_property("src_path")
    streamflow_forecast_path = t.path_property("streamflow_forecast_path")
    flood_stage_path = t.path_property("flood_stage_path")
    fim_path = t.path_property("fim_path")
    fsm_inundation_path = t.path_property("fsm_inundation_path")
    fsm_dephier_path = t.path_property("fsm_dephier_path")
    fsm_labels_path = t.path_property("fsm_labels_path")
    fsm_flowdir_path = t.path_property("fsm_flowdir_path")
    coastal_inundation_path = t.path_property("coastal_inundation_path")



    @property
    def config(self):
        """
        Getter for the config property. Returns the PGF_Config instance.
        """
        return self._config

    @config.setter
    def config(self, value):
        """
        Setter for the config property. Accepts either a PGF_Config instance or a dict.
        Converts dicts to PGF_Config instances.
        """
        if isinstance(value, PGF_Config):
            self._config = value
        elif isinstance(value, dict):
            self._config = PGF_Config(value)
        elif value is None:
            self._config = None
        else:
            raise ValueError(
                "Config must be a PGF_Config instance, a dict, or None."
            )

    def __init__(
        self,
        dem_path,
        project_dir=None,
        config=None,
    ):
        """
        Create a new pygeoflood model instance.

        Parameters
        ----------
        dem_path : `str`, `os.PathLike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.PathLike`, optional
            Path to project directory. Default is the directory containing the
            DEM. All outputs will be saved to this directory.
        **kwargs : `dict`, optional
        """
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self.project_dir = project_dir
        elif dem_path is not None:
            self.project_dir = Path(dem_path).parent
        else:
            self.project_dir = None
        # automatically becomes a pathlib.Path object if not aleady
        self.dem_path = dem_path
        
        # Initialize default paths
        default_prefix=f"{self.project_dir}/{self.dem_path.stem}"
        # these all become Path objects with t.path_property
        self.filtered_dem_path = f"{default_prefix}_filtered.tif"
        self.slope_path = f"{default_prefix}_slope.tif"
        self.curvature_path = f"{default_prefix}_curvature.tif"
        self.filled_path = f"{default_prefix}_filled.tif"
        self.mfd_fac_path = f"{default_prefix}_mfd_fac.tif"
        self.d8_fdr_path = f"{default_prefix}_d8_fdr.tif"
        self.basins_path = f"{default_prefix}_basins.tif"
        self.outlets_path = f"{default_prefix}_outlets.tif"
        self.flow_skeleton_path = f"{default_prefix}_flow_skeleton.tif"
        self.curvature_skeleton_path = f"{default_prefix}_curvature_skeleton.tif"
        self.combined_skeleton_path = f"{default_prefix}_combined_skeleton.tif"
        self.cost_function_geodesic_path = f"{default_prefix}_cost_function_geodesic.tif"
        self.geodesic_distance_path = f"{default_prefix}_geodesic_distance.tif"
        self.channel_heads_path = f"{default_prefix}_channel_heads.shp"
        self.flowline_path = f"{default_prefix}_flowline.shp"
        self.endpoints_path = f"{default_prefix}_endpoints.csv"
        self.binary_hand_path = f"{default_prefix}_binary_hand.tif"
        self.channel_network_path = f"{default_prefix}_channel_network.shp"
        self.channel_network_raster_path = f"{default_prefix}_channel_network_raster.tif"
        self.cost_function_channel_path = f"{default_prefix}_cost_function_channel.tif"
        self.hand_path = f"{default_prefix}_HAND.tif"
        self.segmented_channel_network_path = f"{default_prefix}_segmented_channel_network.shp"
        self.segmented_channel_network_raster_path = f"{default_prefix}_segmented_channel_network_raster.tif"
        self.segment_catchments_raster_path = f"{default_prefix}_segment_catchments.tif"
        self.segment_catchments_vector_path = f"{default_prefix}_segment_catchments.shp"
        self.river_attributes_path = f"{default_prefix}_river_attributes.csv"
        self.catchment_path = f"{default_prefix}_cathcment.shp"
        self.src_path = f"{default_prefix}_src.csv"
        self.flood_stage_path = f"{default_prefix}_flood_stage.csv"
        self.fim_path = f"{default_prefix}_fim.tif"
        self.custom_flowline_path = f"{default_prefix}_custom_flowline.shp"
        self.custom_flowline_raster_path = f"{default_prefix}_custom_flowline_raster.tif"
        self.streamflow_forecast_path = f"{default_prefix}_streamflow_forecast"
        self.fsm_inundation_path = f"{default_prefix}_fsm_inundation.tif"
        self.fsm_dephier_path = f"{default_prefix}_fsm_dephier.json"
        self.fsm_labels_path = f"{default_prefix}_fsm_labels.npy"
        self.fsm_flowdir_path = f"{default_prefix}_fsm_flowdir.npy"
        self.coastal_inundation_path = f"{default_prefix}_coastal_inundation.tif"



        # check if 'config' is a dictionary and not an instance of PGF_Config
        if isinstance(config, dict):
            # create PGF_Config instance from dictionary
            self.config = PGF_Config(config)
        elif isinstance(config, PGF_Config):
            # directly use PGF_Config instance if provided
            self.config = config
        else:
            self.config = None

    # string representation of class
    # output can be used to recreate instance
    def __repr__(self):
        if all(val is None for val in self.__dict__.values()):
            return f"{self.__class__.__name__}()"
        else:
            attrs = "\n    ".join(
                (
                    f'{k[1:]}="{v}",'
                    if isinstance(v, (str, Path))
                    else f"{k[1:]}={v!r},"
                )
                for k, v in self.__dict__.items()
                if v is not None
            )
            return f"{self.__class__.__name__}(\n    {attrs}\n)"

    def to_paths(self, file_path):
        with open(file_path, "w") as file:
            for attr, value in vars(self).items():
                if (
                    attr.endswith("_path") or attr.endswith("_dir")
                ) and value is not None and Path(value).is_file():
                    # remove leading "_" if necessary
                    attr = attr.lstrip("_")
                    file.write(f'{attr}="{value}"\n')
        print(f"Paths written to {file_path}")

    @staticmethod
    def from_paths(file_path):
        # read attributes from file
        attributes = {}
        with open(file_path, "r") as file:
            for line in file:
                # Assuming the format is exactly 'attribute="value",\n'
                line = line.strip()  # Remove whitespace and newline characters
                if line:
                    attr, value = line.split("=")
                    # Remove quotation marks
                    attributes[attr] = value.strip('"')
        if "project_dir" not in attributes.keys():
            attributes["project_dir"] = None
        # create instance of PyGeoFlood with attributes
        loaded_pgf = PyGeoFlood(
            dem_path=attributes["dem_path"],
            project_dir=attributes["project_dir"],
        )
        for attr, value in attributes.items():
            if attr != "dem_path" and attr != "project_dir":
                setattr(loaded_pgf, attr, value)
        print(f"PyGeoFlood instance created from {file_path}")
        print("Note: config attribute must be set separately.")
        return loaded_pgf

    @t.time_it
    @t.use_config_defaults
    def apply_nonlinear_filter(
        self,
        custom_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        method: str = "PeronaMalik2",
        smoothing_quantile: float = 0.9,
        time_increment: float = 0.1,
        n_iter: int = 50,
        sigma_squared: float = 0.05,
    ):
        """
        Apply nonlinear filter to DEM. The dem_path attribute must be set before
        calling this method.

        Parameters
        ----------
        custom_dem : `str`, `os.PathLike`, optional
            Custom file path for input DEM. If not provided, default DEM used. 
            A custom_path is required when a custom_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom file path to save filtered DEM. If not provided, filtered DEM
            will be saved in project directory.
        method : `str`, optional
            Filter method to apply to DEM. Options include:
            - "PeronaMalik1": TODO: detailed description
            - "PeronaMalik2": TODO: detailed description
            - "Gaussian": Smoothes DEM with a Gaussian filter.
            Default is "PeronaMalik2".
        smoothing_quantile : `float`, optional
            Quantile for calculating Perona-Malik nonlinear filter
            edge threshold value (kappa). Default is 0.9.
        time_increment : `float`, optional
            Time increment for Perona-Malik nonlinear filter. Default is 0.1.
            AKA gamma, a higher makes diffusion process faster but can lead to
            instability.
        n_iter : `int`, optional
            Number of iterations for Perona-Malik nonlinear filter. Default is 50.
        sigma_squared : `float`, optional
            Variance of Gaussian filter. Default is 0.05.
        """
        if custom_dem is None:
            dem=self.dem_path
        else:
            dem=custom_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom DEM is provided.")

        t.check_attributes(
            [("DEM", dem)], "apply_nonlinear_filter"
        )

        # read original DEM
        dem, dem_profile = t.read_raster(dem)
        pixel_scale = dem_profile["transform"].a
        edgeThresholdValue = t.lambda_nonlinear_filter(
            dem, pixel_scale, smoothing_quantile
        )

        filtered_dem = t.anisodiff(
            img=dem,
            niter=n_iter,
            kappa=edgeThresholdValue,
            gamma=time_increment,
            step=(pixel_scale, pixel_scale),
            option=method,
        )

        # set output file path for filtered DEM
        if custom_path is None:
            output_filtered_dem_path = self.filtered_dem_path
        else:
            output_filtered_dem_path = f"{custom_path}.tif"

        # write filtered DEM
        t.write_raster(
            raster=filtered_dem,
            profile=dem_profile,
            file_path=output_filtered_dem_path,
        )
        print(f"Filtered DEM written to {output_filtered_dem_path}")

    @t.time_it
    @t.use_config_defaults
    def calculate_slope(
        self,
        custom_filtered_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate slope of DEM.

        Parameters
        ----------
        custom_filtered_dem : `str`, `os.PathLike`, optional
            Custom file path for input filtered dem. If not provided default 
            filtered_dem is used. A custom_path is required when a 
            custom_filtered_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save slope raster. If not provided, slope raster
            will be saved in project directory.
        """
        if custom_filtered_dem is None:
            filtered_dem = self.filtered_dem_path
        else:
            filtered_dem = custom_filtered_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filtered DEM is provided.")

        t.check_attributes(
            [("Filtered DEM", filtered_dem)], "calculate_slope"
        )

        # read filtered DEM
        filtered_dem, filtered_dem_profile = t.read_raster(
            filtered_dem
        )
        # pixel scale must be the same in x and y directions
        # transform.a is in x direction, transform.e is in y direction
        pixel_scale = filtered_dem_profile["transform"].a
        slope_array = t.compute_dem_slope(filtered_dem, pixel_scale)

        # set output file path for slope array
        if custom_path is None:
            output_slope_path = self.slope_path
        else:
            output_slope_path = f"{custom_path}.tif"

        # write slope array
        t.write_raster(
            raster=slope_array,
            profile=filtered_dem_profile,
            file_path=output_slope_path,
        )
        print(f"Slope raster written to {output_slope_path}")

    @t.time_it
    @t.use_config_defaults
    def calculate_curvature(
        self,
        custom_filtered_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        method: str = "geometric",
    ):
        """
        Calculate curvature of DEM.

        Parameters
        ----------
        custom_filtered_dem : `str`, `os.PathLike`, optional
            Custom file path for input filtered dem. If not provided default 
            filtered_dem is used. A custom_path is required when a 
            custom_filtered_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save curvature raster. If not provided, curvature
            raster will be saved in project directory.
        method : `str`, optional
            Method for calculating curvature. Options include:
            - "geometric": TODO: detailed description
            - "laplacian": TODO: detailed description
            Default is "geometric".
        """
        if custom_filtered_dem is None:
            filtered_dem = self.filtered_dem_path
        else:
            filtered_dem = custom_filtered_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filtered DEM is provided.")

        t.check_attributes(
            [("Filtered DEM", filtered_dem)], "calculate_curvature"
        )

        # read filtered DEM
        filtered_dem, filtered_dem_profile = t.read_raster(
            filtered_dem
        )
        pixel_scale = filtered_dem_profile["transform"].a
        curvature_array = t.compute_dem_curvature(
            filtered_dem,
            pixel_scale,
            method,
        )

        # set output file path for curvature array
        if custom_path is None:
            output_curvature_path = self.curvature_path
        else:
            output_curvature_path = f"{custom_path}.tif"

        # write curvature array
        t.write_raster(
            raster=curvature_array,
            profile=filtered_dem_profile,
            file_path=output_curvature_path,
        )
        print(f"Curvature raster written to {output_curvature_path}")

    @t.time_it
    @t.use_config_defaults
    def fill_depressions(
        self,
        custom_filtered_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Fill filtered DEM depressions. This is a wrapper for the WhiteboxTools
        `fill_depressions` function.

        Parameters
        ----------
        custom_filtered_dem : `str`, `os.PathLike`, optional
            Custom file path for input filtered dem. If not provided default 
            filtered_dem is used. A custom_path is required when a 
            custom_filtered_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save filled DEM. If not provided, filled DEM will be saved
            in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `fill_depressions`
            function. See WhiteboxTools documentation for details.
        """
        if custom_filtered_dem is None:
            filtered_dem = self.filtered_dem_path
        else:
            filtered_dem = custom_filtered_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filtered DEM is provided.")

        t.check_attributes(
            [("Filtered DEM", filtered_dem)], "fill_depressions"
        )

        # set file path for filled DEM
        if custom_path is None:
            output_filled_path = self.filled_path
        else:
            output_filled_path = f"{custom_path}.tif"

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # fill DEM depressions
        # use absolute paths to avoid errors
        # Set default value for fix_flats if not provided
        if "fix_flats" not in wbt_args:
            wbt_args["fix_flats"] = True
        wbt.fill_depressions(
            dem=Path(filtered_dem).resolve(),
            output=Path(output_filled_path).resolve(),
            **wbt_args,
        )

        print(f"Filled DEM written to {output_filled_path}")

    @t.time_it
    @t.use_config_defaults
    def calculate_mfd_flow_accumulation(
        self,
        custom_filled_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate MFD flow accumulation. This is a wrapper for the WhiteboxTools
        `quinn_flow_accumulation` function.

        Parameters
        ----------
        custom_filled_dem : `str`, `os.PathLike`, optional
            Custom file path for input filled dem. If not provided default 
            filled_dem is used. A custom_path is required when a 
            custom_filled_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save MFD flow accumulation raster. If not provided, MFD flow
            accumulation raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `quinn_flow_accumulation`
            function. See WhiteboxTools documentation for details.
        """
        if custom_filled_dem is None:
            filled_dem = self.filled_path
        else:
            filled_dem = custom_filled_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filled DEM is provided.")

        t.check_attributes(
            [("Filled DEM", filled_dem)],
            "calculate_mfd_flow_accumulation",
        )

        # set file path for mfd_fac
        if custom_path is None:
            output_mfd_fac_path = self.mfd_fac_path
        else:
            output_mfd_fac_path = f"{custom_path}.tif"

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate MFD flow accumulation
        # use absolute paths to avoid errors
        if "out_type" not in wbt_args:
            wbt_args["out_type"] = "cells"
        wbt.quinn_flow_accumulation(
            dem=Path(filled_dem).resolve(),
            output=Path(output_mfd_fac_path).resolve(),
            **wbt_args,
        )

        print(
            f"MFD flow accumulation raster written to {output_mfd_fac_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_d8_flow_direction(
        self,
        custom_filled_dem: str | PathLike = None,
        custom_filtered_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate D8 flow direction. This is a wrapper for the WhiteboxTools
        `d8_pointer` function.

        Parameters
        ----------
        custom_filled_dem : `str`, `os.PathLike`, optional
            Custom file path for input filled dem. If not provided default 
            filled_dem is used. A custom_path is required when a 
            custom_filtered_dem is provided.
        custom_filtered_dem : `str`, `os.PathLike`, optional
            Custom file path for input filtered dem. If not provided default 
            filtered_dem is used. A custom_path is required when a 
            custom_filtered_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save D8 flow direction raster. If not provided, D8 flow
            direction raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `d8_pointer`
            function. See WhiteboxTools documentation for details.
        """
        if custom_filled_dem is None:
            filled_dem = self.filled_path
        else:
            filled_dem = custom_filled_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filled DEM is provided.")

        if custom_filtered_dem is None:
            filtered_dem = self.filtered_dem_path
        else:
            filtered_dem = custom_filtered_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom filtered DEM is provided.")

        check_rasters = [
            ("Filled DEM", filled_dem),
            ("Filtered DEM", filtered_dem)
        ]

        t.check_attributes(check_rasters, "calculate_d8_flow_direction")


        # set file path for filled DEM
        if custom_path is None:
            output_d8_fdr_path = self.d8_fdr_path
        else:
            output_d8_fdr_path = f"{custom_path}.tif"

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate D8 flow direction
        # use absolute paths to avoid errors
        wbt.d8_pointer(
            dem=Path(filled_dem).resolve(),
            output=Path(output_d8_fdr_path).resolve(),
            **wbt_args,
        )

        # for some reason WBT assigns D8 values to nodata cells
        # add back nodata cells from filtered DEM
        filtered_dem, filtered_profile = t.read_raster(filtered_dem)
        filtered_dem[filtered_dem == filtered_profile["nodata"]] = np.nan
        # read D8 flow direction raster
        d8_fdr, d8_profile = t.read_raster(output_d8_fdr_path)
        d8_fdr[np.isnan(filtered_dem)] = d8_profile["nodata"]
        # write D8 flow direction raster
        t.write_raster(
            raster=d8_fdr,
            profile=d8_profile,
            file_path=output_d8_fdr_path,
        )

        print(f"D8 flow direction raster written to {output_d8_fdr_path}")

    @t.time_it
    @t.use_config_defaults
    def find_outlets(
        self,
        custom_d8_fdr: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Create outlets raster. Outlets are cells which have no downslope neighbors
        according to the D8 flow direction. Outlets are designated by 1, all other
        cells are designated by 0.

        Parameters
        ----------
        custom_d8_fdr : `str`, `os.PathLike`, optional
            Custom file path for input d8 flow direction. If not provided default 
            d8_fdr is used. A custom_path is required when a 
            custom_d8_fdr is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save outlets raster. If not provided, outlets raster will be
            saved in project directory.
        """
        if custom_d8_fdr is None:
            d8_fdr = self.d8_fdr_path
        else:
            d8_fdr = custom_d8_fdr
            if custom_path is None:
                raise ValueError("A custom path is required when a custom d8_fdr is provided.")

        t.check_attributes(
            [("D8 flow direction raster", d8_fdr)],
            "find_outlets",
        )

        # read D8 flow direction raster, outlets designated by WBT as 0
        outlets, profile = t.read_raster(d8_fdr)
        nan_mask = outlets == profile["nodata"]
        # get outlets as 1, all else as 0
        # make all cells 1 that are not outlets
        outlets[outlets != 0] = 1
        # flip to get outlets as 1, all else as 0
        outlets = 1 - outlets
        # reset nodata cells, which were set to 0 above
        outlets[nan_mask] = profile["nodata"]

        # set file path for outlets path
        if custom_path is None:
            output_outlets_path = self.outlets_path
        else:
            output_outlets_path = f"{custom_path}.tif"



        # write outlets raster
        t.write_raster(
            raster=outlets,
            profile=profile,
            file_path=output_outlets_path,
        )

        print(f"Outlets raster written to {output_outlets_path}")

    @t.time_it
    @t.use_config_defaults
    def delineate_basins(
        self,
        custom_d8_fdr: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Delineate basins. This is a wrapper for the WhiteboxTools `basins` function.

        Parameters
        ----------
        custom_d8_fdr : `str`, `os.PathLike`, optional
            Custom file path for input d8 flow direction. If not provided default 
            d8_fdr is used. A custom_path is required when a 
            custom_d8_fdr is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save basins raster. If not provided, basins raster will be
            saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `basins` function.
            See WhiteboxTools documentation for details.
        """
        if custom_d8_fdr is None:
            d8_fdr = self.d8_fdr_path
        else:
            d8_fdr = custom_d8_fdr
            if custom_path is None:
                raise ValueError("A custom path is required when a custom d8_fdr is provided.")

        t.check_attributes(
            [("D8 flow direction raster", d8_fdr)],
            "delineate_basins",
        )

        # set file path for basins path
        if custom_path is None:
            output_basins_path = self.basins_path
        else:
            output_basins_path = f"{custom_path}.tif"
        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # delineate basins
        # use absolute paths to avoid errors
        wbt.basins(
            d8_pntr=Path(d8_fdr).resolve(),
            output=Path(output_basins_path).resolve(),
            **wbt_args,
        )

        print(f"Basins raster written to {output_basins_path}")

    @t.time_it
    @t.use_config_defaults
    def define_skeleton(
        self,
        custom_curvature: str | PathLike = None,
        custom_mfd_fac: str | PathLike = None,
        custom_path: str | PathLike = None,
        fac_threshold: float = 3000,
        write_flow_skeleton: bool = False,
        write_curvature_skeleton: bool = False
    ):
        """
        Define skeleton from flow and curvature.

        Parameters
        ----------
        custom_curvature : `str`, `os.PathLike`, optional
            Custom file path for input curvature. If not provided default
            curvature is used. A custom_path is required when a custom_curvature is provided.
        custom_mfd_fac : `str`, `os.PathLike`, optional
            Custom file path for input mfd_fac. If not provided default
            mfd_fac is used. A custom_path is required when a custom_mfd_fac is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save combined skeleton. If not provided, combined
            skeleton will be saved in project directory.
        fac_threshold : `float`, optional
            Flow accumlulation threshold for defining flow skeleton. Default is 3000.
        write_flow_skeleton : `bool`, optional
            Whether to write flow skeleton to file. Default is False.
        write_curvature_skeleton : `bool`, optional
            Whether to write curvature skeleton to file. Default is False.
        """
        if custom_curvature is None:
            curvature = self.curvature_path
        else:
            curvature = custom_curvature
            if custom_path is None:
                raise ValueError("A custom path is required when a custom curvature is provided.")

        if custom_mfd_fac is None:
            mfd_fac = self.mfd_fac_path
        else:
            mfd_fac = custom_mfd_fac
            if custom_path is None:
                raise ValueError("A custom path is required when a custom mfd_fac is provided.")

        check_rasters = [
            ("Curvature raster", curvature),
            ("Flow accumulation raster", mfd_fac),
        ]

        t.check_attributes(check_rasters, "define_skeleton")

        # get skeleton from curvature only
        curvature, curvature_profile = t.read_raster(curvature)
        finite_curvature = curvature[np.isfinite(curvature)]
        curvature_mean = np.nanmean(finite_curvature)
        curvature_std = np.nanstd(finite_curvature)
        print("Curvature mean: ", curvature_mean)
        print("Curvature standard deviation: ", curvature_std)
        print(f"Curvature Projection: {str(curvature_profile['crs'])}")
        thresholdCurvatureQQxx = 1.5
        curvature_threshold = (
            curvature_mean + thresholdCurvatureQQxx * curvature_std
        )
        curvature_skeleton = t.get_skeleton(curvature, curvature_threshold)

        # get skeleton from flow only
        mfd_fac, _ = t.read_raster(mfd_fac)
        mfd_fac[np.isnan(curvature)] = np.nan
        mfd_fac_mean = np.nanmean(mfd_fac)
        print("Mean upstream flow: ", mfd_fac_mean)
        fac_skeleton = t.get_skeleton(mfd_fac, fac_threshold)

        # get skeleton from flow and curvature
        combined_skeleton = t.get_skeleton(
            curvature, curvature_threshold, mfd_fac, fac_threshold
        )

        skeleton_profile = curvature_profile.copy()
        skeleton_profile.update(dtype="int16", nodata=-32768)

        if write_flow_skeleton:
            # set file path for flow skeleton
            output_flow_skeleton_path = self.flow_skeleton_path

            t.write_raster(
                raster=fac_skeleton,
                profile=skeleton_profile,
                file_path=output_flow_skeleton_path,
            )
            print(f"Flow skeleton written to {output_flow_skeleton_path}")

        if write_curvature_skeleton:
            # setfile path for curvature skeleton
            output_curvature_skeleton_path = self.curvature_skeleton_path
            t.write_raster(
                raster=curvature_skeleton,
                profile=skeleton_profile,
                file_path=output_curvature_skeleton_path,
            )
            print(
                f"Curvature skeleton written to {output_curvature_skeleton_path}"
            )

        # set combined skeleton file path
        if custom_path is None:
            output_combined_skeleton_path = self.combined_skeleton_path
        else:
            output_combined_skeleton_path = f"{custom_path}.tif"

        t.write_raster(
            raster=combined_skeleton,
            profile=skeleton_profile,
            file_path=output_combined_skeleton_path,
        )
        print(
            f"Combined skeleton written to {output_combined_skeleton_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_geodesic_distance(
        self,
        custom_curvature: str | PathLike = None,
        custom_mfd_fac: str | PathLike = None,
        custom_outlets: str | PathLike = None,
        custom_basins: str | PathLike = None,
        custom_combined_skeleton: str | PathLike = None,
        custom_filtered_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
        write_cost_function: bool = False,
        basin_elements: int = 2,
        area_threshold: float = 0.1,
        normalize_curvature: bool = True,
        local_cost_min: float | None = None,
    ):
        """
        Calculate geodesic distance.

        Parameters
        ----------
        custom_curvature : `str`, `os.PathLike`, optional
            Custom file path to input curvature. If not provided default 
            curvature is used. A custom_path is required when a custom_curvature is provided.
        custom_mfd_fac : `str`, `os.PathLike`, optional
            Custom file path to input mfd_fac. If not provided default 
            mfd_fac is used. A custom_path is required when a custom_mfd_fac is provided.
        custom_outlets : `str`, `os.PathLike`, optional
            Custom file path to input outlets. If not provided default 
            outlets is used. A custom_path is required when a custom_outlets is provided.
        custom_basins : `str`, `os.PathLike`, optional
            Custom file path to input basins. If not provided default 
            basins is used. A custom_path is required when a custom_basins is provided.
        custom_combined_skeleton : `str`, `os.PathLike`, optional
            Custom file path to input skeleton. If not provided default 
            combined skeleton is used. A custom_path is required when a custom_combined_skeleton is provided.
        custom_filtered_dem: `str`, `os.PathLike`, optional
            Custom file path to input filtered dem. If not provided default 
            filtered dem is used. A custom_path is required when a custom_filtered_dem is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save geodesic distance raster. If not provided, geodesic
            distance raster will be saved in project directory.
        write_cost_function : `bool`, optional
            Whether to write cost function raster to file. Default is False.
        basin_elements : `int`, optional
            Number of basin elements. Default is 2.
        area_threshold : `float`, optional
            Area threshold for fast marching method. Default is 0.1.
        normalize_curvature : `bool`, optional
            Whether to normalize curvature. Default is True.
        local_cost_min : `float`, optional
            Minimum local cost. Default is None.
        """
        if custom_curvature is None:
            curvature = self.curvature_path
        else:
            curvature = custom_curvature
        if custom_mfd_fac is None:
            mfd_fac = self.mfd_fac_path
        else:
            mfd_fac = custom_mfd_fac
        if custom_outlets is None:
            outlets = self.outlets_path
        else:
            outlets = custom_outlets
        if custom_basins is None:
            basins = self.basins_path
        else:
            basins = custom_basins
        if custom_combined_skeleton is None:
            combined_skeleton = self.combined_skeleton_path
        else:
            combined_skeleton = custom_combined_skeleton
        if custom_filtered_dem is None:
            filtered_dem = self.filtered_dem_path
        else:
            filtered_dem = custom_filtered_dem

        for input in [custom_curvature,
                      custom_mfd_fac,
                      custom_outlets,
                      custom_basins,
                      custom_combined_skeleton,
                      custom_filtered_dem]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")

        check_rasters = [
            ("Curvature raster", curvature),
            ("Flow accumulation raster", mfd_fac),
            ("Outlets raster", outlets),
            ("Basins raster", basins),
            ("Combined skeleton raster", combined_skeleton),
            ("filtered dem", filtered_dem),
        ]

        t.check_attributes(check_rasters, "calculate_geodesic_distance")

        outlets, o_profile = t.read_raster(outlets)
        outlets = outlets.astype(np.float32)
        outlets[(outlets == 0) | (outlets == o_profile["nodata"])] = np.nan
        outlets = np.transpose(np.argwhere(~np.isnan(outlets)))
        basins, _ = t.read_raster(basins)
        curvature, _ = t.read_raster(curvature)
        mfd_fac, _ = t.read_raster(mfd_fac)
        filtered_dem, filt_profile = t.read_raster(filtered_dem)
        mfd_fac[np.isnan(filtered_dem)] = np.nan
        del filtered_dem
        combined_skeleton, _ = t.read_raster(combined_skeleton)

        # get start points for Fast Marching Method
        fmm_start_points = t.get_fmm_points(
            basins, outlets, basin_elements, area_threshold
        )

        # Computing the local cost function
        # min-max normalization of curvature (0 to 1)
        if normalize_curvature:
            curvature = t.minmax_scale(curvature)
        curvature[np.isnan(curvature)] = 0

        # calculate cost function
        # Calculate the local reciprocal cost (weight, or propagation speed
        # in the eikonal equation sense)
        # (mfd_fac + flowMean * combined_skeleton + flowMean * curvature)
        flowMean = np.nanmean(mfd_fac)
        weights_arrays = [
            (1, mfd_fac),
            (flowMean, combined_skeleton),
            (flowMean, curvature),
        ]
        cost_function_geodesic = t.get_combined_cost(
            weights_arrays, return_reciprocal=True
        )
        if local_cost_min is not None:
            cost_function_geodesic[cost_function_geodesic < local_cost_min] = (
                1.0
            )
        # print("1/cost min: ", np.nanmin(cost_function))
        # print("1/cost max: ", np.nanmax(cost_function))
        del curvature, combined_skeleton

        # Compute the geodesic distance using Fast Marching Method
        geodesic_distance = t.fast_marching(
            fmm_start_points, basins, mfd_fac, cost_function_geodesic
        )

        if write_cost_function:
            # set cost function name
            output_cost_function_geodesic_path = self.cost_function_geodesic_path

            t.write_raster(
                raster=cost_function_geodesic,
                profile=filt_profile,
                file_path=output_cost_function_geodesic_path,
            )
            print(
                f"Cost function written to {output_cost_function_geodesic_path}"
            )

        #set geodesic distance path 
        if custom_path is None:
            output_geodesic_distance_path = self.geodesic_distance_path
        else:
            output_geodesic_distance_path = f"{custom_path}.tif"
        # write geodesic distance
        t.write_raster(
            raster=geodesic_distance,
            profile=filt_profile,
            file_path=output_geodesic_distance_path,
        )

        print(
            f"Geodesic distance raster written to {output_geodesic_distance_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def identify_channel_heads(
        self,
        custom_combined_skeleton: str | PathLike = None,
        custom_geodesic_distance: str | PathLike = None,
        custom_path: str | PathLike = None,
        channel_head_median_dist: int = 30,
        vector_extension: str = "shp",
        max_channel_heads: int = 10000,
    ):
        """
        Define channel heads.

        Parameters
        ----------
        custom_combined_skeleton : `str`, `os.PathLike`, optional
            Custom file path to input skeleton. If not provided default 
            combined skeleton is used. A custom_path is required when a custom_combined_skeleton is provided.
        custom_geodesic_distance : `str`, `os.PathLike`, optional
            Custom file path to input geodesic distsance. If not provided default 
            geodesic distance is used. A custom_path is required when a custom_geodesic_distance is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save channel heads shapefile. If not provided,
            channel heads shapefile will be saved in project directory.
        channel_head_median_dist : `int`, optional
            Median hillslope of the input DEM, i.e. the distance between
            each pixel and the first channelized downslope pixel. Default is 30.
        vector_extension : `str`, optional
            Extension for vector file. Default is "shp".
        max_channel_heads : `int`, optional
            Maximum number of channel heads to extract. Default is 10000.
            (useful for pre-allocation of memory for large rasters)
        """
        if custom_combined_skeleton is None:
            combined_skeleton = self.combined_skeleton_path
        else:
            combined_skeleton = custom_combined_skeleton
        if custom_geodesic_distance is None:
            geodesic_distance = self.geodesic_distance_path
        else:
            geodesic_distance = custom_geodesic_distance
        
        for input in [custom_combined_skeleton,
                    custom_geodesic_distance]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")


        check_rasters = [
            ("Combined skeleton raster", combined_skeleton),
            ("Geodesic distance raster", geodesic_distance),
        ]
       
        t.check_attributes(check_rasters, "identify_channel_heads")

        # read combined skeleton and geodesic distance rasters
        combined_skeleton, _ = t.read_raster(combined_skeleton)

        geodesic_distance, geo_profile = t.read_raster(
            geodesic_distance
        )

        # get channel heads
        ch_rows, ch_cols = t.get_channel_heads(
            combined_skeleton,
            geodesic_distance,
            channel_head_median_dist,
            max_channel_heads,
        )

        # set channel head outputs
        if custom_path is None:
            output_channel_heads_path = self.channel_heads_path    
        else:
            output_channel_heads_path = f"{custom_path}.{vector_extension}"
        # write channel heads points shapefile
        t.write_vector_points(
            rows=ch_rows,
            cols=ch_cols,
            profile=geo_profile,
            dataset_name="channel_heads",
            file_path=output_channel_heads_path,
        )

        print(
            f"Channel heads shapefile written to {output_channel_heads_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def find_endpoints(
        self,
        custom_flowline: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Save flowline endpoints in a csv file.

        Parameters
        ----------
        custom_flowline : `str`, `os.PathLike`, optional
            Custom file path to input flowlines. If not provided default 
            flowline is used. A custom_path is required when a custom_flowline is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save endpoints csv. If not provided, endpoints
            csv will be saved in project directory.
        """
        if custom_flowline is None:
            flowline = self.flowline_path
        else:
            flowline = custom_flowline
            if custom_path is None:
                raise ValueError("A custom path is required when a custom_flowline d8_fdr is provided.")

        t.check_attributes(
            [("PyGeoFlood.flowline_path", flowline)],
            "find_endpoints",
        )

        flowline = gpd.read_file(flowline)
        endpoints = t.get_endpoints(flowline)

        # set file path for endpoints
        if custom_path is None:
            output_endpoints_path = self.endpoints_path    
        else:
            output_endpoints_path = f"{custom_path}.csv"

        # write endpoints csv
        endpoints.to_csv(output_endpoints_path, index=False)

        print(f"Endpoints csv written to {output_endpoints_path}")

    @t.time_it
    @t.use_config_defaults
    def calculate_binary_hand(
        self,
        custom_dem: str | PathLike = None,
        custom_flowline: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Creates binary HAND raster with values of 1 given to pixels at a lower
        elevation than the NHD MR Flowline pixels they drain to (shortest D8 path).
        A value of zero is given to all other pixels in the image, i.e. pixels
        at a higher elevation than the NHD MR Flowlines.

        Parameters
        ----------
        custom_dem : `str`, `os.PathLike`, optional
            Custom file path to input dem. If not provided default 
            dem is used. A custom_path is required when a custom_dem is provided.
        custom_flowline : `str`, `os.PathLike`, optional
            Custom file path to input flowlines. If not provided default 
            flowline is used. A custom_path is required when a custom_flowline is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save binary HAND raster. If not provided, binary HAND
            raster will be saved in project directory.
        """
        if custom_flowline is None:
            flowline = self.flowline_path
        else:
            flowline = custom_flowline
        if custom_dem is None:
            dem = self.dem_path
        else:
            dem = custom_dem


        for input in [custom_flowline,
                      custom_dem]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")



        required_files = [
            ("DEM", dem),
            ("PyGeoFlood.flowline_path", flowline),
        ]

        t.check_attributes(required_files, "calculate_binary_hand")

        flowline = gpd.read_file(flowline)
        dem, dem_profile = t.read_raster(dem)
        binary_hand = t.get_binary_hand(flowline, dem, dem_profile)
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        binary_hand[dem == dem_profile["nodata"]] = out_profile["nodata"]
        binary_hand[np.isnan(dem)] = out_profile["nodata"]

        # set file path for binary hand
        if custom_path is None:
            output_binary_hand_path = self.binary_hand_path    
        else:
            output_binary_hand_path = f"{custom_path}.tif"

        # write binary hand
        t.write_raster(
            raster=binary_hand,
            profile=out_profile,
            file_path=output_binary_hand_path,
        )

        print(f"Binary HAND raster written to {output_binary_hand_path}")

    @t.time_it
    @t.use_config_defaults
    def rasterize_custom_flowline(
        self,
        custom_flowline: str | PathLike = None,
        custom_path: str | PathLike = None,
        layer: str | int = 0,
    ):
        """
        Create custom flowline raster from user-provided flowline vector file.
        This flowline could be obtained from the NHD HR dataset, for example.
        The attribute `custom_flowline_path` must be set before running this method.
        The flowline is buffered by 5 units and cropped to the extent of the DEM.
        The crs of the flowline will be reprojected to the crs of the DEM.
        Note: if you already have a custom flowline raster, you can skip this step
        and set the `custom_flowline_raster_path` attribute directly. Used to set the 
        path properties self.custom_flowline_path and self.custom_flowline_raster_path 

        Parameters
        ----------
        custom_flowline : `str`, `os.PathLike`, optional
            Custom file path to input flowlines. If not provided default 
            custom_flowline is used. A custom_path is required when a custom_flowline is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save custom flowline raster. If not provided, custom
            flowline raster will be saved in project directory. The flowline
            raster has data type int16 with 1=channel and 0=non-channel. Default path is project
            directory with suffix _custom_flowline.shp and _custom_flowline.tif
        layer : `str` or `int`, optional
            Layer name or number in flowline vector file. Default is 0.
        """
        if custom_flowline is None:
            flowline = self.custom_flowline_path
        else:
            flowline = custom_flowline
            if custom_path is None:
                raise ValueError(f"A custom path is required when a custom_flowline is provided.")
            
        t.check_attributes(
            [("PyGeoFlood.custom_flowline_path", flowline)],
            "rasterize_custom_flowline",
        )

        # get bounding box and crs from DEM to clip flowline
        with rio.open(self.dem_path) as ds:
            bbox = ds.bounds
            dem_profile = ds.profile

        # transform bounding box to crs of flowline
        with fiona.open(flowline, layer=layer) as ds:
            out_crs = ds.crs

        bbox = transform_bounds(dem_profile["crs"], out_crs, *bbox)

        # read custom flowline within bounding box and specified layer
        # layer default is 0, which will read the first (and likely only) layer
        custom_flowline = gpd.read_file(
            flowline,
            bbox=bbox,
            layer=layer,
        )

        # will reproject to dem_profile crs if necessary
        custom_flowline_raster = t.rasterize_flowline(
            flowline_gdf=custom_flowline,
            ref_profile=dem_profile,
            buffer=5,
        )

        # set custom_flowline_raster_path
        if custom_path is None:
            output_custom_flowline_raster_path   = self.custom_flowline_raster_path    
        else:
            output_custom_flowline_raster_path = f"{custom_path}.tif"

        # write custom flowline raster
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=custom_flowline_raster,
            profile=out_profile,
            file_path=output_custom_flowline_raster_path,
        )

        print(
            f"Custom flowline raster written to {output_custom_flowline_raster_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def extract_channel_network(
        self,
        custom_flowline: str | PathLike = None,
        custom_curvature: str | PathLike = None,
        custom_mfd_fac: str | PathLike = None,
        custom_endpoints: str | PathLike = None,
        custom_binary_hand: str | PathLike = None,
        custom_path: str | PathLike = None,
        retrace_flowline: bool = True,
        vector_extension: str = "shp",
        write_cost_function: bool = False,
        use_custom_flowline: bool = False,
        no_flowline: bool = False,
        custom_weight_curvature: float | None = None,
        custom_weight_mfd_fac: float | None = None,
        custom_weight_binary_hand: float | None = None,
        custom_weight_custom_flowline: float | None = None,
    ):
        """
        Extract channel network. The channel network will be written to raster
        and vector datasets with the same name and different extensions.
        By default, curvature, flow accumulation, and binary HAND (information
        from NHD MR flowline) are used to calculate the cost function. A custom
        flowline such as the NHD HR flowline can be included in the cost
        function with use_custom_flowline=True. Only curvature and flow
        accumulataion will be considered if no_flowline=True. The cost function
        is calculated as the reciprocal of the weighted sum of these of these
        rasters. The cost function is thresholded and used to extract the
        channel network.

        Parameters
        ----------
        custom_flowline : `str`, `os.PathLike`, optional
            Custom file path to input flowlines. If not provided default 
            flowline is used. Only used if retrace_flowline is False.
            A custom_path is required when a custom_flowline is provided.
        custom_curvature : `str`, `os.PathLike`, optional
            Custom file path to input curvature. If not provided default 
            curvature is used. A custom_path is required when a custom_curvature is provided.
        custom_mfd_fac : `str`, `os.PathLike`, optional
            Custom file path to input mfd_fac. If not provided default 
            mfd_fac is used. A custom_path is required when a custom_mfd_fac is provided.
        custom_endpoints : `str`, `os.PathLike`, optional
            Custom file path to input endpoints. If not provided default 
            endpoints is used. A custom_path is required when a custom_endpoints is provided.
        custom_binary_hand : `str`, `os.PathLike`, optional
            Custom file path to input binary_hand. If not provided default 
            bindary_hand is used. A custom_path is required when a custom_binary_hand is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save channel network raster. If not provided, channel
            network raster will be saved in project directory. The channel network
            vector file will have an identical name and path, but with the extension
            `vector_extension` (shp by default).
        retrace_flowline : `bool`, optional
            Whether to retrace flowline. Default is True. If False, the existing
            NHD MR flowline will be used as the channel network, no extraction
            will be performed, and the channel network raster and vector will be
            written to file. Note: setting this option to False is not recommended,
            but may apply to specific use cases, such as on low-relief terrain.
        vector_extension : `str`, optional
            Extension for vector file. Default is "shp".
        write_cost_function : `bool`, optional
            Whether to write cost function raster to file. Default is False.
        use_custom_flowline : `bool`, optional
            Whether to use custom flowline raster in cost function. Default is False.
            Does not automatically use custom_flowline shapefile input, 
            The custom_flowline raster should be set by running 
            `rasterize_custom_flowline` method first.
        no_flowline : `bool`, optional
            Whether to use the NHD MR flowline information (binary HAND raster)
            in the cost function. Default is False, so binary HAND raster will
            be used by default.
        custom_weight_curvature : `float`, optional
            Custom weight for curvature in cost function. Default is None.
        custom_weight_mfd_fac : `float`, optional
            Custom weight for flow accumulation in cost function. Default is None.
        custom_weight_binary_hand : `float`, optional
            Custom weight for binary HAND in cost function. Default is None.
        custom_weight_custom_flowline : `float`, optional
            Custom weight for custom flowline in cost function. Default is None.
        """
        if not retrace_flowline:
            print("Using existing NHD MR flowline...")

            if custom_flowline is None:
                flowline = self.flowline_path
            else:
                flowline = custom_flowline
                if custom_path is None:
                    raise ValueError(f"A custom path is required when a custom_flowline is provided.")

            required_files = [("PyGeoFlood.flowline_path", flowline)]
            t.check_attributes(required_files, "extract_channel_network")

            # set channel network raster and vectorpath
            if custom_path is None:
                output_channel_network_raster_path = self.channel_network_raster_path
                output_channel_network_vector_path = self.channel_network_path
            else:
                output_channel_network_raster_path = f"{custom_path}.tif"
                output_channel_network_vector_path = f"{custom_path}.{vector_extension}"

            _, dem_profile = t.read_raster(self.dem_path)
            out_profile = dem_profile.copy()
            out_profile.update(dtype="int16", nodata=-32768)

            flowline = gpd.read_file(flowline)

            channel_network = t.rasterize_flowline(
                flowline, dem_profile, buffer=None
            )

            t.write_raster(
                raster=channel_network,
                profile=out_profile,
                file_path=output_channel_network_raster_path,
            )

            print(
                f"Channel network raster written to {output_channel_network_raster_path}"
            )

            flowline["Type"] = "ChannelNetwork_NHD"
            # 0 indexed hydroids
            flowline["HYDROID"] = flowline.index
            flowline = flowline[["HYDROID", "Type", "geometry"]]
            flowline.to_file(output_channel_network_vector_path)
            print(
                f"Channel network vector written to {output_channel_network_vector_path}"
            )
            print(
                "Note: No channel network extraction performed. The NHD MR flowline was used."
            )

        else:
            print("Retracing flowline...")

            if custom_curvature is None:
                curvature = self.curvature_path
            else:
                curvature = custom_curvature
            if custom_mfd_fac is None:
                mfd_fac = self.mfd_fac_path
            else:
                mfd_fac = custom_mfd_fac
            if custom_endpoints is None:
                endpoints = self.endpoints_path
            else:
                endpoints = custom_endpoints
            if custom_binary_hand is None:
                binary_hand = self.binary_hand_path
            else:
                binary_hand = custom_binary_hand

            for input in [custom_curvature,
                        custom_mfd_fac,
                        custom_endpoints,
                        custom_binary_hand]:
                if input is not None and custom_path is None:
                    raise ValueError(f"A custom path is required when a custom {input} is provided.")

            required_files = [
                ("Curvature raster", curvature),
                ("Flow accumulation raster", mfd_fac),
                ("Endpoints csv", endpoints),
                ("Binary Hand", binary_hand)
            ]

            t.check_attributes(required_files, "extract_channel_network")

            # read and prepare required rasters
            mfd_fac, fac_profile = t.read_raster(mfd_fac)
            mfd_fac[mfd_fac == fac_profile["nodata"]] = np.nan
            mfd_fac = np.log(mfd_fac)
            mfd_fac = t.minmax_scale(mfd_fac)

            curvature, _ = t.read_raster(curvature)
            curvature[(curvature < -10) | (curvature > 10)] = np.nan
            curvature = t.minmax_scale(curvature)

            binary_hand, _ = t.read_raster(self.binary_hand_path)

            ### get cost surface array
            # use custom (likely NHD HR) flowline, NHD MR flowlines (binary HAND),
            # curvature, and flow accumulation in cost function
            if use_custom_flowline:
                    
                required_files = [
                    (
                        "Custom flowline raster",
                        self.custom_flowline_raster_path,
                    ),
                ]
                t.check_attributes(
                    required_files,
                    "extract_channel_network with use_custom_flowline=True",
                )
                # int16, 1 channel, 0 not
                custom_flowline_raster, _ = t.read_raster(
                    self.custom_flowline_raster_path
                )
                weight_binary_hand = 0.75
                weight_custom_flowline = 1

            # use NHD MR flowlines (binary HAND), curvature, and flow accumulation
            # in cost function (no custom flowline)
            elif not no_flowline:
                custom_flowline_raster = None
                weight_binary_hand = 0.75
                weight_custom_flowline = 0

            # use curvature and flow accumulation in cost function
            # (no NHD MR or custom flowlines)
            else:
                custom_flowline_raster = None
                weight_binary_hand = 0
                weight_custom_flowline = 0

            # default curvature and flow accumulation weights
            curv_weight_str = " (mean flow accumulation)"
            weight_curvature = np.nanmean(mfd_fac)
            weight_mfd_fac = 1

            # set custom weights if provided
            if custom_weight_curvature is not None:
                weight_curvature = custom_weight_curvature
                curv_weight_str = ""
            if custom_weight_mfd_fac is not None:
                weight_mfd_fac = custom_weight_mfd_fac
            if custom_weight_binary_hand is not None:
                weight_binary_hand = custom_weight_binary_hand
            if custom_weight_custom_flowline is not None:
                weight_custom_flowline = custom_weight_custom_flowline

            print("Cost function weights:")
            print(f"curvature          {weight_curvature:.4f}{curv_weight_str}")
            print(f"mfd_fac            {weight_mfd_fac:.4f}")
            print(f"binary_hand        {weight_binary_hand:.4f}")
            print(f"custom_flowline    {weight_custom_flowline:.4f}")

            weights_arrays = [
                (weight_curvature, curvature),
                (weight_mfd_fac, mfd_fac),
                (weight_binary_hand, binary_hand),
                (weight_custom_flowline, custom_flowline_raster),
            ]
            print(weight_curvature, weight_mfd_fac, weight_binary_hand, weight_custom_flowline)

            cost = t.get_combined_cost(weights_arrays)

            print(f"Cost min: {np.nanmin(cost)}")
            print(f"Cost max: {np.nanmax(cost)}")
            print(f"cost shape: {cost.shape}")

            if write_cost_function:
                t.write_raster(
                    raster=cost,
                    profile=fac_profile,
                    file_path=self.cost_function_channel_path,
                )
                print(
                    f"Cost function written to {str(self.cost_function_channel_path)}"
                )
            # threshold cost surface
            # get 2.5% quantile
            cost_quantile = np.quantile(cost[~np.isnan(cost)], 0.025)
            artificial_high_cost = 100000
            cost[(cost >= cost_quantile) | np.isnan(cost)] = (
                artificial_high_cost
            )
            channel_network, stream_rowcol, stream_keys = t.get_channel_network(
                cost,
                endpoints,
                fac_profile["transform"],
            )

            if custom_path is None:
                output_channel_network_raster_path = self.channel_network_raster_path
                output_channel_network_vector_path = self.channel_network_path
            else:
                output_channel_network_raster_path = f"{custom_path}.tif"
                output_channel_network_vector_path = f"{custom_path}.{vector_extension}"


            out_profile = fac_profile.copy()
            out_profile.update(dtype="int16", nodata=-32768)
            t.write_raster(
                raster=channel_network,
                profile=out_profile,
                file_path=self.channel_network_raster_path,
            )
            print(
                f"Channel network raster written to {output_channel_network_raster_path}"
            )

            t.write_vector_lines(
                rowcol_list=stream_rowcol,
                keys=stream_keys,
                profile=fac_profile,
                dataset_name="ChannelNetwork",
                file_path=self.channel_network_path,
            )
            print(
                f"Channel network vector written to {output_channel_network_vector_path}"
            )

    @t.time_it
    @t.use_config_defaults
    def calculate_hand(
        self,
        custom_filled_dem: str | PathLike = None,
        custom_channel_network_raster: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate Height Above Nearest Drainage (HAND). Returns a raster with
        each cell's vertical elevation above its nearest stream cell, measured
        along the downslope D8 flowpath from the cell. This is a wrapper for
        the WhiteboxTools `elevation_above_stream` function.

        Parameters
        ----------
        custom_filled_dem : `str`, `os.PathLike`, optional
            Custom file path to input filled dem. If not provided default 
            filled_dem is used. A custom_path is required when a custom_filled_dem is provided.
        custom_channel_network_raster : `str`, `os.PathLike`, optional
            Custom file path to input channel network raster. If not provided default 
            channel_network_raster is used. A custom_path is required when 
            a custom_channel_network_raster is provided.
        custom_path : `str`, `os.PathLike`, optional
            Path to save HAND raster. If not provided, basins raster will be
            saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools
            `elevation_above_stream` function. See WhiteboxTools documentation
            for details.
        """
        if custom_filled_dem is None:
            filled_dem = self.filled_path
        else:
            filled_dem = custom_filled_dem
        if custom_channel_network_raster is None:
            channel_network_raster = self.channel_network_raster_path
        else:
            channel_network_raster = custom_channel_network_raster


        for input in [custom_filled_dem,
                      custom_channel_network_raster]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")

        required_rasters = [
            ("Filled DEM", filled_dem),
            ("Channel network raster", channel_network_raster),
        ]
        t.check_attributes(required_rasters, "calculate_hand")

        # get file path for HAND
        self.hand_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="HAND",
        )
        # set output file path for HAND
        if custom_path is None:
            output_hand_path = self.hand_path    
        else:
            output_hand_path = f"{custom_path}.tif"

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate HAND
        # use absolute paths to avoid errors
        wbt.elevation_above_stream(
            dem=Path(filled_dem).resolve(),
            streams=Path(channel_network_raster).resolve(),
            output=Path(output_hand_path).resolve(),
            **wbt_args,
        )

        print(f"HAND raster written to {output_hand_path}")

    @t.time_it
    @t.use_config_defaults
    def segment_channel_network(
        self,
        custom_channel_network_vector: str | PathLike = None,
        custom_path: str | PathLike = None,
        vector_extension: str = "shp",
        segment_length: int | float = 1000,
    ):
        """
        Divide channel network into segments of a specified length.

        Parameters
        ----------
        custom_channel_network_vector : `str`, `os.PathLike`, optional
            Custom file path to input channel network vector. If not provided default 
            channel_network_vector is used. A custom_path is required 
            when a custom_channel_network_vector is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save segmented channel network. If not provided,
            segmented channel network will be saved in project directory.
        vector_extension : `str`, optional
            Extension for vector file. Default is "shp".
        segment_length : `int` or `float`, optional
            Length of segments. Default is 1000 units.
        """
        if custom_channel_network_vector is None:
            channel_network_vector = self.channel_network_path
        else:
            channel_network_vector = custom_channel_network_vector
            if custom_path is None:
                raise ValueError(f"A custom path is required when a custom custom_channel_network_vector is provided.")

        check_files = [
            ("Channel network vector", channel_network_vector),
        ]
        t.check_attributes(check_files, "segment_channel_network")

        channel_network = gpd.read_file(channel_network_vector)

        segments = t.split_network(channel_network, segment_length)

        # HydroID: 1, 2, 3, ... len(segments)
        out_gdf = gpd.GeoDataFrame(
            {
                "HYDROID": [i + 1 for i in range(len(segments))],
                "Length": [round(segment.length, 4) for segment in segments],
                "geometry": segments,
            },
            crs=channel_network.crs,
        )

        # set file path
        if custom_path is None:
            output_segmented_channel_network_path = self.segmented_channel_network_path    
        else:
            output_segmented_channel_network_path = f"{custom_path}.{vector_extension}"

        out_gdf.to_file(output_segmented_channel_network_path)

        print(
            f"Segmented channel network written to {output_segmented_channel_network_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def delineate_segment_catchments(
        self,
        custom_segmented_channel_network: str | PathLike = None,
        custom_d8_fdr: str | PathLike = None,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Delineate catchments for each segment of the channel network.
        The D8 flow direction raster and segmented channel network vector are
        required to run delineate_segment_catchments. This is a wrapper for
        the WhiteboxTools watershed function.

        Parameters
        ----------
        custom_segmented_channel_network_path : `str`, `os.PathLike`, optional
            Custom file path to input segmented channel network path. If not provided default 
            is used. A custom_path is required when a custom_segmented_channel_network_path is provided.
        custom_d8_fdr : `str`, `os.PathLike`, optional
            Custom file path to input d8 fdr. If not provided default 
            d8_fdr is used. A custom_path is required when a custom_d8_fdr is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save segment catchments raster. If not provided,
            segment catchments will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools
        """
        if custom_segmented_channel_network is None:
            segmented_channel_network = self.segmented_channel_network_path
        else:
            segmented_channel_network = custom_segmented_channel_network
        if custom_d8_fdr is None:
            d8_fdr = self.d8_fdr_path
        else:
            d8_fdr = custom_d8_fdr

        for input in [custom_segmented_channel_network,
                    custom_d8_fdr]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")

        required_files = [
            (
                "Segmented channel network vector",
                segmented_channel_network,
            ),
            ("D8 flow direction raster", d8_fdr),
        ]
        t.check_attributes(required_files, "delineate_segment_catchments")

        # rasterize segmented channel network to use in wbt.watershed()
        with rio.open(d8_fdr) as ds:
            profile = ds.profile
        gdf = gpd.read_file(segmented_channel_network)
        segments_raster = rasterize(
            zip(gdf.geometry, gdf["HYDROID"]),
            out_shape=(profile["height"], profile["width"]),
            dtype="int16",
            transform=profile["transform"],
            fill=0,
        )

        # set segmented channel network raster name
        output_segmented_channel_network_raster_path = self.segmented_channel_network_raster_path

        # write segmented channel network raster
        out_profile = profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=segments_raster,
            profile=out_profile,
            file_path=output_segmented_channel_network_raster_path,
        )

        # set segmented channel network catchments path
        if custom_path is None:
            output_segment_catchments_path = self.segment_catchments_raster_path    
        else:
            output_segment_catchments_path = f"{custom_path}.tif"


        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # delineate catchments for each segment
        # use absolute paths to avoid errors
        wbt.watershed(
            d8_pntr=Path(d8_fdr).resolve(),
            pour_pts=Path(output_segmented_channel_network_raster_path).resolve(),
            output=Path(output_segment_catchments_path).resolve(),
            **wbt_args,
        )

        print(
            f"Segment catchments written to {output_segment_catchments_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def vectorize_segment_catchments(
        self,
        custom_segment_catchments_raster: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Vectorize segment catchments raster.

        Parameters
        ----------
        custom_segment_catchments_raster : `str`, `os.PathLike`, optional
            Custom file path to input segment catchments raster. If not provided default 
            segment_catchments_raster is used.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save segment catchments vector file. If not provided,
            segment catchments vector shapefile will be saved in project directory.
        """
        if custom_segment_catchments_raster is None:
            segment_catchments_raster = self.segment_catchments_raster_path
        else:
            segment_catchments_raster = custom_segment_catchments_raster

        for input in [custom_segment_catchments_raster]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")

        required_files = [("Segment catchments raster", segment_catchments_raster)]
        t.check_attributes(required_files, "vectorize_segment_catchments")

        # read segment catchments raster
        seg_catch, profile = t.read_raster(segment_catchments_raster)

        # vectorize segment catchments, use columns from classic GeoFlood workflow
        features = list(
            {"properties": {"HYDROID": int(v)}, "geometry": s}
            for (s, v) in shapes(
                seg_catch,
                connectivity=8,  # avoids isolated single pixel catchments
                transform=profile["transform"],
            )
            if v > 0
        )
        segment_catchments_vector = gpd.GeoDataFrame.from_features(
            features, crs=profile["crs"]
        )
        segment_catchments_vector["AreaSqKm"] = segment_catchments_vector.area / 1e6

        # set output vectorized segment catchments path
        if custom_path is None:
            output_segment_catchments_vector_path = self.segment_catchments_vector_path    
        else:
            output_segment_catchments_vector_path = custom_path

        # write vectorized segment catchments
        segment_catchments_vector.to_file(output_segment_catchments_vector_path)

        print(
            f"Segment catchments vector file written to {output_segment_catchments_vector_path}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_src(
        self,
        custom_dem: str | PathLike = None,
        custom_segmented_channel_network: str | PathLike = None,
        custom_segment_catchments_raster: str | PathLike = None,
        custom_catchments: str | PathLike = None,
        custom_hand: str | PathLike = None,
        custom_path: str | PathLike = None,
        # write_segment_catchments_features: bool = False,
        # vector_extension: str = None,
        # write_river_attributes: bool = False,
        min_slope: float = 0.000001,
        max_stage: float = 20,
        incr_stage: float = 0.1,
        custom_roughness_path: str | PathLike = None,
    ):
        """
        Calculate synthetic rating curves (SRC) for each segment of the channel
        network. The SRC are based on the channel geometry attributes and the
        stage-height relationship. The SRC are written to a csv file.

        Parameters
        ----------
        custom_dem : `str`, `os.PathLike`, optional
            Custom file path to input dem. If not provided default DEM
            is used. A custom_path is required when a custom_dem is provided.
        custom_segmented_channel_network_path : `str`, `os.PathLike`, optional
            Custom file path to input segmented channel network path. If not provided default 
            is used. A custom_path is required when a 
            custom_segmented_channel_network_path is provided.
        custom_segment_catchments_raster : `str`, `os.PathLike`, optional
            Custom file path to input segmented catchment raster path. If not provided default 
            is used. A custom_path is required when a 
            custom_segment_catchments_raster is provided.
        custom_catchments: `str`, `os.PathLike`, optional
            Custom file path to input catchments. If not provided default 
            catchments is used. A custom_path is required when a 
            custom_catchments is provided.
        custom_hand : `str`, `os.PathLike`, optional
            Custom file path to input HAND. If not provided default 
            hand is used. A custom_path is required when a custom_hand is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save synthetic rating curves. If not provided, SRC
            will be saved in project directory.
        min_slope : `float`, optional
            Minimum slope allowed on channel segments. Default is 0.000001.
        max_stage : `float`, optional
            Maximum stage height in SRC. Default is 20.
        incr_stage : `float`, optional
            Increment with which to calculate SRC, from 0 to max_stage.
            Default is 0.1.
        custom_roughness_path : `str`, `os.PathLike`, optional
            Custom path to a csv file with roughness, a.k.a. Manning's n values
            for each COMID in the study area. Must have columns "COMID" and
            "Roughness". If not provided, default Manning's n values, which
            have been determined for each COMID in CONUS, will be used. These
            default roughness values are determined from stream order:

            stream order | roughness
            ------------ | ---------
            1            | 0.200
            2            | 0.100
            3            | 0.065
            4            | 0.045
            5            | 0.030
            6            | 0.010
            7            | 0.025
        """
        if custom_dem is None:
            dem = self.dem_path
        else:
            dem = custom_dem

        if custom_segmented_channel_network is None:
            segmented_channel_network = self.segmented_channel_network_path
        else:
            segmented_channel_network = custom_segmented_channel_network

        if custom_segment_catchments_raster is None:
            segment_catchment_raster = self.segment_catchments_raster_path
        else:
            segment_catchment_raster = custom_segment_catchments_raster

        if custom_catchments is None:
            catchments = self.catchment_path
        else:
            catchments = custom_catchments
        
        if custom_hand is None:
            hand = self.hand_path
        else:
            hand = custom_hand

        for input in [custom_dem,
                    custom_segmented_channel_network,
                    custom_catchments,
                    custom_segment_catchments_raster,
                    custom_hand]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")



        required_files = [
            ("DEM", dem),
            ("Segmented channel network vector",segmented_channel_network,),
            ("Channel network segment catchments",segment_catchment_raster,),
            ("PyGeoFlood.catchment_path", catchments),
            ("HAND raster", hand),
        ]

        t.check_attributes(required_files, "calculate_src")

        segmented_channel_network = gpd.read_file(
            segmented_channel_network
        )

        nwm_catchments = gpd.read_file(catchments)
        nwm_catchments = nwm_catchments.to_crs(segmented_channel_network.crs)

        with rio.open(dem) as ds:
            msg = "Segmented channel network crs does not match DEM crs"
            assert ds.crs == segmented_channel_network.crs, msg

        segment_catchments, profile = t.read_raster(
            segment_catchment_raster
        )

        hand, _ = t.read_raster(hand)

        river_attributes = t.get_river_attributes(
            dem,
            segment_catchments,
            nwm_catchments,
            segmented_channel_network,
            profile,
            min_slope,
        )

        # river attributes path
        river_attributes_path = self.river_attributes_path
        river_attributes.to_csv(river_attributes_path, index=False)
        print(f"River attributes written to {river_attributes_path}")

        # slope raster from unfiltered DEM (*_slope.tif is from filtered DEM)
        unfilt_dem, _ = t.read_raster(dem)
        unfilt_slope = t.compute_dem_slope(
            unfilt_dem, profile["transform"].a, verbose=False
        )
        unfilt_slope = np.nan_to_num(unfilt_slope, nan=0)
        unfilt_dem = None
        # default stage heights: 0, 0.1, 0.2, ..., 20
        heights = np.arange(0, max_stage + incr_stage, incr_stage)
        cell_area = abs(profile["transform"].a * profile["transform"].e)
        # add channel geometry attributes to synthetic rating curves
        src_df = t.catchhydrogeo(
            hand,
            segment_catchments,
            river_attributes[["HYDROID"]].values,
            unfilt_slope,
            heights,
            cell_area,
            river_attributes,
            custom_roughness_path,
        )

        # self.src_path = t.get_file_path(
        #     custom_path=custom_path,
        #     project_dir=self.project_dir,
        #     dem_name=self.dem_path.stem,
        #     suffix="src",
        #     extension="csv",
        # )
        # set src path
        if custom_path is None:
            output_src_path = self.src_path    
        else:
            output_src_path = f"{custom_path}.csv"

        src_df.to_csv(output_src_path, index=False)
        print(f"Synthetic rating curves written to {output_src_path}")

    @t.time_it
    @t.use_config_defaults
    def calculate_flood_stage(
        self,
        custom_src: str | PathLike = None,
        custom_streamflow_forecast_path: str | PathLike = None,
        custom_path: str | PathLike = None,
        custom_Q: int | float = None,
    ):
        """
        Calculate flood stage for each segment of the channel network.
        Forecasted streamflow values for each COMID (feature ID) must be set
        in `PyGeoFlood.streamflow_forecast_path` before running if custom_Q is not set.
        If the streamflow forecast is a netCDF file it must be in NWM format
        (in xarray: 'streamflow' variable with a "feature_id" or "COMID" dim/coord).
        If the streamflow forecast is a CSV file, it must have columns
        "feature_id" (or "COMID") and "streamflow".

        Parameters
        ----------
        custom_src : `str`, `os.PathLike`, optional
            Custom file path to input synthetic rating curve. If not provided default 
            src is used. A custom_path is required when a custom_src is provided.
        custom_streamflow_forecast_path : `str`, `os.PathLike`, optional
            Custom file path to input streamflow_forecast_path. If not provided default 
            is used. A custom_path is required when a custom_streamflow_forecast_path is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save flood stage interpolated from synthetic rating
            curves. If not provided, flood stage will be saved in project
            directory.
        custom_Q : `int` or `float`, optional
            Constant streamflow value to assign to all segments. Default is None.
            If set, custom_Q will be used to calculate flood stage instead of
            forecasted streamflow values.
        """
        if custom_src is None:
            src = self.src_path
        else:
            src = custom_src
        if custom_streamflow_forecast_path is None:
            streamflow_forcast = self.streamflow_forecast_path
        else:
            streamflow_forcast = custom_streamflow_forecast_path

        for input in [custom_src,
                custom_streamflow_forecast_path]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")


        required_files = [
            ("Synthetic rating curves", src),
            (
                "PyGeoFlood.streamflow_forecast_path",
                streamflow_forcast,
            ),
        ]

        if custom_Q is None:
            t.check_attributes(required_files, "calculate_flood_stage")
        else:
            print(f"Applying custom streamflow to each segment: {custom_Q} cms")
            t.check_attributes([required_files[0]], "calculate_flood_stage")

        # read synthetic rating curves
        src = pd.read_csv(src)
        src = src[["HYDROID", "Stage_m", "Volume_m3", "COMID", "Discharge_cms"]]

        out_df = t.get_flood_stage(src, streamflow_forcast, custom_Q)

        self.flood_stage_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="flood_stage",
            extension="csv",
        )
        # set file path
        if custom_path is None:
            output_flood_stage_depth_path = self.flood_stage_path    
        else:
            output_flood_stage_depth_path = f"{custom_path}.csv"

        out_df.to_csv(output_flood_stage_depth_path, index=False)
        print(f"Flood stages written to {output_flood_stage_depth_path}")

    @t.time_it
    @t.use_config_defaults
    def inundate(
        self,
        custom_hand: str | PathLike = None,
        custom_flood_stage: str | PathLike = None,
        custom_segment_catchments_raster: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate flood inundation raster based on HAND and flood stage.

        Parameters
        ----------
        custom_hand : `str`, `os.PathLike`, optional
            Custom file path to input sHAND. If not provided default 
            HAND is used. A custom_path is required when a custom_hand is provided.
        custom_flood_stage : `str`, `os.PathLike`, optional
            Custom file path to input flood stage. If not provided default 
            is used. A custom_path is required when a custom_flood_stage is provided.
        custom_segment_catchments : `str`, `os.PathLike`, optional
            Custom file path to input segment_catchments. If not provided default 
            is used. A custom_path is required when a custom_segment_catchments is provided.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save flood inundation raster. If not provided, flood
            inundation raster will be saved in project directory.
        """
        if custom_hand is None:
            hand = self.hand_path
        else:
            hand = custom_hand
        
        if custom_flood_stage is None:
            flood_stage = self.flood_stage_path
        else:
            flood_stage = f"{custom_flood_stage}.csv"

        if custom_segment_catchments_raster is None:
            segment_catchment_raster = self.segment_catchments_raster_path
        else:
            segment_catchment_raster = custom_segment_catchments_raster

        for input in [custom_hand,
                    custom_flood_stage,
                    custom_segment_catchments_raster]:
            if input is not None and custom_path is None:
                raise ValueError(f"A custom path is required when a custom {input} is provided.")

        required_files = [
            ("HAND raster", hand),
            ("Flood stages", flood_stage),
            ("Segment catchments", segment_catchment_raster),
        ]
        t.check_attributes(required_files, "inundate")

        hand, profile = t.read_raster(hand)
        seg_catch, _ = t.read_raster(segment_catchment_raster)
        df = pd.read_csv(flood_stage)
        df = df.sort_values(by="HYDROID")
        # inundated = t.get_inun(hand, seg_catch, df)
        hydroids = df["HYDROID"].to_numpy()
        stage_m = df["Stage_m"].to_numpy()
        inundated = t.jit_inun(hand, seg_catch, hydroids, stage_m)

        # set nan values to 0
        inundated[np.isnan(inundated)] = -9999

        # set file path
        if custom_path is None:
            output_fim_path = self.fim_path    
        else:
            output_fim_path = f"{custom_path}.tif"


        out_profile = profile.copy()
        out_profile.update(dtype="float32")

        t.write_raster(
            raster=inundated,
            profile=out_profile,
            file_path=output_fim_path,
        )

        print(f"Flood inundation raster written to {output_fim_path}")

    @t.time_it
    @t.use_config_defaults
    def fill_spill_merge(
    self,
    uniform_depth: float = 0.5,
    gridded_depth: str | PathLike = None,
    custom_dem: str | PathLike = None,
    overwrite_dephier: bool = False,
    custom_path: str | PathLike = None
    ):
        """
        Runs Fill-Spill-Merge function from RichDEM to simulate pluvial flooding. 

        The method will save intermediary json files of the depression hierarchy, flow directions, 
        and flow labels to make successive runs faster.

        Parameters
        ----------
        uniform_depth : `float`
            Uniform depth to inundate the landscape. Units will be the units of the DEM.
            Defaults to 0.5.
        gridded_depth : `str`, `os.PathLike`, optional
            Path to a gridded depth raster to inundate the DEM. If a gridded_depth is 
            provided, the uniform_depth will be ignored. The gridded_depth will be clipped, 
            reprojected, and resampled to have the same resolution as the DEM if it 
            extends past it and is not the same spatial resolution.
        custom_dem : `str`, `os.PathLike`, optional
            Custom file path to input dem. If not provided default 
            DEM is used. A custom_path is required when a custom_hand is provided. Intermediary files
            will be saved to match the name of the input custom_dem and be within the default working folder.
        overwrite_dephier : `bool`
            If True, fill_spill_merge will overwrite all existing .json files associated
            with that DEM.
        custom_path :  `str`, `os.PathLike`, optional
            Custom path to save the inundated raster. If not provided, flooded raster 
            will be saved in the project directory with the default name.
        
        """
        try:
            import richdem as rd
            from _richdem.depression_hierarchy import Depression
        except ImportError as e:
            print(f"Could not import richdem! to install follow the following commands:")
            print("git clone https://github.com/mdp0023/richdem.git richdem")
            print("cd richdem/wrappers/pyrichdem")
            print("pip install .")
        if custom_dem is None:
            dem = self.dem_path
            fsm_inundation = self.fsm_inundation_path
            fsm_dephier = self.fsm_dephier_path
            fsm_labels = self.fsm_labels_path
            fsm_flowdir = self.fsm_flowdir_path
        else:
            dem = custom_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom DEM is provided.")
            name = custom_dem.split('/')[-1][:-4]
            fsm_inundation = custom_path
            fsm_dephier = f"{self.project_dir}/{name}_fsm_dephier.json"
            fsm_labels = f"{self.project_dir}/{name}_fsm_labels.npy"
            fsm_flowdir = f"{self.project_dir}/{name}_fsm_flowdir.npy"


        t.check_attributes(
            [("DEM", dem)], "fill_spill_merge"
        )
        # read original DEM
        dem, dem_profile = t.read_raster(dem)
        dem = rd.rdarray(dem,no_data=-9999, geotransform=dem_profile['transform']).astype(np.double)
        dem[np.isnan(dem)] = dem_profile['nodata']
        def get_bounds_from_profile(profile):
            transform = profile['transform']
            width = profile['width']
            height = profile['height']
            
            # Calculate coordinates of top left corner
            top_left_x, top_left_y = transform * (0, 0)
            
            # Calculate coordinates of bottom right corner
            bottom_right_x, bottom_right_y = transform * (width, height)
            
            # Form the bounds: left, bottom, right, top
            bounds = (top_left_x, bottom_right_y, bottom_right_x, top_left_y)
            
            return bounds

        # function to extract depression attributes from dephier object and returns as a dictionary that can be saved for later use
        def extract_depression_attributes(depression):
            return{#'__class__':depression.__class__,
                    'cell_count':depression.cell_count, 
                    'dep_label':depression.dep_label, 
                    'dep_vol':depression.dep_vol, 
                    'geolink':depression.geolink, 
                    'lchild':depression.lchild, 
                    'ocean_linked':depression.ocean_linked, 
                    'ocean_parent':depression.ocean_parent, 
                    'odep':depression.odep, 
                    'out_cell':depression.out_cell, 
                    'out_elev':depression.out_elev,
                    'parent':depression.parent, 
                    'pit_cell':depression.pit_cell, 
                    'pit_elev':depression.pit_elev, 
                    'rchild':depression.rchild,
                    'total_elevation':depression.total_elevation, 
                    'water_vol':depression.water_vol}

        # function to convert saved dictionary back to dephier object 
        def dict_to_instance(data):
            instance = Depression()
            for key, value in data.items():
                setattr(instance, key, value)  # Set each attribute on the instance
            return instance

        # determine depth
        if gridded_depth is not None:
            print(f'Using gridded_depth file: {gridded_depth}')
            depth, depth_profile = t.read_raster(gridded_depth)
            # calculate bounds of both rasters
            dem_bounds = get_bounds_from_profile(dem_profile)
            depth_bounds = get_bounds_from_profile(depth_profile)
            
            # ensure same CRS
            if depth_profile['crs'] != dem_profile['crs']:
                print(f"Reprojecting depth raster from {depth_profile['crs']} to {dem_profile['crs']}")
                # Calculate the transform and dimensions for the destination array
                transform, width, height = calculate_default_transform(
                    depth_profile['crs'], dem_profile['crs'], depth_profile['width'], depth_profile['height'], *depth_bounds
                )

                # Create a template profile for the destination based on the source profile
                destination_profile = depth_profile.copy()
                destination_profile.update({
                    'crs': dem_profile['crs'],
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # Create an empty array for the destination data
                destination_array = np.empty((height, width), dtype=depth.dtype)

                # Reproject the source array to the destination array
                reproject(
                    source=depth,
                    destination=destination_array,
                    src_transform=depth_profile['transform'],
                    src_crs=depth_profile['crs'],
                    dst_transform=transform,
                    dst_crs=dem_profile['crs'],
                    resampling=Resampling.nearest  # Choose an appropriate resampling algorithm
                ) 
                depth = destination_array
                depth_profile = destination_profile
           
            # ensure same bounds 
            if depth_bounds != dem_bounds:
                print(f"Clipping depth_raster to dem_raster")
                # Calculate overlap bounds
                overlap_bounds = (
                    max(depth_bounds[0], dem_bounds[0]),  # left
                    max(depth_bounds[1], dem_bounds[1]),  # bottom
                    min(depth_bounds[2], dem_bounds[2]),  # right
                    min(depth_bounds[3], dem_bounds[3])   # top
                )

                # Convert overlap bounds to array indices for depth_array
                depth_transform = depth_profile['transform']
                top_left = ~depth_transform * (overlap_bounds[0], overlap_bounds[3])  # ~transform is the inverse of transform
                bottom_right = ~depth_transform * (overlap_bounds[2], overlap_bounds[1])
                top_left_row, top_left_col = map(int, np.floor(top_left))
                bottom_right_row, bottom_right_col = map(int, np.ceil(bottom_right))

                # Clip the depth_array
                clipped_depth_array = depth[top_left_row:bottom_right_row, top_left_col:bottom_right_col]

                # Update the depth_profile
                new_transform = from_bounds(*overlap_bounds, clipped_depth_array.shape[1], clipped_depth_array.shape[0])
                depth_profile.update({
                    'height': clipped_depth_array.shape[0],
                    'width': clipped_depth_array.shape[1],
                    'transform': new_transform
                })
                depth = clipped_depth_array
                dem_bounds = get_bounds_from_profile(dem_profile)
                depth_bounds = get_bounds_from_profile(depth_profile)
            
            # ensure same cell size
            depth_cell_size = (depth_profile['transform'][0], depth_profile['transform'][0])
            dem_cell_size = ((dem_profile['transform'][0], dem_profile['transform'][0]))
            if depth_cell_size != dem_cell_size:
                print("Resampling depth raster to have same cell size as dem raster")

                # Update the depth_profile to match the dem_profile's grid
                resampled_depth_profile = depth_profile.copy()
                resampled_depth_profile.update({
                    'transform': dem_profile['transform'],
                    'width': dem_profile['width'],
                    'height': dem_profile['height'],
                    'crs': dem_profile['crs']  # Update this only if you've reprojected the depth raster to match the DEM's CRS
                })

                # Create an empty array for the resampled depth data
                resampled_depth_array = np.empty((dem_profile['height'], dem_profile['width']), dtype=depth.dtype)

                # Reproject (resample) the depth_array to match the DEM's grid
                reproject(
                    source=depth,
                    destination=resampled_depth_array,
                    src_transform=depth_profile['transform'],
                    src_crs=depth_profile['crs'],
                    dst_transform=dem_profile['transform'],
                    dst_crs=dem_profile['crs'],
                    resampling=Resampling.bilinear  # Bilinear resampling is often a good choice for continuous data
                )

                # Update the depth_array and depth_profile to use the resampled data and profile
                depth = resampled_depth_array
                depth_profile = resampled_depth_profile

            # create water depth array
            water_depth = rd.rdarray(depth,
                               no_data=-9999, 
                               geotransform=depth_profile['transform']).astype(np.double)
        
        else:
            # create a water depth array of 0s
            water_depth = rd.rdarray(np.zeros(dem.shape), 
                                     no_data=-9999, 
                                     geotransform=dem_profile['transform'])

            # #set the water depth
            water_depth += uniform_depth
            
        # if not all intermediary files exist or if overwrite is True:
        int_files = [fsm_dephier, fsm_labels, fsm_flowdir]   
        files_exist = all(os.path.exists(path) for path in int_files)
        
        if not files_exist or overwrite_dephier:
            print('Calculating depheir')
            # Get a simple labels array indicating all the edge cells belong to the ocean
            labels = rd.get_new_depression_hierarchy_labels(dem.shape)

            # Generate the Depression Hierarchy
            dephier, flowdirs = rd.get_depression_hierarchy(dem, labels)

            # convert depression data to list of dictioanry
            depression_data = [extract_depression_attributes(dep) for dep in dephier]

            # SAVE FILES
            # Save depression dictioanry to json file
            with open(fsm_dephier, 'w') as f:
                json.dump(depression_data, f)
            np.save(fsm_flowdir, flowdirs)
            np.save(fsm_labels, labels)

        else:
            print('Using saved dephier')

        # LOAD FILES
        # read in json file 
        with open(fsm_dephier, 'r') as f:
            depression_data = json.load(f)

        # convert json back to dephier object
        deps_object =[]
        for dep in depression_data:
            deps_object.append(dict_to_instance(dep))

        labels=np.load(fsm_labels)
        flowdirs=np.load(fsm_flowdir)
        labels = rd.rdarray(labels,no_data=-9999, 
                            geotransform=dem_profile['transform'])
        flowdirs = rd.rdarray(flowdirs,no_data=-9999,
                               geotransform=dem_profile['transform'])
        print('running')
        # #run FSM (the result is placed in the same water_depth array):
 
        rd.fill_spill_merge(dem=dem, 
                            labels=labels, 
                            flowdirs=flowdirs, 
                            deps=deps_object, 
                            wtd=water_depth)
        print('ran')
        water_depth[water_depth == 0] = None


        # save output FSM
        # Save the water_depth raster to the specified file path
        dem_profile.update(compress='lzw')
        with rio.open(fsm_inundation, 'w', **dem_profile) as dst:
            dst.write(water_depth, 1)
            
        print(f"FSM inundation saved to {fsm_inundation}")

    @t.time_it
    @t.use_config_defaults
    def c_hand(
        self,
        ocean_coords: tuple[float, float] = None,
        xy: bool = True,
        gage_el: float = None,
        custom_dem: str | PathLike = None,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate coastal inundation with the c-HAND method.

        Parameters
        ----------
        ocean_coords : `tuple`
            Tuple of coordinates (x, y) of a DEM cell in the ocean. Coordinates should be in the same CRS as the DEM.
        xy : `bool`, optional
            If False, ocean_coords are (row, col) of an array rather than (x, y) in a CRS. Default is True.
        gage_el : `float`, optional
            Constant water surface elevation to apply to DEM. Units and vertical datum must match DEM.
        custom_dem : `str`, `os.PathLike`, optional
            Custom file path to input dem. If not provided default 
            DEM is used. A custom_path is required when a custom_dem is provided. Intermediary files
            will be saved to match the name of the input custom_dem and be within the default working folder.
        custom_path :  `str`, `os.PathLike`, optional
            Custom path to save the coastal inundation raster. If not provided, coastal inundation raster 
            will be saved in the project directory with the default name.
        """
        if custom_dem is None:
            dem=self.dem_path
        else:
            dem=custom_dem
            if custom_path is None:
                raise ValueError("A custom path is required when a custom DEM is provided.")

        t.check_attributes([("DEM", dem)], "c_hand")

        if ocean_coords is None:
            raise ValueError("ocean_coords must be provided")
        if gage_el is None:
            raise ValueError("gage_el must be provided")

        # read original DEM
        dem, dem_profile = t.read_raster(dem)

        if xy:
            ocean_coords = rowcol(dem_profile["transform"], *ocean_coords)

        coastal_inun = t.get_c_hand(dem, gage_el, ocean_coords)

        # set file path
        if custom_path is None:
            output_coastal_inundation_path = self.coastal_inundation_path    
        else:
            output_coastal_inundation_path = f"{custom_path}.tif"

        t.write_raster(
            raster=coastal_inun,
            profile=dem_profile,
            file_path=output_coastal_inundation_path,
        )

        print(f"Coastal inundation raster written to {output_coastal_inundation_path}")
        

    @t.time_it
    def run_fim_workflow(self):
        """
        Run the full PyGeoFlood workflow.
        """
        self.apply_nonlinear_filter()
        self.calculate_slope()
        self.calculate_curvature()
        self.fill_depressions()
        self.calculate_mfd_flow_accumulation()
        self.calculate_d8_flow_direction()
        self.find_outlets()
        self.delineate_basins()
        self.define_skeleton()
        # self.calculate_geodesic_distance()
        # self.identify_channel_heads()
        self.find_endpoints()
        self.calculate_binary_hand()
        if self.custom_flowline_path.is_file():
            self.rasterize_custom_flowline()
        self.extract_channel_network()
        self.calculate_hand()
        self.segment_channel_network()
        self.delineate_segment_catchments()
        self.calculate_src()
        self.calculate_flood_stage()
        self.inundate()


# get dictionary of PyGeoFlood methods and their parameters
pgf_methods = [
    method
    for method in dir(PyGeoFlood)
    if inspect.isfunction(getattr(PyGeoFlood, method))
    and not method.startswith("__")
]
pgf_params = {}
for method in pgf_methods:
    pgf_params[method] = [
        param
        for param in inspect.signature(
            getattr(PyGeoFlood, method)
        ).parameters.keys()
        if param != "self"
    ]


class PGF_Config:

    def __init__(self, dict=None, **options):
        self.config = {}
        self.update_options(dict, **options)

    def __repr__(self):
        json_str = json.dumps(self.config, indent=4)
        indented = "\n".join("    " + line for line in json_str.splitlines())
        return f"PGF_Config(\n{indented}\n)"

    def update_options(self, dict=None, **new_options):
        """
        Update a PGF_Config instance with new options. Either a dictionary of
        options can be passed with format
        {method1: {option1: value1, option2: value2}, method2: ...}, or new
        options can be passed as keyword arguments with format
        method={option1: value1, option2: value2}. Unrecognized methods or
        options will be ignored.

        Parameters
        ----------
        dict : `dict`, optional
            Dictionary of options to update. Default is None.
        **new_options : `dict`, optional
            New options to add to the dictionary.
        """
        dict = {} if dict is None else dict
        dict.update(new_options)
        for method, opts in dict.items():
            if method in pgf_params.keys():
                for opt in opts.keys():
                    if opt in pgf_params[method]:
                        if method not in self.config:
                            self.config[method] = {}
                        if opt not in self.config[method]:
                            self.config[method][opt] = {}
                        self.config[method].update(opts)
                    else:
                        print(
                            f"Warning: Option '{opt}' not recognized for method '{method}'"
                        )
            else:
                print(f"Warning: Method '{method}' not recognized")

    def get_method_options(self, method_name):
        return self.config.get(method_name, {})
