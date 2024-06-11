import geopandas as gpd
import fiona
import inspect
import json
import numpy as np
import pandas as pd
import rasterio as rio
import sys

from . import tools as t
from os import PathLike
from pathlib import Path
from rasterio.features import rasterize, shapes
from rasterio.warp import transform_bounds


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
    
    # below are set to None initially
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
    segmented_channel_network_path = t.path_property(
        "segmented_channel_network_path"
    )
    segmented_channel_network_raster_path = t.path_property(
        "segmented_channel_network_raster_path"
    )
    segment_catchments_raster_path = t.path_property(
        "segment_catchments_raster_path"
    )
    river_attributes_path = t.path_property("river_attributes_path")
    segment_catchments_path = t.path_property("segment_catchments_path")
    catchment_path = t.path_property("catchment_path")
    src_path = t.path_property("src_path")
    streamflow_forecast_path = t.path_property("streamflow_forecast_path")
    flood_stage_path = t.path_property("flood_stage_path")
    fim_path = t.path_property("fim_path")

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
        dem_path=None,
        project_dir=None,
        filtered_dem_path=None,
        slope_path=None,
        curvature_path=None,
        filled_path=None,
        mfd_fac_path=None,
        d8_fdr_path=None,
        basins_path=None,
        outlets_path=None,
        flow_skeleton_path=None,
        curvature_skeleton_path=None,
        combined_skeleton_path=None,
        cost_function_geodesic_path=None,
        geodesic_distance_path=None,
        channel_heads_path=None,
        flowline_path=None,
        endpoints_path=None,
        binary_hand_path=None,
        custom_flowline_path=None,
        custom_flowline_raster_path=None,
        channel_network_path=None,
        channel_network_raster_path=None,
        cost_function_channel_path=None,
        hand_path=None,
        segmented_channel_network_path=None,
        segmented_channel_network_raster_path=None,
        segment_catchments_raster_path=None,
        river_attributes_path=None,
        segment_catchments_path=None,
        catchment_path=None,
        src_path=None,
        streamflow_forecast_path=None,
        flood_stage_path=None,
        fim_path=None,
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
        self.filtered_dem_path = filtered_dem_path
        self.slope_path = slope_path
        self.curvature_path = curvature_path
        self.filled_path = filled_path
        self.mfd_fac_path = mfd_fac_path
        self.d8_fdr_path = d8_fdr_path
        self.basins_path = basins_path
        self.outlets_path = outlets_path
        self.flow_skeleton_path = flow_skeleton_path
        self.curvature_skeleton_path = curvature_skeleton_path
        self.combined_skeleton_path = combined_skeleton_path
        self.cost_function_geodesic_path = cost_function_geodesic_path
        self.geodesic_distance_path = geodesic_distance_path
        self.channel_heads_path = channel_heads_path
        self.flowline_path = flowline_path
        self.endpoints_path = endpoints_path
        self.binary_hand_path = binary_hand_path
        self.custom_flowline_path = custom_flowline_path
        self.custom_flowline_raster_path = custom_flowline_raster_path
        self.channel_network_path = channel_network_path
        self.channel_network_raster_path = channel_network_raster_path
        self.cost_function_channel_path = cost_function_channel_path
        self.hand_path = hand_path
        self.segmented_channel_network_path = segmented_channel_network_path
        self.segmented_channel_network_raster_path = (
            segmented_channel_network_raster_path
        )
        self.segment_catchments_raster_path = segment_catchments_raster_path
        self.river_attributes_path = river_attributes_path
        self.segment_catchments_path = segment_catchments_path
        self.catchment_path = catchment_path
        self.src_path = src_path
        self.streamflow_forecast_path = streamflow_forecast_path
        self.flood_stage_path = flood_stage_path
        self.fim_path = fim_path
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
                ) and value is not None:
                    # remove leading "_" if necessary
                    attr = attr.lstrip("_")
                    file.write(f'{attr}="{value}"\n')
        print(f"Paths written to {file_path}")

    @staticmethod
    def from_paths(file_path):
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
        return PyGeoFlood(**attributes)

    @t.time_it
    @t.use_config_defaults
    def apply_nonlinear_filter(
        self,
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
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save filtered DEM. If not provided, filtered DEM
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

        t.check_attributes(
            [("PyGeoFlood.dem_path", self.dem_path)], "apply_nonlinear_filter"
        )

        # read original DEM
        dem, dem_profile = t.read_raster(self.dem_path)
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

        # get file path for filtered DEM
        self.filtered_dem_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="filtered",
        )

        # write filtered DEM
        t.write_raster(
            raster=filtered_dem,
            profile=dem_profile,
            file_path=self.filtered_dem_path,
        )
        print(f"Filtered DEM written to {str(self.filtered_dem_path)}")

    @t.time_it
    @t.use_config_defaults
    def calculate_slope(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate slope of DEM.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save slope raster. If not provided, slope raster
            will be saved in project directory.
        """

        t.check_attributes(
            [("Filtered DEM", self.filtered_dem_path)], "calculate_slope"
        )

        # read filtered DEM
        filtered_dem, filtered_dem_profile = t.read_raster(
            self.filtered_dem_path
        )
        # pixel scale must be the same in x and y directions
        # transform.a is in x direction, transform.e is in y direction
        pixel_scale = filtered_dem_profile["transform"].a
        slope_array = t.compute_dem_slope(filtered_dem, pixel_scale)

        # get file path for slope array
        self.slope_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="slope",
        )

        # write slope array
        t.write_raster(
            raster=slope_array,
            profile=filtered_dem_profile,
            file_path=self.slope_path,
        )
        print(f"Slope raster written to {str(self.slope_path)}")

    @t.time_it
    @t.use_config_defaults
    def calculate_curvature(
        self,
        custom_path: str | PathLike = None,
        method: str = "geometric",
    ):
        """
        Calculate curvature of DEM.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save curvature raster. If not provided, curvature
            raster will be saved in project directory.
        method : `str`, optional
            Method for calculating curvature. Options include:
            - "geometric": TODO: detailed description
            - "laplacian": TODO: detailed description
            Default is "geometric".
        """

        t.check_attributes(
            [("Filtered DEM", self.filtered_dem_path)], "calculate_curvature"
        )

        # read filtered DEM
        filtered_dem, filtered_dem_profile = t.read_raster(
            self.filtered_dem_path
        )
        pixel_scale = filtered_dem_profile["transform"].a
        curvature_array = t.compute_dem_curvature(
            filtered_dem,
            pixel_scale,
            method,
        )

        # get file path for curvature array
        self.curvature_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="curvature",
        )

        # write curvature array
        t.write_raster(
            raster=curvature_array,
            profile=filtered_dem_profile,
            file_path=self.curvature_path,
        )
        print(f"Curvature raster written to {str(self.curvature_path)}")

    @t.time_it
    @t.use_config_defaults
    def fill_depressions(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Fill filtered DEM depressions. This is a wrapper for the WhiteboxTools
        `fill_depressions` function.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Path to save filled DEM. If not provided, filled DEM will be saved
            in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `fill_depressions`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filtered DEM", self.filtered_dem_path)], "fill_depressions"
        )

        # get file path for filled DEM
        self.filled_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="filled",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # fill DEM depressions
        # use absolute paths to avoid errors
        wbt.fill_depressions(
            dem=self.filtered_dem_path.resolve(),
            output=self.filled_path.resolve(),
            fix_flats=True,
            **wbt_args,
        )

        print(f"Filled DEM written to {str(self.filled_path)}")

    @t.time_it
    @t.use_config_defaults
    def calculate_mfd_flow_accumulation(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate MFD flow accumulation. This is a wrapper for the WhiteboxTools
        `quinn_flow_accumulation` function.

        Parameters
        ----------
        mfd_fac_path : `str`, `os.PathLike`, optional
            Path to save MFD flow accumulation raster. If not provided, MFD flow
            accumulation raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `quinn_flow_accumulation`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filled DEM", self.filled_path)],
            "calculate_mfd_flow_accumulation",
        )

        # get file path for MFD flow accumulation
        self.mfd_fac_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="mfd_fac",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate MFD flow accumulation
        # use absolute paths to avoid errors
        wbt.quinn_flow_accumulation(
            dem=self.filled_path.resolve(),
            output=self.mfd_fac_path.resolve(),
            out_type="cells",
            **wbt_args,
        )

        print(
            f"MFD flow accumulation raster written to {str(self.mfd_fac_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_d8_flow_direction(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate D8 flow direction. This is a wrapper for the WhiteboxTools
        `d8_pointer` function.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Path to save D8 flow direction raster. If not provided, D8 flow
            direction raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `d8_pointer`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filled DEM", self.filled_path)], "calculate_d8_flow_direction"
        )

        # get file path for D8 flow direction
        self.d8_fdr_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="d8_fdr",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate D8 flow direction
        # use absolute paths to avoid errors
        wbt.d8_pointer(
            dem=self.filled_path.resolve(),
            output=self.d8_fdr_path.resolve(),
            **wbt_args,
        )

        # for some reason WBT assigns D8 values to nodata cells
        # add back nodata cells from filtered DEM
        filtered_dem, filtered_profile = t.read_raster(self.filtered_dem_path)
        filtered_dem[filtered_dem == filtered_profile["nodata"]] = np.nan
        # read D8 flow direction raster
        d8_fdr, d8_profile = t.read_raster(self.d8_fdr_path)
        d8_fdr[np.isnan(filtered_dem)] = d8_profile["nodata"]
        # write D8 flow direction raster
        t.write_raster(
            raster=d8_fdr,
            profile=d8_profile,
            file_path=self.d8_fdr_path,
        )

        print(f"D8 flow direction raster written to {str(self.d8_fdr_path)}")

    @t.time_it
    @t.use_config_defaults
    def find_outlets(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Create outlets raster. Outlets are cells which have no downslope neighbors
        according to the D8 flow direction. Outlets are designated by 1, all other
        cells are designated by 0.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Path to save outlets raster. If not provided, outlets raster will be
            saved in project directory.
        """

        t.check_attributes(
            [("D8 flow direction raster", self.d8_fdr_path)],
            "find_outlets",
        )

        # read D8 flow direction raster, outlets designated by WBT as 0
        outlets, profile = t.read_raster(self.d8_fdr_path)
        nan_mask = outlets == profile["nodata"]
        # get outlets as 1, all else as 0
        # make all cells 1 that are not outlets
        outlets[outlets != 0] = 1
        # flip to get outlets as 1, all else as 0
        outlets = 1 - outlets
        # reset nodata cells, which were set to 0 above
        outlets[nan_mask] = profile["nodata"]

        # get file path for outlets raster
        self.outlets_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="outlets",
        )

        # write outlets raster
        t.write_raster(
            raster=outlets,
            profile=profile,
            file_path=self.outlets_path,
        )

        print(f"Outlets raster written to {str(self.outlets_path)}")

    @t.time_it
    @t.use_config_defaults
    def delineate_basins(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Delineate basins. This is a wrapper for the WhiteboxTools `basins` function.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Path to save basins raster. If not provided, basins raster will be
            saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `basins` function.
            See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("D8 flow direction raster", self.d8_fdr_path)],
            "delineate_basins",
        )

        # get file path for basins
        self.basins_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="basins",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # delineate basins
        # use absolute paths to avoid errors
        wbt.basins(
            d8_pntr=self.d8_fdr_path.resolve(),
            output=self.basins_path.resolve(),
            **wbt_args,
        )

        print(f"Basins raster written to {str(self.basins_path)}")

    @t.time_it
    @t.use_config_defaults
    def define_skeleton(
        self,
        custom_path: str | PathLike = None,
        fac_threshold: float = 3000,
        write_flow_skeleton: bool = False,
        write_curvature_skeleton: bool = False,
    ):
        """
        Define skeleton from flow and curvature.

        Parameters
        ----------
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

        check_rasters = [
            ("Curvature raster", self.curvature_path),
            ("Flow accumulation raster", self.mfd_fac_path),
        ]

        t.check_attributes(check_rasters, "define_skeleton")

        # get skeleton from curvature only
        curvature, curvature_profile = t.read_raster(self.curvature_path)
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
        mfd_fac, _ = t.read_raster(self.mfd_fac_path)
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
            # get file path for flow skeleton
            self.flow_skeleton_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="flow_skeleton",
            )
            t.write_raster(
                raster=fac_skeleton,
                profile=skeleton_profile,
                file_path=self.flow_skeleton_path,
            )
            print(f"Flow skeleton written to {str(self.flow_skeleton_path)}")

        if write_curvature_skeleton:
            # get file path for curvature skeleton
            self.curvature_skeleton_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="curvature_skeleton",
            )
            t.write_raster(
                raster=curvature_skeleton,
                profile=skeleton_profile,
                file_path=self.curvature_skeleton_path,
            )
            print(
                f"Curvature skeleton written to {str(self.curvature_skeleton_path)}"
            )

        # write combined skeleton
        self.combined_skeleton_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="combined_skeleton",
        )
        t.write_raster(
            raster=combined_skeleton,
            profile=skeleton_profile,
            file_path=self.combined_skeleton_path,
        )
        print(
            f"Combined skeleton written to {str(self.combined_skeleton_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_geodesic_distance(
        self,
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

        check_rasters = [
            ("Curvature raster", self.curvature_path),
            ("Flow accumulation raster", self.mfd_fac_path),
            ("Outlets raster", self.outlets_path),
            ("Basins raster", self.basins_path),
            ("Combined skeleton raster", self.combined_skeleton_path),
        ]

        t.check_attributes(check_rasters, "calculate_geodesic_distance")

        outlets, o_profile = t.read_raster(self.outlets_path)
        outlets = outlets.astype(np.float32)
        outlets[(outlets == 0) | (outlets == o_profile["nodata"])] = np.nan
        outlets = np.transpose(np.argwhere(~np.isnan(outlets)))
        basins, _ = t.read_raster(self.basins_path)
        curvature, _ = t.read_raster(self.curvature_path)
        mfd_fac, _ = t.read_raster(self.mfd_fac_path)
        filtered_dem, filt_profile = t.read_raster(self.filtered_dem_path)
        mfd_fac[np.isnan(filtered_dem)] = np.nan
        del filtered_dem
        combined_skeleton, _ = t.read_raster(self.combined_skeleton_path)

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
            self.cost_function_geodesic_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="cost_function_geodesic",
            )
            t.write_raster(
                raster=cost_function_geodesic,
                profile=filt_profile,
                file_path=self.cost_function_geodesic_path,
            )
            print(
                f"Cost function written to {str(self.cost_function_geodesic_path)}"
            )

        # get file path for geodesic distance
        self.geodesic_distance_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="geodesic_distance",
        )

        # write geodesic distance
        t.write_raster(
            raster=geodesic_distance,
            profile=filt_profile,
            file_path=self.geodesic_distance_path,
        )

        print(
            f"Geodesic distance raster written to {str(self.geodesic_distance_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def identify_channel_heads(
        self,
        custom_path: str | PathLike = None,
        channel_head_median_dist: int = 30,
        vector_extension: str = "shp",
        max_channel_heads: int = 10000,
    ):
        """
        Define channel heads.

        Parameters
        ----------
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

        check_rasters = [
            ("Combined skeleton raster", self.combined_skeleton_path),
            ("Geodesic distance raster", self.geodesic_distance_path),
        ]

        t.check_attributes(check_rasters, "identify_channel_heads")

        # read combined skeleton and geodesic distance rasters
        combined_skeleton, _ = t.read_raster(self.combined_skeleton_path)

        geodesic_distance, geo_profile = t.read_raster(
            self.geodesic_distance_path
        )

        # get channel heads
        ch_rows, ch_cols = t.get_channel_heads(
            combined_skeleton,
            geodesic_distance,
            channel_head_median_dist,
            max_channel_heads,
        )

        # get file path for channel heads
        self.channel_heads_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="channel_heads",
            extension=vector_extension,
        )

        # write channel heads points shapefile
        t.write_vector_points(
            rows=ch_rows,
            cols=ch_cols,
            profile=geo_profile,
            dataset_name="channel_heads",
            file_path=self.channel_heads_path,
        )

        print(
            f"Channel heads shapefile written to {str(self.channel_heads_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def find_endpoints(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Save flowline endpoints in a csv file.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save endpoints csv. If not provided, endpoints
            csv will be saved in project directory.
        """

        t.check_attributes(
            [("PyGeoFlood.flowline_path", self.flowline_path)],
            "find_endpoints",
        )

        flowline = gpd.read_file(self.flowline_path)
        endpoints = t.get_endpoints(flowline)

        # get file path for endpoints
        self.endpoints_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="endpoints",
            extension="csv",
        )

        # write endpoints csv
        endpoints.to_csv(self.endpoints_path, index=False)

        print(f"Endpoints csv written to {str(self.endpoints_path)}")

    @t.time_it
    @t.use_config_defaults
    def calculate_binary_hand(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Creates binary HAND raster with values of 1 given to pixels at a lower
        elevation than the NHD MR Flowline pixels they drain to (shortest D8 path).
        A value of zero is given to all other pixels in the image, i.e. pixels
        at a higher elevation than the NHD MR Flowlines.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save binary HAND raster. If not provided, binary HAND
            raster will be saved in project directory.
        """

        required_files = [
            ("DEM", self.dem_path),
            ("PyGeoFlood.flowline_path", self.flowline_path),
        ]

        t.check_attributes(required_files, "calculate_binary_hand")

        flowline = gpd.read_file(self.flowline_path)
        dem, dem_profile = t.read_raster(self.dem_path)
        binary_hand = t.get_binary_hand(flowline, dem, dem_profile)
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        binary_hand[dem == dem_profile["nodata"]] = out_profile["nodata"]
        binary_hand[np.isnan(dem)] = out_profile["nodata"]

        # get file path for binary hand
        self.binary_hand_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="binary_hand",
        )

        # write binary hand
        t.write_raster(
            raster=binary_hand,
            profile=out_profile,
            file_path=self.binary_hand_path,
        )

        print(f"Binary HAND raster written to {str(self.binary_hand_path)}")

    @t.time_it
    @t.use_config_defaults
    def rasterize_custom_flowline(
        self,
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
        and set the `custom_flowline_raster_path` attribute directly.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save custom flowline raster. If not provided, custom
            flowline raster will be saved in project directory. The flowline
            raster has data type int16 with 1=channel and 0=non-channel.
        layer : `str` or `int`, optional
            Layer name or number in flowline vector file. Default is 0.
        """

        t.check_attributes(
            [("PyGeoFlood.custom_flowline_path", self.custom_flowline_path)],
            "rasterize_custom_flowline",
        )

        # get bounding box and crs from DEM to clip flowline
        with rio.open(self.dem_path) as ds:
            bbox = ds.bounds
            dem_profile = ds.profile

        # transform bounding box to crs of flowline
        with fiona.open(self.custom_flowline_path, layer=layer) as ds:
            out_crs = ds.crs

        bbox = transform_bounds(dem_profile["crs"], out_crs, *bbox)

        # read custom flowline within bounding box and specified layer
        # layer default is 0, which will read the first (and likely only) layer
        custom_flowline = gpd.read_file(
            self.custom_flowline_path,
            bbox=bbox,
            layer=layer,
        )

        # will reproject to dem_profile crs if necessary
        custom_flowline_raster = t.rasterize_flowline(
            flowline_gdf=custom_flowline,
            ref_profile=dem_profile,
            buffer=5,
        )

        # get file path for custom flowline raster
        self.custom_flowline_raster_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="custom_flowline",
        )

        # write custom flowline raster
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=custom_flowline_raster,
            profile=out_profile,
            file_path=self.custom_flowline_raster_path,
        )

        print(
            f"Custom flowline raster written to {str(self.custom_flowline_raster_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def extract_channel_network(
        self,
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

            required_files = [("PyGeoFlood.flowline_path", self.flowline_path)]
            t.check_attributes(required_files, "extract_channel_network")

            ### write channel network raster and shapefile
            # raster
            self.channel_network_raster_path = t.get_file_path(
                custom_path=custom_path,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="channel_network",
            )
            _, dem_profile = t.read_raster(self.dem_path)
            out_profile = dem_profile.copy()
            out_profile.update(dtype="int16", nodata=-32768)

            flowline = gpd.read_file(self.flowline_path)

            channel_network = t.rasterize_flowline(
                flowline, dem_profile, buffer=None
            )

            t.write_raster(
                raster=channel_network,
                profile=out_profile,
                file_path=self.channel_network_raster_path,
            )

            print(
                f"Channel network raster written to {str(self.channel_network_raster_path)}"
            )

            # write vector dataset
            self.channel_network_path = (
                self.channel_network_raster_path.with_suffix(
                    f".{vector_extension}"
                )
            )
            flowline["Type"] = "ChannelNetwork_NHD"
            # 0 indexed hydroids
            flowline["HYDROID"] = flowline.index
            flowline = flowline[["HYDROID", "Type", "geometry"]]
            flowline.to_file(self.channel_network_path)
            print(
                f"Channel network vector written to {str(self.channel_network_path)}"
            )
            print("Note: No channel network extraction performed. The NHD MR flowline was used.")

        else:
            print("Retracing flowline...")
            required_files = [
                ("Curvature raster", self.curvature_path),
                ("Flow accumulation raster", self.mfd_fac_path),
                ("Endpoints csv", self.endpoints_path),
            ]

            t.check_attributes(required_files, "extract_channel_network")

            # read and prepare required rasters
            mfd_fac, fac_profile = t.read_raster(self.mfd_fac_path)
            mfd_fac[mfd_fac == fac_profile["nodata"]] = np.nan
            mfd_fac = np.log(mfd_fac)
            mfd_fac = t.minmax_scale(mfd_fac)

            curvature, _ = t.read_raster(self.curvature_path)
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
                    ("Binary HAND raster", self.binary_hand_path),
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
                required_files = [
                    ("Binary HAND raster", self.binary_hand_path),
                ]
                t.check_attributes(required_files, "extract_channel_network")
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

            cost = t.get_combined_cost(weights_arrays)

            print(f"Cost min: {np.nanmin(cost)}")
            print(f"Cost max: {np.nanmax(cost)}")
            print(f"cost shape: {cost.shape}")

            if write_cost_function:
                self.cost_function_channel_path = t.get_file_path(
                    custom_path=None,
                    project_dir=self.project_dir,
                    dem_name=self.dem_path.stem,
                    suffix="cost_function_channel",
                )
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
                self.endpoints_path,
                fac_profile["transform"],
            )

            ### write channel network raster and shapefile
            # raster
            self.channel_network_raster_path = t.get_file_path(
                custom_path=custom_path,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="channel_network",
            )
            out_profile = fac_profile.copy()
            out_profile.update(dtype="int16", nodata=-32768)
            t.write_raster(
                raster=channel_network,
                profile=out_profile,
                file_path=self.channel_network_raster_path,
            )
            print(
                f"Channel network raster written to {str(self.channel_network_raster_path)}"
            )
            # write vector dataset
            self.channel_network_path = (
                self.channel_network_raster_path.with_suffix(
                    f".{vector_extension}"
                )
            )
            t.write_vector_lines(
                rowcol_list=stream_rowcol,
                keys=stream_keys,
                profile=fac_profile,
                dataset_name="ChannelNetwork",
                file_path=self.channel_network_path,
            )
            print(
                f"Channel network vector written to {str(self.channel_network_path)}"
            )

    @t.time_it
    @t.use_config_defaults
    def calculate_hand(
        self,
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
        custom_path : `str`, `os.PathLike`, optional
            Path to save HAND raster. If not provided, basins raster will be
            saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools
            `elevation_above_stream` function. See WhiteboxTools documentation
            for details.
        """

        required_rasters = [
            ("Filled DEM", self.filled_path),
            ("Channel network raster", self.channel_network_raster_path),
        ]
        t.check_attributes(required_rasters, "calculate_hand")

        # get file path for HAND
        self.hand_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="HAND",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate HAND
        # use absolute paths to avoid errors
        wbt.elevation_above_stream(
            dem=self.filled_path.resolve(),
            streams=self.channel_network_raster_path.resolve(),
            output=self.hand_path.resolve(),
            **wbt_args,
        )

        print(f"HAND raster written to {str(self.hand_path)}")

    @t.time_it
    @t.use_config_defaults
    def segment_channel_network(
        self,
        custom_path: str | PathLike = None,
        vector_extension: str = "shp",
        segment_length: int | float = 1000,
    ):
        """
        Divide channel network into segments of a specified length.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save segmented channel network. If not provided,
            segmented channel network will be saved in project directory.
        vector_extension : `str`, optional
            Extension for vector file. Default is "shp".
        segment_length : `int` or `float`, optional
            Length of segments. Default is 1000 units.
        """
        check_files = [
            ("Channel network vector", self.channel_network_path),
            ("PyGeoFlood.catchment_path", self.catchment_path),
        ]
        t.check_attributes(check_files, "segment_channel_network")

        channel_network = gpd.read_file(self.channel_network_path)

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

        self.segmented_channel_network_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="segmented_channel_network",
            extension=vector_extension,
        )

        out_gdf.to_file(self.segmented_channel_network_path)

        print(
            f"Segmented channel network written to {str(self.segmented_channel_network_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def delineate_segment_catchments(
        self,
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
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save segment catchments raster. If not provided,
            segment catchments will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools
        """

        required_files = [
            (
                "Segmented channel network vector",
                self.segmented_channel_network_path,
            ),
            ("D8 flow direction raster", self.d8_fdr_path),
        ]
        t.check_attributes(required_files, "delineate_segment_catchments")

        # rasterize segmented channel network to use in wbt.watershed()
        with rio.open(self.d8_fdr_path) as ds:
            profile = ds.profile
        gdf = gpd.read_file(self.segmented_channel_network_path)
        segments_raster = rasterize(
            zip(gdf.geometry, gdf["HYDROID"]),
            out_shape=(profile["height"], profile["width"]),
            dtype="int16",
            transform=profile["transform"],
            fill=0,
        )

        # get file path for segmented channel network raster
        self.segmented_channel_network_raster_path = t.get_file_path(
            custom_path=None,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="segmented_channel_network",
        )

        # write segmented channel network raster
        out_profile = profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=segments_raster,
            profile=out_profile,
            file_path=self.segmented_channel_network_raster_path,
        )

        # get file path for segmented channel network catchments
        self.segment_catchments_raster_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="segment_catchments",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # delineate catchments for each segment
        # use absolute paths to avoid errors
        wbt.watershed(
            d8_pntr=self.d8_fdr_path.resolve(),
            pour_pts=self.segmented_channel_network_raster_path.resolve(),
            output=self.segment_catchments_raster_path.resolve(),
            **wbt_args,
        )

        print(
            f"Segment catchments written to {str(self.segment_catchments_raster_path)}"
        )

    @t.time_it
    @t.use_config_defaults
    def calculate_src(
        self,
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

        required_files = [
            ("DEM", self.dem_path),
            (
                "Segmented channel network vector",
                self.segmented_channel_network_path,
            ),
            (
                "Channel network segment catchments",
                self.segment_catchments_raster_path,
            ),
            ("PyGeoFlood.catchment_path", self.catchment_path),
            ("HAND raster", self.hand_path),
        ]

        t.check_attributes(required_files, "calculate_src")

        segmented_channel_network = gpd.read_file(
            self.segmented_channel_network_path
        )

        nwm_catchments = gpd.read_file(self.catchment_path)
        nwm_catchments = nwm_catchments.to_crs(segmented_channel_network.crs)

        with rio.open(self.dem_path) as ds:
            msg = "Segmented channel network crs does not match DEM crs"
            assert ds.crs == segmented_channel_network.crs, msg

        segment_catchments, profile = t.read_raster(
            self.segment_catchments_raster_path
        )

        hand, _ = t.read_raster(self.hand_path)

        river_attributes = t.get_river_attributes(
            self.dem_path,
            segment_catchments,
            nwm_catchments,
            segmented_channel_network,
            profile,
            min_slope,
        )

        # if write_river_attributes:
        self.river_attributes_path = t.get_file_path(
            custom_path=None,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="river_attributes",
            extension="csv",
        )
        river_attributes.to_csv(self.river_attributes_path, index=False)
        print(f"River attributes written to {str(self.river_attributes_path)}")

        # slope raster from unfiltered DEM (*_slope.tif is from filtered DEM)
        unfilt_dem, _ = t.read_raster(self.dem_path)
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

        self.src_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="src",
            extension="csv",
        )

        src_df.to_csv(self.src_path, index=False)
        print(f"Synthetic rating curves written to {str(self.src_path)}")

    @t.time_it
    @t.use_config_defaults
    def calculate_flood_stage(
        self,
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
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save flood stage interpolated from synthetic rating
            curves. If not provided, flood stage will be saved in project
            directory.
        custom_Q : `int` or `float`, optional
            Constant streamflow value to assign to all segments. Default is None.
            If set, custom_Q will be used to calculate flood stage instead of
            forecasted streamflow values.
        """
        required_files = [
            ("Synthetic rating curves", self.src_path),
            (
                "PyGeoFlood.streamflow_forecast_path",
                self.streamflow_forecast_path,
            ),
        ]

        if custom_Q is None:
            t.check_attributes(required_files, "calculate_flood_stage")
        else:
            print(f"Applying custom streamflow to each segment: {custom_Q} cms")
            t.check_attributes([required_files[0]], "calculate_flood_stage")

        # read synthetic rating curves
        src = pd.read_csv(self.src_path)
        src = src[["HYDROID", "Stage_m", "Volume_m3", "COMID", "Discharge_cms"]]

        out_df = t.get_flood_stage(src, self.streamflow_forecast_path, custom_Q)

        self.flood_stage_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="flood_stage",
            extension="csv",
        )

        out_df.to_csv(self.flood_stage_path, index=False)
        print(f"Flood stages written to {str(self.flood_stage_path)}")

    @t.time_it
    @t.use_config_defaults
    def inundate(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate flood inundation raster based on HAND and flood stage.

        Parameters
        ----------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save flood inundation raster. If not provided, flood
            inundation raster will be saved in project directory.
        """
        required_files = [
            ("HAND raster", self.hand_path),
            ("Flood stages", self.flood_stage_path),
            ("Segment catchments", self.segment_catchments_raster_path),
        ]

        t.check_attributes(required_files, "inundate")

        hand, profile = t.read_raster(self.hand_path)
        seg_catch, _ = t.read_raster(self.segment_catchments_raster_path)
        df = pd.read_csv(self.flood_stage_path)
        df = df.sort_values(by="HYDROID")
        # inundated = t.get_inun(hand, seg_catch, df)
        hydroids = df["HYDROID"].to_numpy()
        stage_m = df["Stage_m"].to_numpy()
        inundated = t.jit_inun(hand, seg_catch, hydroids, stage_m)

        self.fim_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="fim",
        )

        out_profile = profile.copy()
        out_profile.update(dtype="float32")

        t.write_raster(
            raster=inundated,
            profile=out_profile,
            file_path=self.fim_path,
        )

        print(f"Flood inundation raster written to {str(self.fim_path)}")

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
        if self.custom_flowline_path is not None:
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
