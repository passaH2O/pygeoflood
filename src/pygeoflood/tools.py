import geopandas as gpd
import inspect
import numpy as np
import pandas as pd
import psutil
import rasterio as rio
import skfmm
import sys
import time
import warnings
import xarray as xr
import os

from functools import wraps
from importlib import resources
from numba import jit, prange
from os import PathLike
from pathlib import Path
from rasterio.features import geometry_mask, shapes, rasterize
from rasterio.transform import rowcol, xy
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.stats.mstats import gmean, mquantiles
from shapely.geometry import LineString, Point, shape
from shapely.ops import linemerge, snap, split
from skimage.graph import route_through_array
from skimage.measure import label
from whitebox import WhiteboxTools

warnings.filterwarnings(
    action="ignore",
    message="Invalid value encountered",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    action="ignore",
    message="divide by zero encountered",
    category=RuntimeWarning,
)


def wbt_callback(value):
    if not "%" in value:
        print(value)


def path_property(name: str) -> property:
    """
    Create a path property with a storage name prefixed by an underscore.

    Parameters
    ----------
    name : `str`
        Name of path property.

    Returns
    -------
    prop : `property`
        Path property with storage name prefixed by an underscore.
    """
    storage_name = f"_{name}"

    @property
    def prop(self) -> str | PathLike:
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value: str | PathLike):
        # convert to Path object unless None
        if value is None:
            setattr(self, storage_name, value)
        else:
            if isinstance(value, (str, PathLike)):
                setattr(self, storage_name, Path(value))
            else:
                raise TypeError(
                    f"{name} must be a string or os.PathLike object"
                )

    return prop


# time function dectorator
def time_it(func: callable) -> callable:
    """
    Decorator function to time the execution of a function

    Parameters
    ----------
    func : `function`
        Function to time.

    Returns
    -------
    wrapper : `function`
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        method_name = func.__name__
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Check if duration is over 60 minutes (3600 seconds)
        if duration > 3600:
            print(f"{func.__name__} completed in {duration / 3600:.4f} hours\n")
        # Check if duration is over 60 seconds
        elif duration > 60:
            print(f"{func.__name__} completed in {duration / 60:.4f} minutes\n")
        else:
            print(f"{func.__name__} completed in {duration:.4f} seconds\n")
        return result

    return wrapper


# sets parameters method will use. precedence:
# 1. explicitly passed arguments
# 2. configuration values
# 3. default values in the function signature
def use_config_defaults(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get the signature of the function
        sig = inspect.signature(func)

        # Prepare to collect explicitly passed arguments
        explicitly_passed_args = set(kwargs.keys())
        params = sig.parameters
        arg_names = list(params.keys())[1:]  # Skip 'self'

        # Determine which positional args were explicitly passed
        for i, arg in enumerate(args):
            if i < len(arg_names):
                explicitly_passed_args.add(arg_names[i])

        # Start by applying the default values
        final_params = {
            k: v.default
            for k, v in params.items()
            if v.default is not inspect.Parameter.empty
        }

        # Update with configuration values, if available
        method_name = func.__name__
        if hasattr(self, "config") and self.config:
            config_options = self.config.get_method_options(method_name)
            final_params.update(config_options)

        # Override with explicitly passed arguments
        bound_args = sig.bind_partial(self, *args, **kwargs)
        bound_args.apply_defaults()
        explicitly_passed_args_values = {
            k: bound_args.arguments[k]
            for k in explicitly_passed_args
            if k != "self" and k in bound_args.arguments
        }
        final_params.update(explicitly_passed_args_values)

        # Combine final_params with remaining kwargs (probably wbt_kwargs)
        final_params.update(kwargs)

        # Print the parameters being used
        print(f"Running {method_name} with parameters:")
        for key, val in final_params.items():
            print(f"    {key} = {val}")

        # Execute the function with the final parameters
        return func(self, **final_params)

    return wrapper


def check_attributes(
    attr_list: list[tuple[str, str | PathLike]], method
) -> None:
    """
    Check if required attributes are present.

    Parameters
    ----------
    attr_list : `list`
        List of (dataset name, path) tuples.
    """

    for name, path in attr_list:
        if path is None or not Path(path).is_file():
            raise ValueError(f"{name} invalid. {name} must be created before running {method}")
            # raise ValueError(f"{name} must be created before running {method}")


def read_raster(
    file_path: str,
) -> tuple[np.ndarray, rio.profiles.Profile | dict]:
    """
    Read a raster file and return the array, rasterio profile, and pixel scale.

    Parameters
    ----------
    file_path : `str`
        Path to raster file.


    Returns
    -------
    raster : `numpy.ndarray`
        Array of raster values.
    profile : `rasterio.profiles.Profile` | `dict`
        Raster profile.
    """
    with rio.open(file_path) as ds:
        raster = ds.read(1)
        profile = ds.profile
        msg = "Pixel width must be equal to pixel height"
        assert round(abs(ds.transform.a), 4) == round(abs(ds.transform.e), 4), msg
    # convert nodata to np.nan if dtype is float
    if "float" in profile["dtype"].lower():
        raster[raster == profile["nodata"]] = np.nan
    # set profile nodata to -9999
    profile.update(nodata=-9999)
    return raster, profile


def write_raster(
    raster: np.ndarray,
    profile: rio.profiles.Profile | dict,
    file_path: str | PathLike,
    compression: str = "lzw",
) -> str | PathLike:
    """
    Write a raster file.

    Parameters
    ----------
    raster : `numpy.ndarray`
        Array of raster values.
    profile : `rasterio.profiles.Profile` | `dict`
        Raster profile.
    compression : `str`, optional
        Compression method. Default is "lzw".
    """

    profile.update(compress=compression)

    with rio.open(file_path, "w", **profile) as ds:
        ds.write(raster, 1)


def write_vector_points(
    rows: np.ndarray,
    cols: np.ndarray,
    profile: rio.profiles.Profile | dict,
    dataset_name: str,
    file_path: str | PathLike,
) -> None:
    """
    Write points to a vector file.

    Parameters
    ----------
    rows : `numpy.ndarray`
        Array of row values.
    cols : `numpy.ndarray`
        Array of column values.
    profile : `rasterio.profiles.Profile` | `dict`
        Raster profile.
    dataset_name : `str`
        Name of "Type" field.
    file_path : `str` | `os.PathLike`
        Path to write file.
    """

    transform = profile["transform"]
    crs = profile["crs"]

    # Use rasterio.transform.xy to project xx, yy points
    rows, cols = np.array(rows), np.array(cols)
    xy_proj = [
        xy(transform, rows[i], cols[i], offset="center")
        for i in range(len(rows))
    ]
    # Unpack the projected coordinates to easting and northing for UTM
    easting, northing = zip(*xy_proj)

    # Create Point geometries
    geometry = [Point(x, y) for x, y in zip(easting, northing)]

    # Create a GeoDataFrame with Northing and Easting fields
    gdf = gpd.GeoDataFrame(
        {
            "Type": [dataset_name] * len(geometry),
            "Northing": northing,
            "Easting": easting,
        },
        geometry=geometry,
    )

    # Set the CRS for the GeoDataFrame from rasterio CRS
    gdf.crs = crs

    # Write the GeoDataFrame to a shapefile
    gdf.to_file(file_path)


def write_vector_lines(
    rowcol_list: list[tuple[np.ndarray, np.ndarray]],
    keys: list[str],
    profile: rio.profiles.Profile | dict,
    dataset_name: str,
    file_path: str | PathLike,
) -> None:
    """
    Write lines to a vector file.

    Parameters
    ----------
    rowcol_list : `list`
        List of (row, col) arrays.
    keys : `list`
        List of keys.
    profile : `rasterio.profiles.Profile` | `dict`
        Raster profile.
    dataset_name : `str`
        Name of "Type" field.
    file_path : `str` | `os.PathLike`
        Path to write file.
    """
    transform = profile["transform"]
    crs = profile["crs"]

    lines = []
    geometry = []
    for i, (rows, cols) in enumerate(rowcol_list):
        # convert cells (row, col) to points (x, y) with rasterio.transform.xy
        xy_coords = [
            xy(transform, row, col, offset="center")
            for row, col in zip(rows, cols)
        ]

        # Create LineString from projected coordinates
        if len(xy_coords) >= 2:
            geometry.append(LineString(xy_coords))
            lines.append({"Type": dataset_name, "HYDROID": keys[i]})
        else:
            print(
                f"Skipping line with less than 2 coordinates. HYDROID: {keys[i]}"
            )

    # Create a GeoDataFrame from the lines and set the CRS
    gdf = gpd.GeoDataFrame(lines, geometry=geometry, crs=crs)

    # Write the GeoDataFrame to a shapefile
    gdf.to_file(file_path)


def rasterize_flowline(
    flowline_gdf: gpd.GeoDataFrame,
    ref_profile: rio.profiles.Profile | dict,
    buffer: float | None = None,
) -> np.ndarray:
    """
    Rasterize flowline GeoDataFrame.

    Parameters
    ----------
    flowline_gdf : `geopandas.GeoDataFrame`
        GeoDataFrame of flowline.
    ref_profile : `rasterio.profiles.Profile` | `dict`
        Reference raster profile.
    buffer : `float`, optional
        Buffer distance. If None, no buffer is applied. Default is None.

    Returns
    -------
    flowline_raster : `numpy.ndarray`
        Rasterized flowline of type int16.
    """

    # reproject flowline to match reference profile if necessary
    if flowline_gdf.crs != ref_profile["crs"]:
        flowline_gdf = flowline_gdf.to_crs(ref_profile["crs"])

    if buffer is not None:
        flowline_gdf["geometry"] = flowline_gdf.buffer(buffer)

    shapes = ((geom, 1) for geom in flowline_gdf.geometry)

    flowline_raster = rasterize(
        shapes=shapes,
        fill=0,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        dtype="int16",
    )

    return flowline_raster


def get_file_path(
    custom_path: str | PathLike,
    project_dir: str | PathLike,
    dem_name: str,
    suffix: str,
    extension: str = "tif",
) -> str | PathLike:
    """
    Get file path.

    Parameters
    ----------
    custom_path : `str` | `os.PathLike`
        Optional custom path to save file.
    project_dir : `str` | `os.PathLike`
        Path to project directory.
    dem_name : `str`
        Name of DEM file.
    suffix : `str`
        Suffix to append to DEM filename. Only used if `write_path` is not
        provided.
    extension : `str`, optional
        File extension. Default is ".tif".

    Returns
    -------
    file_path : `str` | `os.PathLike`
        Path to write file.
    """
    if custom_path is not None:
        file_path = Path(custom_path)
    else:
        # append to DEM filename, save in project directory
        file_path = Path(
            project_dir,
            f"{dem_name}_{suffix}.{extension}",
        )

    return file_path


def minmax_scale(array: np.ndarray) -> np.ndarray:
    """
    Min-max scale an array to have values between 0 and 1.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array to scale.

    Returns
    -------
    scaled_array : `numpy.ndarray`
        Min-max scaled array.
    """
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))


def df_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 columns to float32.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame to convert.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame with float64 columns converted to float32.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def get_WhiteboxTools(
    verbose: bool = True,
    compress: bool = True,
    callback: callable = wbt_callback,
    whitebox_exe_dir: str | PathLike = None,
):
    """
    Get preconfigured WhiteboxTools instance.

    Parameters
    ----------
    verbose : `bool`, optional
        Verbose mode. Default is False.
    compress : `bool`, optional
        Compress rasters. Default is True.
    callback : `callable`, optional
        Callback function. Default is wbt_callback, which prints all output
        except for progress percentage updates.
    whitebox_exe_dir : `str` | `os.PathLike`, optional
        Path to WhiteboxTools executable directory. If None, the default
        installation will be used. Use this if you compiled WhiteboxTools
        from source, e.g. on an HPC. Default is None.

    Returns
    -------
    wbt : `WhiteboxTools`
        WhiteboxTools instance.
    """
    wbt = WhiteboxTools()
    # whitebox_exe_dir = "/path/to/whitebox/bin"
    whitebox_exe_dir = os.getenv("WBT_PATH", None)
    if whitebox_exe_dir:
        wbt.set_whitebox_dir(whitebox_exe_dir)
    wbt.set_verbose_mode(verbose)
    wbt.set_compress_rasters(compress)
    wbt.set_default_callback(callback)
    return wbt


def get_combined_cost(
    weights_arrays: list[tuple[float, np.ndarray]],
    return_reciprocal: bool = False,
) -> np.ndarray:
    """
    Get combined cost array of the form:
    combined_cost = 1 / (w1 * array1 + w2 * array2 + ... + wn * arrayn)

    Parameters
    ----------
    weights_arrays : `list`
        List of (weight, array) tuples. dtype: (`float`, `numpy.ndarray`)
    return_reciprocal : `bool`, optional
        Return reciprocal of combined cost. Default is False.

    Returns
    -------
    combined_cost : `numpy.ndarray`
        Combined cost array.
    """
    cost_repciprocal = sum(
        weight * array for weight, array in weights_arrays if array is not None
    )
    if return_reciprocal:
        return cost_repciprocal
    else:
        return 1 / cost_repciprocal


# Gaussian Filter
def simple_gaussian_smoothing(
    inputDemArray: np.ndarray,
    kernelWidth,
    diffusionSigmaSquared: float,
) -> np.ndarray:
    """
    smoothing input array with gaussian filter
    Code is vectorized for efficiency Harish Sangireddy

    Parameters
    ----------
    inputDemArray : `numpy.ndarray`
        Array of input DEM values.
    kernelWidth :
        Width of Gaussian kernel.
    diffusionSigmaSquared : `float`
        Diffusion sigma squared.

    Returns
    -------
    smoothedDemArray : `numpy.ndarray`
        Array of smoothed DEM values.
    """
    [Ny, Nx] = inputDemArray.shape
    halfKernelWidth = int((kernelWidth - 1) / 2)
    # Make a ramp array with 5 rows each containing [-2, -1, 0, 1, 2]
    x = np.linspace(-halfKernelWidth, halfKernelWidth, kernelWidth)
    y = x
    xv, yv = np.meshgrid(x, y)
    gaussianFilter = np.exp(
        -(xv**2 + yv**2) / (2 * diffusionSigmaSquared)
    )  # 2D Gaussian
    gaussianFilter = gaussianFilter / np.sum(gaussianFilter)  # Normalize
    print(inputDemArray[0, 0:halfKernelWidth])
    xL = np.nanmean(inputDemArray[:, 0:halfKernelWidth], axis=1)
    print(f"xL: {xL}")
    xR = np.nanmean(inputDemArray[:, Nx - halfKernelWidth : Nx], axis=1)
    print(f"xR: {xR}")
    part1T = np.vstack((xL, xL))
    part1 = part1T.T
    part2T = np.vstack((xR, xR))
    part2 = part2T.T
    eI = np.hstack((part1, inputDemArray, part2))
    xU = np.nanmean(eI[0:halfKernelWidth, :], axis=0)
    xD = np.nanmean(eI[Ny - halfKernelWidth : Ny, :], axis=0)
    part3 = np.vstack((xU, xU))
    part4 = np.vstack((xD, xD))
    # Generate the expanded DTM array, 4 pixels wider in both x,y directions
    eI = np.vstack((part3, eI, part4))
    # The 'valid' option forces the 2d convolution to clip 2 pixels off
    # the edges NaNs spread from one pixel to a 5x5 set centered on
    # the NaN
    fillvalue = np.nanmean(inputDemArray)
    smoothedDemArray = convolve2d(eI, gaussianFilter, "valid")
    del inputDemArray, eI
    return smoothedDemArray


def anisodiff(
    img: np.ndarray,
    niter: int,
    kappa: float,
    gamma: float,
    step: tuple[float, float] = (1.0, 1.0),
    option: str = "PeronaMalik2",
) -> np.ndarray:
    """
    Anisotropic diffusion.

    Parameters
    ----------
    img : `numpy.ndarray`
        Array of input image values.
    niter : `int`
        Number of iterations.
    kappa : `float`
        Edge threshold value.
    gamma : `float`
        Time increment.
    step : `tuple`, optional
        Step size. Default is (1.0, 1.0).
    option : `str`, optional
        Diffusion option. Default is "PeronaMalik2".

    Returns
    -------
    imgout : `numpy.ndarray`
        Array of filtered image values.
    """

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    for _ in range(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        if option == "PeronaMalik2":
            # gS = gs_diff(deltaS,kappa,step1)
            # gE = ge_diff(deltaE,kappa,step2)
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[1]
        elif option == "PeronaMalik1":
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]
        # update matrices
        E = gE * deltaE
        S = gS * deltaS
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        # **above comments from original GeoNet code**
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # update the image
        mNS = np.isnan(NS)
        mEW = np.isnan(EW)
        NS[mNS] = 0
        EW[mEW] = 0
        NS += EW
        mNS &= mEW
        NS[mNS] = np.nan
        imgout += gamma * NS
    return imgout


def lambda_nonlinear_filter(
    nanDemArray: np.ndarray,
    demPixelScale: float,
    smoothing_quantile: float,
) -> float:
    """
    Compute the threshold lambda used in Perona-Malik nonlinear filtering.

    Parameters
    ----------
    nanDemArray : `numpy.ndarray`
        Array of input DEM values.
    demPixelScale : `float`
        Pixel scale of DEM.
    smoothing_quantile : `float`
        Quantile for calculating Perona-Malik nonlinear filter edge threshold
        value (kappa).

    Returns
    -------
    edgeThresholdValue : `float`
        Edge threshold value.
    """

    print("Computing slope of raw DTM")
    slopeXArray, slopeYArray = np.gradient(nanDemArray, demPixelScale)
    slopeMagnitudeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
    print(("DEM slope array shape:"), slopeMagnitudeDemArray.shape)

    # plot the slope DEM array
    # if defaults.doPlot == 1:
    #    raster_plot(slopeMagnitudeDemArray, 'Slope of unfiltered DEM')

    # Computation of the threshold lambda used in Perona-Malik nonlinear
    # filtering. The value of lambda (=edgeThresholdValue) is given by the 90th
    # quantile of the absolute value of the gradient.

    print("Computing lambda = q-q-based nonlinear filtering threshold")
    slopeMagnitudeDemArray = slopeMagnitudeDemArray.flatten()
    slopeMagnitudeDemArray = slopeMagnitudeDemArray[
        ~np.isnan(slopeMagnitudeDemArray)
    ]
    print("DEM smoothing Quantile:", smoothing_quantile)
    edgeThresholdValue = (
        mquantiles(
            np.absolute(slopeMagnitudeDemArray),
            smoothing_quantile,
        )
    ).item()
    print("Edge Threshold Value:", edgeThresholdValue)
    return edgeThresholdValue


def compute_dem_slope(
    dem: np.ndarray,
    pixel_scale: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute slope of DEM.

    Parameters
    ----------
    dem : `numpy.ndarray`
        DEM on which to calculate slope.
    pixel_scale : `float`
        Pixel scale of DEM.

    Returns
    -------
    slopeDemArray : `numpy.ndarray`
        Array of DEM slope values.
    """

    slopeYArray, slopeXArray = np.gradient(dem, pixel_scale)
    slopeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
    if verbose:
        # Computation of statistics of slope
        print(" slope statistics")
        print(
            " min angle:",
            np.arctan(np.nanpercentile(slopeDemArray, 0.1)) * 180 / np.pi,
        )
        print(
            " max angle:",
            np.arctan(np.nanpercentile(slopeDemArray, 99.9)) * 180 / np.pi,
        )
        print(" mean slope:", np.nanmean(slopeDemArray))
        print(" stdev slope:", np.nanstd(slopeDemArray))
    slopeDemArray[np.isnan(dem)] = np.nan
    return slopeDemArray


def compute_dem_curvature(
    filteredDemArray: np.ndarray,
    pixelDemScale: float,
    curvatureCalcMethod: str,
) -> np.ndarray:
    """
    Compute curvature of DEM.

    Parameters
    ----------
    filteredDemArray : `numpy.ndarray`
        Array of DEM values.
    pixelDemScale : `float`
        Pixel scale of DEM.
    curvatureCalcMethod : `str`, optional
        Method for calculating curvature. Options include:
        - "geometric": TODO: detailed description
        - "laplacian": TODO: detailed description
        Default is "geometric".

    Returns
    -------
    curvatureDemArray : `numpy.ndarray`
        Array of DEM curvature values.
    """

    # OLD:
    # gradXArray, gradYArray = np.gradient(demArray, pixelDemScale)
    # NEW:
    gradYArray, gradXArray = np.gradient(filteredDemArray, pixelDemScale)

    slopeArrayT = np.sqrt(gradXArray**2 + gradYArray**2)
    if curvatureCalcMethod == "geometric":
        # Geometric curvature
        print(" using geometric curvature")
        gradXArrayT = np.divide(gradXArray, slopeArrayT)
        gradYArrayT = np.divide(gradYArray, slopeArrayT)
    elif curvatureCalcMethod == "laplacian":
        # do nothing..
        print(" using laplacian curvature")
        gradXArrayT = gradXArray
        gradYArrayT = gradYArray

    # NEW:
    tmpy, gradGradXArray = np.gradient(gradXArrayT, pixelDemScale)
    gradGradYArray, tmpx = np.gradient(gradYArrayT, pixelDemScale)

    curvatureDemArray = gradGradXArray + gradGradYArray
    curvatureDemArray[np.isnan(curvatureDemArray)] = 0
    del tmpy, tmpx
    # Computation of statistics of curvature
    print(" curvature statistics")
    tt = curvatureDemArray[~np.isnan(curvatureDemArray)]
    print(" non-nan curvature cell number:", tt.shape[0])
    finiteCurvatureDemList = curvatureDemArray[np.isfinite(curvatureDemArray)]
    print(" non-nan finite curvature cell number:", end=" ")
    finiteCurvatureDemList.shape[0]
    curvatureDemMean = np.nanmean(finiteCurvatureDemList)
    curvatureDemStdDevn = np.nanstd(finiteCurvatureDemList)
    print(" mean: ", curvatureDemMean)
    print(" standard deviation: ", curvatureDemStdDevn)
    curvatureDemArray[np.isnan(filteredDemArray)] = np.nan
    return curvatureDemArray


def get_skeleton(
    inputArray1: np.ndarray,
    threshold1: float,
    inputArray2: np.ndarray = None,
    threshold2: float = None,
) -> np.ndarray:
    """
    Creates a channel skeleton by thresholding grid measures such as flow or curvature.
    Can operate on single or dual thresholds depending on input.

    Parameters
    ----------
    inputArray1 : `numpy.ndarray`
        First array of input values.
    threshold1 : float
        Threshold value for the first input array.
    inputArray2 : `numpy.ndarray`, optional
        Second array of input values. If provided, dual thresholding will be applied.
    threshold2 : `float`, optional
        Threshold value for the second input array, required if inputArray2 is provided.

    Returns
    -------
    skeletonArray : `numpy.ndarray`
        Binary array (dtype: int) of skeleton values.
    """

    mask1 = inputArray1 > threshold1

    if inputArray2 is not None and threshold2 is not None:
        mask2 = inputArray2 > threshold2
        skeletonArray = (mask1 & mask2).astype(int)
    else:
        skeletonArray = mask1.astype(int)

    return skeletonArray


@jit(nopython=True, parallel=True)
def get_fmm_points(
    basins,
    outlets,
    basin_elements,
    area_threshold,
):
    fmmX = []
    fmmY = []
    basins_ravel = basins.ravel()
    n_pixels = basins.size
    for label in prange(outlets.shape[1]):
        numelements = np.sum(basins_ravel == (label + 1))
        percentBasinArea = numelements * 100.00001 / n_pixels
        if (percentBasinArea > area_threshold) and (
            numelements > basin_elements
        ):
            fmmX.append(outlets[1, label])
            fmmY.append(outlets[0, label])

    return np.array([fmmY, fmmX])


def get_ram_usage() -> str:
    """
    Get the current system RAM usage and return it in a human-readable format.

    Returns:
    --------
    str
        A string representing the current RAM usage in GB with 2 decimal places.
    """
    # Fetch RAM usage information
    mem = psutil.virtual_memory()
    # Convert bytes to GB for more human-friendly reading
    avail_memory_gb = mem.available / (1024**3)
    total_memory_gb = mem.total / (1024**3)
    total_less_avail = total_memory_gb - avail_memory_gb

    return f"RAM usage: {total_less_avail:.2f}/{total_memory_gb:.2f} GB ({mem.percent}%)"


def fast_marching(
    fmm_start_points,
    basins,
    mfd_fac,
    cost_function,
):
    # Fast marching
    print("Performing fast marching")
    # Do fast marching for each sub basin
    geodesic_distance = np.zeros_like(basins)
    geodesic_distance[geodesic_distance == 0] = np.inf
    fmm_total_iter = len(fmm_start_points[0])
    for i in range(fmm_total_iter):
        basinIndexList = basins[fmm_start_points[0, i], fmm_start_points[1, i]]
        # print("start point :", fmm_start_points[:, i])
        maskedBasin = np.zeros_like(basins)
        maskedBasin[basins == basinIndexList] = 1
        maskedBasinFAC = np.zeros_like(basins)
        maskedBasinFAC[basins == basinIndexList] = mfd_fac[
            basins == basinIndexList
        ]

        phi = np.zeros_like(cost_function)
        speed = np.zeros_like(cost_function)
        phi[maskedBasinFAC != 0] = 1
        speed[maskedBasinFAC != 0] = cost_function[maskedBasinFAC != 0]
        phi[fmm_start_points[0, i], fmm_start_points[1, i]] = -1
        del maskedBasinFAC
        # print RAM usage per iteration
        print(f"FMM iteration {i+1}/{fmm_total_iter}: {get_ram_usage()}")
        try:
            travel_time = skfmm.travel_time(phi, speed, dx=0.01)
        except IOError as e:
            print("Error in calculating skfmm travel time")
            print("Error in catchment: ", basinIndexList)
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            # setting travel time to empty array
            travel_time = np.nan * np.zeros_like(cost_function)
            # if defaults.doPlot == 1:
            #    raster_point_plot(speed, fmm_start_points[:,i],
            #                      'speed basin Index'+str(basinIndexList))
            # plt.contour(speed,cmap=cm.coolwarm)
            #    raster_point_plot(phi, fmm_start_points[:,i],
            #                      'phi basin Index'+str(basinIndexList))
        except ValueError:
            print("Error in calculating skfmm travel time")
            print("Error in catchment: ", basinIndexList)
            print("That was not a valid number")
        geodesic_distance[maskedBasin == 1] = travel_time[maskedBasin == 1]
    geodesic_distance[maskedBasin == 1] = travel_time[maskedBasin == 1]
    geodesic_distance[geodesic_distance == np.inf] = np.nan
    # Plot the geodesic array
    # if defaults.doPlot == 1:
    #    geodesic_contour_plot(geodesic_distance,
    #                          'Geodesic distance array (travel time)')
    return geodesic_distance


def get_channel_heads(
    combined_skeleton: np.ndarray,
    geodesic_distance: np.ndarray,
    channel_head_median_dist: int,
    max_channel_heads: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Through the histogram of sorted_label_counts
    (skeletonNumElementsList minus the maximum value which
    corresponds to the largest connected element of the skeleton) we get the
    size of the smallest elements of the skeleton, which will likely
    correspond to small isolated convergent areas. These elements will be
    excluded from the search of end points.
    """
    # Locating end points
    print("Locating skeleton end points")
    structure = np.ones((3, 3))
    labeled, num_labels = ndimage.label(combined_skeleton, structure=structure)
    print("Counting the number of elements of each connected component")
    lbls = np.arange(1, num_labels + 1)
    label_counts = ndimage.labeled_comprehension(
        input=combined_skeleton,
        labels=labeled,
        index=lbls,
        func=np.count_nonzero,
        out_dtype=int,
        default=0,
    )
    sorted_label_counts = np.sort(label_counts)
    num_bins = int(np.floor(np.sqrt(len(sorted_label_counts))))
    histarray, bin_edges = np.histogram(sorted_label_counts[:-1], num_bins)
    # if defaults.doPlot == 1:
    #     raster_plot(labeled, "Skeleton Labeled Array elements Array")
    # Create skeleton gridded array
    labeled_set, label_indices = np.unique(labeled, return_inverse=True)
    skeleton_gridded_array = np.array(
        [label_counts[x - 1] for x in labeled_set]
    )[label_indices].reshape(labeled.shape)
    # if defaults.doPlot == 1:
    #     raster_plot(
    #         skeleton_gridded_array, "Skeleton Num elements Array"
    #     )
    # Elements smaller than count_threshold are not considered in the
    # channel_heads detection
    count_threshold = bin_edges[2]
    print(f"Skeleton region size threshold: {str(count_threshold)}")
    # Scan the array for finding the channel heads
    print("Continuing to locate skeleton endpoints")
    channel_heads = []
    nrows, ncols = combined_skeleton.shape
    channel_heads = jit_channel_heads(
        labeled,
        skeleton_gridded_array,
        geodesic_distance,
        nrows,
        ncols,
        count_threshold,
        channel_head_median_dist,
        max_channel_heads,
    )
    print(f'number of channel heads: {len(channel_heads)}')
    channel_heads = np.transpose(channel_heads)
    ch_rows = channel_heads[0]
    ch_cols = channel_heads[1]
    return ch_rows, ch_cols


@jit(nopython=True)
def jit_channel_heads(
    labeled,
    skeleton_gridded_array,
    geodesic_distance,
    nrows,
    ncols,
    count_threshold,
    channel_head_median_dist,
    max_channel_heads,
):
    # pre-allocate array of channel heads
    channel_heads = np.zeros((max_channel_heads, 2), dtype=np.int32)
    ch_count = 0

    for i in range(nrows):
        for j in range(ncols):
            if (
                labeled[i, j] != 0
                and skeleton_gridded_array[i, j] >= count_threshold
            ):
                my, py, mx, px = i - 1, nrows - i, j - 1, ncols - j
                xMinus, xPlus = min(channel_head_median_dist, mx), min(
                    channel_head_median_dist, px
                )
                yMinus, yPlus = min(channel_head_median_dist, my), min(
                    channel_head_median_dist, py
                )

                search_geodesic_box = geodesic_distance[
                    i - yMinus : i + yPlus + 1, j - xMinus : j + xPlus + 1
                ]
                search_skeleton_box = labeled[
                    i - yMinus : i + yPlus + 1, j - xMinus : j + xPlus + 1
                ]

                v = search_skeleton_box == labeled[i, j]
                v1 = v & (search_geodesic_box > geodesic_distance[i, j])

                if not np.any(v1):
                    channel_heads[ch_count] = [i, j]
                    ch_count += 1
                # Trim to the actual number of channel heads found
                # warn if max_channel_heads was exceeded
                if ch_count > max_channel_heads:
                    print(
                        f"Warning: max_channel_heads ({max_channel_heads}) exceeded. "
                        "Consider increasing max_channel_heads"
                    )
    return channel_heads[:ch_count]


def get_endpoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Extract start and end points directly into the GeoDataFrame
    gdf["START_X"] = gdf.geometry.apply(lambda geom: geom.coords[0][0])
    gdf["START_Y"] = gdf.geometry.apply(lambda geom: geom.coords[0][1])
    gdf["END_X"] = gdf.geometry.apply(lambda geom: geom.coords[-1][0])
    gdf["END_Y"] = gdf.geometry.apply(lambda geom: geom.coords[-1][1])

    # Create a RiverID column
    gdf["RiverID"] = range(1, len(gdf) + 1)

    # Select and order the columns for the output CSV
    endpoints = gdf[["RiverID", "START_X", "START_Y", "END_X", "END_Y"]]

    return endpoints


@jit(nopython=True, parallel=True)
def jit_binary_hand(
    dem: np.ndarray,
    flowline_raster: np.ndarray,
    nodata: float,
):
    """
    Numba JIT-compiled function for creating a binary HAND array.
    """
    # create a binary HAND array
    d_x = np.asarray([-1, -1, 0, 1, 1, 1, 0, -1])
    d_y = np.asarray([0, -1, -1, -1, 0, 1, 1, 1])
    # g_x = np.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    # g_y = np.asarray([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    # r_x = np.full_like(dem, nodata)
    # r_y = np.full_like(dem, nodata)

    distanceArray = np.full_like(dem, nodata)
    allocationArray = np.zeros_like(dem)
    distanceArray = np.where(flowline_raster == 1, 0, np.inf)
    allocationArray = np.where(flowline_raster == 1, dem, np.inf)

    for row in prange(distanceArray.shape[0]):
        for col in prange(distanceArray.shape[1]):
            z = distanceArray[row, col]
            if z != 0:
                z_min = np.inf
                which_cell = 0
                for i in range(4):
                    x = col + d_x[i]
                    y = row + d_y[i]
                    if (
                        (x >= 0)
                        and (x < distanceArray.shape[1])
                        and (y >= 0)
                        and (y < distanceArray.shape[0])
                    ):
                        z2 = distanceArray[y, x]
                        if z2 != nodata:
                            if i == 0:
                                h = 1
                                # h = 2*r_x[y,x]+1
                            elif i == 1:
                                h = 1.414
                                # h = 2*(r_x[y,x]+r_y[y,x]+1)
                            elif i == 2:
                                h = 1
                                # h = 2*r_y[y,x]+1
                            elif i == 3:
                                h = 1.414
                                # h = 2*(r_x[y,x]+r_y[y,x]+1)
                            z2 += h
                            if z2 < z_min:
                                z_min = z2
                                which_cell = i
                if z_min < z:
                    distanceArray[row, col] = z_min
                    x = col + d_x[which_cell]
                    y = row + d_y[which_cell]
                    # r_x[row, col] = r_x[y,x] + g_x[which_cell]
                    # r_y[row, col] = r_y[y,x] + g_y[which_cell]
                    allocationArray[row, col] = allocationArray[y, x]
    for row in range(distanceArray.shape[0] - 1, -1, -1):
        for col in range(distanceArray.shape[1] - 1, -1, -1):
            z = distanceArray[row, col]
            if z != 0:
                z_min = np.inf
                which_cell = 0
                for i in range(4, 8):
                    x = col + d_x[i]
                    y = row + d_y[i]
                    if (
                        (x >= 0)
                        and (x < distanceArray.shape[1])
                        and (y >= 0)
                        and (y < distanceArray.shape[0])
                    ):
                        z2 = distanceArray[y, x]
                        if z2 != nodata:
                            if i == 4:
                                h = 1
                                # h = 2*r_x[y,x]+1
                            elif i == 5:
                                h = 1.414
                                # h = 2*(r_x[y,x]+r_y[y,x]+1)
                            elif i == 6:
                                h = 1
                                # h = 2*r_y[y,x]+1
                            elif i == 7:
                                h = 1.414
                                # h = 2*(r_x[y,x]+r_y[y,x]+1)
                            z2 += h
                            if z2 < z_min:
                                z_min = z2
                                which_cell = i
                if z_min < z:
                    distanceArray[row, col] = z_min
                    x = col + d_x[which_cell]
                    y = row + d_y[which_cell]
                    # r_x[row, col] = r_x[y,x] + g_x[which_cell]
                    # r_y[row, col] = r_y[y,x] + g_y[which_cell]
                    allocationArray[row, col] = allocationArray[y, x]
    allocationArray = np.where(dem == nodata, nodata, allocationArray)
    allocationArray = np.where(
        np.isinf(allocationArray), nodata, allocationArray
    )
    binary_hand = np.where(allocationArray < dem, 0, 1)
    return binary_hand


def get_binary_hand(
    flowline: gpd.GeoDataFrame,
    dem: np.ndarray,
    dem_profile: rio.profiles.Profile | dict,
) -> np.ndarray:
    """
    Create a binary HAND array.

    Parameters
    ----------
    flowline : `geopandas.GeoDataFrame`
        Flowline GeoDataFrame.
    dem : `numpy.ndarray`
        Array of DEM values.
    dem_profile : `rasterio.profiles.Profile` | `dict`
        Raster profile.

    Returns
    -------
    binary_hand : `numpy.ndarray`
        Binary HAND array.
    """

    # convert flowline to array with dem as reference
    flowline_raster = rasterize_flowline(
        flowline_gdf=flowline, ref_profile=dem_profile
    )

    binary_hand = jit_binary_hand(dem, flowline_raster, dem_profile["nodata"])

    return binary_hand


def reconstruct_channel(
    stream_cell: dict[int, list[np.ndarray]],
    numberOfEndPoints: int,
) -> dict[int, list[np.ndarray] | list[int]]:
    # Initialize an empty list to store (row, col) tuples
    rowcol_list = []
    # Loop through each stream path and extend the list with (row, col)tuples
    for i in range(numberOfEndPoints):
        rowcol_list.extend(zip(stream_cell[i][0], stream_cell[i][1]))
    # Create a DataFrame from the list of tuples
    df_channel = pd.DataFrame(rowcol_list, columns=["row", "col"])
    # Group by row and col to calculate the size (count of occurrences) of each pair
    size_sr = df_channel.groupby(["row", "col"]).size().to_dict()

    new_stream_cell = {}
    StartpointList = []
    k = 0
    for i in range(numberOfEndPoints):
        for j in range(stream_cell[i][0].size):
            # append cell to the starting point list if it has an index of zero
            row = stream_cell[i][0, j]
            col = stream_cell[i][1, j]
            if j == 0:
                if i != 0:
                    k += 1
                StartpointList.append([row, col])
                new_stream_cell[k] = [[row], [col]]
            # Checking if the path cell at the current iteration is the same as the
            # previous iteration. If it is, append to the 'new_stream_cell'.
            else:
                prev_row = stream_cell[i][0, j - 1]
                prev_col = stream_cell[i][1, j - 1]
                if size_sr[row, col] == size_sr[prev_row, prev_col]:
                    new_stream_cell[k][0].append(row)
                    new_stream_cell[k][1].append(col)

                # When this condition is satisfied, additional start points are added to the
                # 'StartpointList' variable. This leads to unwanted segmentation of the
                # extracted channel network.
                else:
                    if [row, col] not in StartpointList:
                        continue
                    else:
                        new_stream_cell[k][0].append(row)
                        new_stream_cell[k][1].append(col)
                        k += 1
                        break
    paths_list = []
    keyList = []
    for key in list(new_stream_cell.keys()):
        paths_list.append(np.asarray(new_stream_cell[key]))
        keyList.append(key)
    print(f"Number of endpoints: {len(StartpointList)}")

    return new_stream_cell


def get_channel_network(
    cost: np.ndarray,
    df_flowline_path: str | PathLike,
    transform: rio.transform.Affine,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    df_flowline = pd.read_csv(df_flowline_path)
    channel_network = np.zeros_like(cost, dtype="uint8")
    stream_cell = {}
    total_endpoints = len(df_flowline)
    for i, row in df_flowline.iterrows():
        startXCoord = float(row["START_X"])
        startYCoord = float(row["START_Y"])
        endXCoord = float(row["END_X"])
        endYCoord = float(row["END_Y"])
        startIndexY, startIndexX = rowcol(transform, startXCoord, startYCoord)
        stopIndexY, stopIndexX = rowcol(transform, endXCoord, endYCoord)
        print(f"Creating path {i+1}/{total_endpoints}: {get_ram_usage()}")
        indices, _ = route_through_array(
            cost,
            (startIndexY, startIndexX),
            (stopIndexY, stopIndexX),
            geometric=True,
            fully_connected=True,
        )
        indices = np.array(indices).T
        stream_cell[i] = indices
        channel_network[indices[0], indices[1]] = 1
        del indices
    new_stream_cell = reconstruct_channel(stream_cell, total_endpoints)

    stream_keys = [key for key in new_stream_cell.keys()]
    stream_rowcol = [np.asarray(path) for path in new_stream_cell.values()]

    return channel_network, stream_rowcol, stream_keys


# # snap/interpolate point along line method
# # currently not used
# def old_split_line(
#     line: LineString,
#     segment_length: int | float,
# ) -> list[LineString]:
#     # create segment endpoints along the line using specified segment length
#     distances = np.arange(segment_length, line.length, segment_length)

#     # return original line if segment length is longer than line length
#     if distances.size == 0:
#         return [line]

#     else:
#         split_points = [line.interpolate(distance) for distance in distances]
#         # Initialize the list of split lines
#         split_lines = []
#         current_line = line

#         for i, point in enumerate(split_points):
#             # allow segment distances to v
#             # tolerance should be equal to pixel scale of DEM?
#             snapped_point = snap(point, current_line, tolerance=1)
#             # Attempt to split the current line segment at the point
#             split_result = split(current_line, snapped_point)
#             # Handling the split result
#             if len(split_result.geoms) > 1:
#                 # The line was successfully split; update current_line and store the first part
#                 # The second part becomes the new line to split further
#                 current_line = split_result.geoms[1]
#                 # The first part is stored
#                 split_lines.append(split_result.geoms[0])
#             else:
#                 print(f"Unsuccessful snap at HydroID {i+1}")

#         # Add the remaining part of the line after the last split
#         split_lines.append(current_line)

#         return split_lines


# this function is less efficient than old_split_line above because
# it loops through every point in the line, but is more robust to
# errors. old_split_line will fail if the snapping tolerance is incorrect
# (seems like the tolerance should equal the pixel scale of the DEM)
def split_line(
    line: LineString,
    segment_length: int | float,
) -> list[LineString]:

    # constituent points of LineString
    points = list(line.coords)
    # store segments in a list
    line_segments = []
    # add first segment point
    segment_points = [points[0]]

    for i in range(1, len(points)):
        # potential new segment
        segment = LineString(segment_points + [points[i]])
        if segment.length > segment_length:
            # if adding current point causes current segment length to
            # exceed segment_length, add current segment to segments
            line_segments.append(LineString(segment_points))
            segment_points = [points[i - 1], points[i]]  # Start a new segment
        else:
            # If not, add the current point to the current segment
            segment_points.append(points[i])

    # Add the last segment if it has more than one point
    if len(segment_points) > 1:
        line_segments.append(LineString(segment_points))

    return line_segments


def split_network(
    input_gdf: gpd.GeoDataFrame,
    segment_length: int | float,
) -> list[LineString]:

    all_segments = []
    for _, row in input_gdf.iterrows():
        line = row.geometry
        segmented_line = split_line(line, segment_length)
        # assert that there are no overlapping segments
        msg = "Overlapping segments detected"
        assert linemerge(segmented_line).is_simple, msg
        all_segments.extend(segmented_line)

    return all_segments


def segment_el_perc_dist(dem_path, geometry, percentage):
    """
    Return the elevation at a given percentage distance along a line.
    """
    distance = geometry.length * percentage
    point = geometry.interpolate(distance)
    with rio.open(dem_path) as ds:
        z = list(ds.sample([(point.x, point.y)]))[0]
    return z


def slope_10_85(dem_path, geometry):
    """
    Calculate the slope between the points at 10% and 85% along
    the length of a line.
    """
    run = 0.75 * geometry.length
    assert run > 0, "Segment length must be greater than 0"
    h_85 = segment_el_perc_dist(dem_path, geometry, 0.85)
    h_10 = segment_el_perc_dist(dem_path, geometry, 0.10)
    rise = h_85 - h_10
    return -1 * rise / run


def get_river_attributes(
    dem_path: str | PathLike,
    segment_catchments: np.ndarray,
    nwm_catchments: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    profile: rio.profiles.Profile | dict,
    min_slope: float,
) -> pd.DataFrame:

    # # calculate area from vectorized catchments
    # features = list(
    #     {"properties": {"HYDROID": v}, "geometry": s}
    #     for (s, v) in shapes(
    #         segment_catchments,
    #         connectivity=8,  # avoids isolated single pixel catchments
    #         transform=profile["transform"],
    #     )
    #     if v > 0
    # )
    # segment_catchments_vector = gpd.GeoDataFrame.from_features(
    #     features, crs=segmented_channel_network.crs
    # )
    # segment_catchments_vector["AreaSqKm"] = (
    #     segment_catchments_vector.area / 1e6
    # )
    # if write_segment_catchments_features:
    #     self.segment_catchments_path = t.get_file_path(
    #         custom_path=None,
    #         project_dir=self.project_dir,
    #         dem_name=self.dem_path.stem,
    #         suffix="segment_catchments",
    #         extension=vector_extension,
    #     )
    #     segment_catchments_vector.to_file(self.segment_catchments_path)
    #     print(
    #         f"Segment catchments features written to {str(self.segment_catchments_path)}"
    #     )
    # river_attributes_pre = pd.DataFrame(
    #     segment_catchments_vector.drop(columns="geometry")
    # )

    # calculate area directly from raster
    hydroid, counts = np.unique(segment_catchments, return_counts=True)
    # get area in square kilometers
    msg = "segment catchments raster crs must have defined linear units"
    assert profile["crs"].linear_units != "unknown", msg
    pixel_area = abs(
        profile["transform"].a
        * profile["transform"].e
        * (profile["crs"].linear_units_factor[1] ** 2)
    )
    area_sqkm = (counts * pixel_area) / 1e6
    river_attributes_pre = pd.DataFrame(
        {"HYDROID": hydroid, "Area_km2": area_sqkm}
    )
    river_attributes_pre = river_attributes_pre[
        river_attributes_pre["HYDROID"] != profile["nodata"]
    ]

    ra = river_attributes_pre.copy()
    for _, row in segments.iterrows():
        slope = slope_10_85(dem_path, row.geometry)
        if slope < min_slope:
            print(f"segment HYDROID {row['HYDROID']} has a slope of {slope}")
            print(f"manually setting slope to {min_slope}")
            slope = min_slope
        hydroid = row["HYDROID"]
        ra.loc[ra["HYDROID"] == hydroid, "Slope"] = slope
        ra.loc[ra["HYDROID"] == hydroid, "Length_km"] = (
            row.geometry.length / 1e3
        )
    ra = ra.sort_values(by="HYDROID")
    # set float64 datatypes to float32
    ra = df_float64_to_float32(ra)
    # get NWM COMID (catchment ID) corresponding to each HYDROID (segment)
    segments.geometry = segments.centroid
    # join COMID to HYDROID if HYDROID centroid is within COMID
    joined_comid = gpd.sjoin(
        segments,
        nwm_catchments,
        how="left",
        predicate="within",
    )
    joined_comid = joined_comid[["HYDROID", "FEATUREID"]]
    joined_comid = joined_comid.rename(columns={"FEATUREID": "COMID"})
    # join COMID to river attributes
    ra = pd.merge(
        ra,
        joined_comid,
        on="HYDROID",
        how="left",
    )

    return ra


@jit(nopython=True, parallel=True)
def process_cells(
    hand,
    segment_catchments,
    slope,
    hid_dict_array,
    heights,
    cell_area,
):
    nheight = len(heights)
    ncatch = len(hid_dict_array[:, 0])
    CellCount = np.zeros((nheight, ncatch), dtype=np.int32)
    SurfaceArea = np.zeros((nheight, ncatch), dtype=np.float32)
    BedArea = np.zeros((nheight, ncatch), dtype=np.float32)
    Volume = np.zeros((nheight, ncatch), dtype=np.float32)

    # loop through each row i, col j
    for i in prange(hand.shape[0]):
        for j in prange(hand.shape[1]):
            hydroid = segment_catchments[i, j]
            hydroid_index = np.searchsorted(hid_dict_array[:, 0], hydroid)
            hand_height = hand[i, j]
            slp_value = slope[i, j]
            for h_idx in range(nheight):
                if hand_height < heights[h_idx] or np.isclose(hand_height, 0.0):
                    CellCount[h_idx, hydroid_index] += 1
                    SurfaceArea[h_idx, hydroid_index] += cell_area
                    incBedArea = cell_area * np.sqrt(1 + slp_value**2)
                    BedArea[h_idx, hydroid_index] += incBedArea
                    incVolume = (heights[h_idx] - hand_height) * cell_area
                    Volume[h_idx, hydroid_index] += incVolume
    return CellCount, SurfaceArea, BedArea, Volume


def catchhydrogeo(
    hand,
    segment_catchments,
    hydroids,
    slope,
    heights,
    cell_area,
    ra,
    custom_roughness_path,
):

    hid_dict = {int(item[0]): idx for idx, item in enumerate(hydroids)}
    hid_dict_array = np.array(list(hid_dict.items()), dtype=np.int32)
    CellCount, SurfaceArea, BedArea, Volume = process_cells(
        hand,
        segment_catchments,
        slope,
        hid_dict_array,
        heights,
        cell_area,
    )
    # prepare data for DataFrame

    data = []
    for i in range(len(hydroids)):
        hydroid = int(hydroids[i])
        for h_idx in range(len(heights)):
            row = {
                "HYDROID": hydroid,
                "Stage_m": round(heights[h_idx], 4),
                "NumCells": CellCount[h_idx, i],
                "SurfaceArea_m2": SurfaceArea[h_idx, i],
                "BedArea_m2": BedArea[h_idx, i],
                "Volume_m3": Volume[h_idx, i],
            }
            data.append(row)

    # Create DataFrame
    src_df = pd.DataFrame(data)
    src_df["NumCells"] = src_df["NumCells"].astype(int)

    # join river attributes on HYDROID
    src_df = pd.merge(src_df, ra, on="HYDROID", how="left")
    # join Manning's roughness on COMID
    if custom_roughness_path is not None:
        roughness = pd.read_csv(custom_roughness_path)
    else:
        with resources.path("pygeoflood.data", "COMID_Roughness.csv") as f:
            roughness = pd.read_csv(f)
    src_df = pd.merge(src_df, roughness, on="COMID", how="left")
    src_df["TopWidth_m"] = src_df["SurfaceArea_m2"] / src_df["Length_km"] / 1e3
    src_df["WettedPerimeter_m"] = (
        src_df["BedArea_m2"] / src_df["Length_km"] / 1e3
    )
    src_df["WetArea_m2"] = src_df["Volume_m3"] / src_df["Length_km"] / 1e3
    src_df["HydraulicRadius_m"] = (
        src_df["WetArea_m2"] / src_df["WettedPerimeter_m"]
    )
    src_df["HydraulicRadius_m"] = src_df["HydraulicRadius_m"].fillna(0)
    src_df["Discharge_cms"] = (
        src_df["WetArea_m2"]
        * np.power(src_df["HydraulicRadius_m"], 2 / 3)
        * np.sqrt(src_df["Slope"])
        / src_df["Roughness"]
    )
    src_df["FloodAreaRatio"] = (
        src_df["SurfaceArea_m2"] / src_df["Area_km2"] / 1e6
    )

    msg = "Empty DataFrame, check river attributes and make sure COMID is in COMID_Roughness csv"
    assert src_df["Discharge_cms"].isna().sum() != len(src_df), msg

    # set float64 datatypes to float32
    src_df = df_float64_to_float32(src_df)

    return src_df


def get_flood_stage(src, streamflow_forecast_path, custom_Q):
    comids = src["COMID"].unique()
    # open forecast table with xarray or pandas
    if custom_Q is None:
        if streamflow_forecast_path.suffix in [".nc", ".comp"]:
            with xr.open_dataset(streamflow_forecast_path) as ds:
                reqd_cols_provided = "streamflow" in ds.variables and (
                    "COMID" in ds.variables or "feature_id" in ds.variables
                )
                if reqd_cols_provided:
                    comid = "COMID" if "COMID" in ds.variables else "feature_id"
                    df = ds.streamflow[ds[comid].isin(comids)].to_dataframe()
                    # keep only Discharge_cms and COMID columns
                    df["COMID"] = df.index
                    df = df.reset_index(drop=True)
                    df = df.rename(columns={"streamflow": "Discharge_cms"})
                    df = df[["COMID", "Discharge_cms"]]
                else:
                    raise ValueError(
                        "NetCDF file must have COMID (or feature_id) and streamflow variables"
                    )
        elif streamflow_forecast_path.suffix == ".csv":
            df = pd.read_csv(streamflow_forecast_path)
            reqd_cols_provided = "streamflow" in df.columns and (
                "COMID" in df.columns or "feature_id" in df.columns
            )
            if reqd_cols_provided:
                comid = "COMID" if "COMID" in df.columns else "feature_id"
                # filter forecast table to only COMIDS of interest
                df = df[df[comid].isin(src["COMID"].unique())]
                # keep only streamflow and COMID columns
                df = df.rename(
                    columns={comid: "COMID", "streamflow": "Discharge_cms"}
                )
                df = df[["COMID", "Discharge_cms"]]
            else:
                raise ValueError(
                    "CSV file must have COMID (or feature_id) and streamflow column headers"
                )
    data = []
    for comid in comids:
        # assign constant streamflow to all segments if custom_Q is provided
        if custom_Q is not None:
            q = custom_Q
        else:
            q = df[df["COMID"] == comid]["Discharge_cms"].values[0]
        for hydroid in src["HYDROID"].unique():
            q_lookup = src.loc[src["HYDROID"] == hydroid]["Discharge_cms"]
            h_lookup = src.loc[src["HYDROID"] == hydroid]["Stage_m"]
            # if q is less than q_lookup[0] h = 0
            # if q is greater than q_lookup[-1] h = -9999
            h = np.interp(q, q_lookup, h_lookup, left=0, right=-9999)
            data.append(
                {
                    "HYDROID": hydroid,
                    "COMID": comid,
                    "Discharge_cms": q,
                    "Stage_m": h,
                }
            )

    return df_float64_to_float32(pd.DataFrame(data))


@jit(nopython=True)
def binary_search(arr, x):
    """
    Perform binary search of sorted array arr for x.
    Return the index of x in arr if present, else -1.
    """
    low = 0
    high = arr.size - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid]

        if mid_val < x:
            low = mid + 1
        elif mid_val > x:
            high = mid - 1
        else:
            return mid  # x is found
    return -1  # x is not found


@jit(nopython=True, parallel=True)
def jit_inun(hand, seg_catch, hydroids, stage_m):
    inun = np.empty_like(hand, dtype=np.float32)
    inun.fill(np.nan)
    for i in prange(hand.shape[0]):
        for j in prange(hand.shape[1]):
            hydroid = seg_catch[i, j]
            hand_h = hand[i, j]
            hydroid_idx = binary_search(hydroids, hydroid)
            if hydroid_idx != -1:
                h = stage_m[hydroid_idx]
            else:
                h = -9999
            if h > hand_h:
                inun[i, j] = h - hand_h
    return inun


def get_inun(hand, seg_catch, df):
    # map HYDROID to Stage_m for each HYDROID in seg_catch raster
    catch_h_mapped = (
        pd.Series(seg_catch.flatten())
        .map(df.set_index("HYDROID")["Stage_m"])
        .to_numpy()
        .reshape(seg_catch.shape)
    )
    inun = np.where(catch_h_mapped > hand, catch_h_mapped - hand, np.nan)
    return inun

def get_c_hand(dem, gage_el, opix):

    # initialize array with nan values
    inun = np.full(dem.shape, np.nan, dtype=np.float32)

    # initial inun array: 0 if (DEM ≥ gage_el) else (gage_el - DEM)
    inun = np.where(dem >= (gage_el), 0, (gage_el) - dem)

    # masked inun array: 255 if inun > 0 else 0
    inun_mask = np.where(inun == 0, 0, 255)

    # label connected regions of inundation
    regions = label(inun_mask)

    # only keep region containing the ocean pixel
    inun = np.where(regions == regions[opix], inun, 0)

    # return masked array if fed one
    if isinstance(dem, np.ma.MaskedArray):
        inun = np.ma.masked_array(inun, dem.mask)

    return inun