import numpy as np

def export_lunar_geotiff(warped_array: np.ndarray, output_path: str, transform: tuple, epsg_code: str = "EPSG:30100"):
    """
    Writes orthorectified GeoTIFF with IAU Moon 2000 coordinate reference system (EPSG:30100).
    """
    pass

def generate_pds4_label(product_id: str, output_xml_path: str, metadata: dict):
    """
    Generates standard PDS4 XML product label compliant with ISRO/NASA planetary archive.
    """
    pass
