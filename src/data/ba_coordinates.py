"""BA coordinate mapping for weather data fetching.

Maps EIA Balancing Authority codes to population-weighted latitude/longitude
coordinates for fetching weather data. Coordinates represent approximate
geographic centers of each BA's service territory.
"""
from __future__ import annotations

# Population-weighted centroids for each BA's service territory
# For multi-state BAs (PJM, MISO), coordinates represent the approximate
# geographic center weighted by load distribution
BA_COORDINATES = {
    # === EASTERN INTERCONNECTION ===
    # Priority 1 - Major ISOs
    "PJM": {"lat": 40.2, "lon": -77.0, "name": "PJM Interconnection"},
    "MISO": {"lat": 41.8, "lon": -93.6, "name": "Midcontinent Independent System Operator"},
    "ISNE": {"lat": 42.5, "lon": -72.0, "name": "ISO New England"},
    "NYIS": {"lat": 42.9, "lon": -76.0, "name": "New York Independent System Operator"},
    # Priority 2 - Large utilities
    "TVA": {"lat": 35.8, "lon": -86.0, "name": "Tennessee Valley Authority"},
    "SOCO": {"lat": 33.0, "lon": -86.8, "name": "Southern Company Services"},
    "DUK": {"lat": 35.2, "lon": -80.8, "name": "Duke Energy Carolinas"},
    "FPL": {"lat": 26.7, "lon": -81.0, "name": "Florida Power & Light"},
    "LGEE": {"lat": 38.2, "lon": -85.7, "name": "Louisville Gas and Electric and Kentucky Utilities"},
    # Priority 3 - Regional utilities
    "CPLE": {"lat": 35.8, "lon": -78.6, "name": "Duke Energy Progress East"},
    "CPLW": {"lat": 35.2, "lon": -81.0, "name": "Duke Energy Progress West"},
    "FPC": {"lat": 28.5, "lon": -81.4, "name": "Duke Energy Florida"},
    "FMPP": {"lat": 28.0, "lon": -82.0, "name": "Florida Municipal Power Pool"},
    "TEC": {"lat": 28.0, "lon": -82.4, "name": "Tampa Electric"},
    # Priority 4 - Municipal/utilities
    "GVL": {"lat": 29.7, "lon": -82.3, "name": "Gainesville Regional Utilities"},
    "HST": {"lat": 25.5, "lon": -80.4, "name": "City of Homestead"},
    "JEA": {"lat": 30.3, "lon": -81.7, "name": "JEA (Jacksonville Electric Authority)"},
    "NSB": {"lat": 29.0, "lon": -80.9, "name": "New Smyrna Beach Utilities Commission"},
    "SEC": {"lat": 28.0, "lon": -81.9, "name": "Seminole Electric Cooperative"},
    "TAL": {"lat": 30.4, "lon": -84.3, "name": "City of Tallahassee"},
    "OVEC": {"lat": 39.0, "lon": -82.4, "name": "Ohio Valley Electric Corporation"},
    "SC": {"lat": 33.8, "lon": -80.0, "name": "South Carolina Public Service Authority (Santee Cooper)"},
    "SCEG": {"lat": 34.0, "lon": -81.0, "name": "Dominion Energy South Carolina"},
    "SEPA": {"lat": 33.9, "lon": -84.5, "name": "Southeastern Power Administration"},
    # Priority 5 - Smaller BAs
    "YAD": {"lat": 35.6, "lon": -80.2, "name": "Alcoa Power Generating (Yadkin)"},
    # === TEXAS INTERCONNECTION ===
    "ERCO": {"lat": 31.0, "lon": -99.0, "name": "Electric Reliability Council of Texas (ERCOT)"},
    # === WESTERN INTERCONNECTION ===
    # Priority 1 - Major ISOs
    "CISO": {"lat": 36.8, "lon": -120.0, "name": "California Independent System Operator (CAISO)"},
    "SWPP": {"lat": 35.5, "lon": -97.5, "name": "Southwest Power Pool"},
    # Priority 2 - Large utilities
    "BPAT": {"lat": 45.6, "lon": -122.0, "name": "Bonneville Power Administration"},
    "PACW": {"lat": 43.6, "lon": -120.8, "name": "PacifiCorp West"},
    "PACE": {"lat": 40.6, "lon": -111.8, "name": "PacifiCorp East"},
    "PSCO": {"lat": 39.7, "lon": -104.9, "name": "Public Service Company of Colorado (Xcel Energy)"},
    "AZPS": {"lat": 33.5, "lon": -112.0, "name": "Arizona Public Service Company"},
    "LDWP": {"lat": 34.1, "lon": -118.2, "name": "Los Angeles Department of Water and Power"},
    "SRP": {"lat": 33.4, "lon": -111.9, "name": "Salt River Project"},
    # Priority 3 - Regional utilities
    "WACM": {"lat": 39.7, "lon": -105.0, "name": "Western Area Power Administration - Rocky Mountain Region"},
    "WALC": {"lat": 33.5, "lon": -112.1, "name": "Western Area Power Administration - Desert Southwest Region"},
    "AVA": {"lat": 47.7, "lon": -117.4, "name": "Avista Corporation"},
    "BANC": {"lat": 38.6, "lon": -121.5, "name": "Balancing Authority of Northern California"},
    "EPE": {"lat": 31.8, "lon": -106.5, "name": "El Paso Electric"},
    "IPCO": {"lat": 43.6, "lon": -116.2, "name": "Idaho Power Company"},
    "NEVP": {"lat": 36.2, "lon": -115.1, "name": "Nevada Power Company (NV Energy)"},
    "NWMT": {"lat": 47.0, "lon": -109.0, "name": "Northwestern Energy (Montana-Dakota)"},
    "PSEI": {"lat": 47.6, "lon": -122.3, "name": "Puget Sound Energy"},
    "SCL": {"lat": 47.6, "lon": -122.3, "name": "Seattle City Light"},
    "TEPC": {"lat": 32.2, "lon": -110.9, "name": "Tucson Electric Power"},
    # Priority 4 - Smaller utilities
    "WAUE": {"lat": 44.9, "lon": -93.1, "name": "Western Area Power Administration - Upper Great Plains East"},
    "WAUW": {"lat": 44.9, "lon": -100.0, "name": "Western Area Power Administration - Upper Great Plains West"},
    "SPA": {"lat": 35.5, "lon": -96.8, "name": "Southwestern Power Administration"},
    "CHPD": {"lat": 47.5, "lon": -120.5, "name": "Public Utility District No. 1 of Chelan County"},
    "DOPD": {"lat": 47.8, "lon": -120.0, "name": "Public Utility District No. 1 of Douglas County"},
    "GCPD": {"lat": 47.2, "lon": -119.3, "name": "Public Utility District No. 2 of Grant County"},
    "IID": {"lat": 32.8, "lon": -115.6, "name": "Imperial Irrigation District"},
    "TIDC": {"lat": 37.5, "lon": -120.8, "name": "Turlock Irrigation District"},
    "TPWR": {"lat": 47.3, "lon": -122.4, "name": "City of Tacoma, Department of Public Utilities Light Division"},
    # Priority 5 - Smallest BAs / Renewables
    "DEAA": {"lat": 33.4, "lon": -112.9, "name": "Arlington Valley Solar Energy II"},
    "GRID": {"lat": 37.8, "lon": -122.4, "name": "Gridforce Energy Management"},
    "GRIF": {"lat": 35.2, "lon": -114.0, "name": "Griffith Energy Services"},
    "GRMA": {"lat": 33.3, "lon": -112.0, "name": "Gila River Power"},
    "GWA": {"lat": 48.9, "lon": -111.9, "name": "NaturEner Power Watch"},
    "HGMA": {"lat": 33.8, "lon": -113.2, "name": "New Harquahala Generating Company"},
    "WWA": {"lat": 48.9, "lon": -111.9, "name": "NaturEner Wind Watch"},
    # Disabled BAs (included for completeness)
    "AEC": {"lat": 31.3, "lon": -86.3, "name": "PowerSouth Energy Cooperative"},
}


def get_ba_coordinates(ba_code: str) -> dict:
    """Get coordinates for a single Balancing Authority.
    
    Args:
        ba_code: The EIA BA code (e.g., "MISO", "PJM")
        
    Returns:
        Dictionary with "lat", "lon", and "name" keys, or empty dict if not found
    """
    return BA_COORDINATES.get(ba_code.upper(), {})


def get_all_ba_coordinates() -> dict:
    """Get all BA coordinates.
    
    Returns:
        Dictionary mapping all BA codes to their coordinates
    """
    return BA_COORDINATES.copy()


def validate_coordinates(cfg_ba_codes: list) -> tuple[bool, list]:
    """Validate that all config BA codes have coordinate entries.
    
    Args:
        cfg_ba_codes: List of BA code strings from config
        
    Returns:
        Tuple of (is_valid, missing_codes) where:
        - is_valid: True if all codes have coordinates
        - missing_codes: List of BA codes without coordinate entries
    """
    missing = [code for code in cfg_ba_codes if code.upper() not in BA_COORDINATES]
    return (len(missing) == 0, missing)


if __name__ == "__main__":
    # Simple validation when run directly
    print(f"Total BA coordinates defined: {len(BA_COORDINATES)}")
    
    # Test key BAs
    test_codes = ["ERCO", "MISO", "PJM", "CISO", "ISNE", "NYIS", "SWPP", "TVA"]
    for code in test_codes:
        coords = get_ba_coordinates(code)
        print(f"{code}: {coords}")
