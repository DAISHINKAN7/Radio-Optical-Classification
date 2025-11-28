# =============================================================================
# ELLIPTICAL GALAXIES BOOSTER - Add more ellipticals to existing dataset
# Relaxed criteria to get more samples!
# =============================================================================

import requests
from pathlib import Path
import time
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
from astroquery.sdss import SDSS
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
base_dir = Path("data/beautiful_dataset_v2")
category = "elliptical_galaxies"

# Target number (will add to existing 72)
TARGET_TOTAL = 1000
EXISTING_COUNT = 72
TARGET_NEW = TARGET_TOTAL - EXISTING_COUNT  # Need 928 more!

MAX_WORKERS = 16
lotss_url = "https://lofar-surveys.org/dr2-cutout.fits"
legacy_jpeg = "https://www.legacysurvey.org/viewer/jpeg-cutout/"
cutout_size_arcmin = 15
pix_scale = 1.5
size_pixels = int(cutout_size_arcmin * 60 / pix_scale)

MIN_FLUX_RATIO = 0.01
MIN_OPTICAL_SIZE = 5000

print("="*70)
print("🔴 ELLIPTICAL GALAXIES BOOSTER")
print("="*70)
print(f"\n📊 Current status:")
print(f"   • Existing: {EXISTING_COUNT}")
print(f"   • Target: {TARGET_TOTAL}")
print(f"   • Need: {TARGET_NEW} more!")
print("\n🎯 Strategy:")
print("   • Query 1: Standard ellipticals (strict)")
print("   • Query 2: Red galaxies (relaxed colors)")
print("   • Query 3: Early-type galaxies (very relaxed)")
print("   • Query 4: Non-star-forming galaxies")
print("="*70 + "\n")

# =============================================================================
# CACHE SYSTEM
# =============================================================================

class DownloadCache:
    def __init__(self, cache_file):
        self.cache_file = Path(cache_file)
        self.failed_coords = set()
        self.load()
    
    def load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.failed_coords = set(map(tuple, data.get('failed', [])))
            except:
                pass
    
    def save(self):
        with open(self.cache_file, 'w') as f:
            json.dump({'failed': list(self.failed_coords)}, f)
    
    def is_failed(self, ra, dec):
        coord = (round(ra, 4), round(dec, 4))
        return coord in self.failed_coords
    
    def add_failed(self, ra, dec):
        coord = (round(ra, 4), round(dec, 4))
        self.failed_coords.add(coord)

def is_in_good_coverage(ra, dec):
    if 120 <= ra <= 240 and 0 <= dec <= 60:
        return True
    if 330 <= ra <= 360 and 0 <= dec <= 15:
        return True
    if 0 <= ra <= 60 and 0 <= dec <= 15:
        return True
    return False

# =============================================================================
# MULTIPLE QUERY STRATEGIES FOR ELLIPTICALS
# =============================================================================

def query_ellipticals_strict(limit=3000):
    """Original strict criteria"""
    print("🔍 Query 1: Strict ellipticals...")
    
    query = f"""
    SELECT TOP {limit}
        p.objID, p.ra, p.dec, p.r as mag,
        p.petroR50_r as size,
        s.z as redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        p.type = 3
        AND s.class = 'GALAXY'
        AND (s.subclass = '' OR s.subclass NOT LIKE '%STAR%')
        AND s.z BETWEEN 0.02 AND 0.15
        AND p.r BETWEEN 13 AND 17.5
        AND (p.u - p.r) > 2.2
        AND (p.g - p.r) > 1.2
        AND p.petroR50_r > 2
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY (p.u - p.r) DESC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Elliptical_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} candidates")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']]
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_red_galaxies(limit=3000):
    """Red galaxies (relaxed color cuts)"""
    print("🔍 Query 2: Red galaxies (relaxed)...")
    
    query = f"""
    SELECT TOP {limit}
        p.objID, p.ra, p.dec, p.r as mag,
        p.petroR50_r as size,
        s.z as redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        p.type = 3
        AND s.class = 'GALAXY'
        AND s.z BETWEEN 0.02 AND 0.20
        AND p.r BETWEEN 13 AND 18
        AND (p.u - p.r) > 1.8
        AND (p.g - p.r) > 0.8
        AND p.petroR50_r > 1.5
        AND (s.subclass = '' OR s.subclass NOT LIKE '%STAR%')
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY (p.u - p.r) DESC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Elliptical_Red_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} candidates")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']]
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_early_type(limit=3000):
    """Early-type galaxies (very relaxed)"""
    print("🔍 Query 3: Early-type galaxies (very relaxed)...")
    
    query = f"""
    SELECT TOP {limit}
        p.objID, p.ra, p.dec, p.r as mag,
        p.petroR50_r as size,
        s.z as redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        p.type = 3
        AND s.class = 'GALAXY'
        AND s.z BETWEEN 0.01 AND 0.25
        AND p.r BETWEEN 12 AND 18.5
        AND (p.u - p.r) > 1.5
        AND p.petroR50_r > 1.0
        AND p.deVRad_r > 0
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY p.deVRad_r DESC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Elliptical_Early_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} candidates")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']]
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_non_starforming(limit=3000):
    """Non-star-forming red galaxies"""
    print("🔍 Query 4: Non-star-forming galaxies...")
    
    query = f"""
    SELECT TOP {limit}
        p.objID, p.ra, p.dec, p.r as mag,
        p.petroR50_r as size,
        s.z as redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        p.type = 3
        AND s.class = 'GALAXY'
        AND s.z BETWEEN 0.02 AND 0.20
        AND p.r BETWEEN 13 AND 18
        AND (p.g - p.r) > 0.7
        AND (p.r - p.i) > 0.3
        AND p.petroR50_r > 1.0
        AND s.subclass NOT LIKE '%STAR%'
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY (p.g - p.r) DESC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Elliptical_NonSF_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} candidates")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']]
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

# =============================================================================
# VALIDATION & DOWNLOAD
# =============================================================================

def validate_radio(fits_path):
    try:
        with fits.open(fits_path, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            if data is None:
                return False
            
            if data.ndim == 4:
                data = data[0, 0]
            elif data.ndim == 3:
                data = data[0]
            
            if np.all(np.isnan(data)) or np.all(data == 0):
                return False
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return False
            
            data_range = np.ptp(valid_data)
            median_val = np.median(valid_data)
            
            if abs(median_val) < 1e-10:
                return False
            
            if data_range < abs(median_val) * MIN_FLUX_RATIO:
                return False
            
            return True
    except:
        return False

def validate_optical(jpg_path):
    try:
        from PIL import Image
        
        if not jpg_path.exists() or jpg_path.stat().st_size < MIN_OPTICAL_SIZE:
            return False
        
        img = Image.open(jpg_path)
        img_array = np.array(img)
        
        if img_array.ndim == 3:
            mean_brightness = np.mean(img_array)
            if mean_brightness < 10:
                return False
            
            std_brightness = np.std(img_array)
            if std_brightness < 5:
                return False
        
        return True
    except:
        return False

def download_optical(ra, dec, output_path):
    # Try SDSS
    url_sdss = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    params_sdss = {
        'ra': ra,
        'dec': dec,
        'scale': 0.4,
        'width': 512,
        'height': 512
    }
    
    try:
        response = requests.get(url_sdss, params=params_sdss, timeout=20)
        if response.status_code == 200 and len(response.content) > MIN_OPTICAL_SIZE:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            if validate_optical(output_path):
                return True
            else:
                output_path.unlink()
    except:
        pass
    
    # Try Legacy
    url_legacy = (f"{legacy_jpeg}?"
                  f"ra={ra:.6f}&dec={dec:.6f}&"
                  f"size={size_pixels}&"
                  f"layer=ls-dr10&"
                  f"pixscale={pix_scale}")
    
    try:
        response = requests.get(url_legacy, timeout=20)
        if response.status_code == 200 and len(response.content) > MIN_OPTICAL_SIZE:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            if validate_optical(output_path):
                return True
            else:
                output_path.unlink()
    except:
        pass
    
    return False

def download_radio(ra, dec, output_fits, output_png):
    params = {
        'pos': f"{ra:.6f} {dec:+.6f}",
        'size': cutout_size_arcmin
    }
    
    try:
        response = requests.get(lotss_url, params=params, timeout=60)
        response.raise_for_status()
        
        with open(output_fits, 'wb') as f:
            f.write(response.content)
        
        if not validate_radio(output_fits):
            output_fits.unlink()
            return False
        
        # Create PNG
        try:
            from PIL import Image
            
            with fits.open(output_fits, ignore_missing_simple=True) as hdul:
                data = hdul[0].data
                
                if data.ndim == 4:
                    data = data[0, 0]
                elif data.ndim == 3:
                    data = data[0]
                
                data_clean = data[~np.isnan(data)]
                if len(data_clean) > 0:
                    vmin, vmax = np.percentile(data_clean, [0.5, 99.5])
                    
                    data_norm = np.arcsinh((data - vmin) / (vmax - vmin + 1e-10))
                    data_norm = (data_norm - np.nanmin(data_norm)) / (np.nanmax(data_norm) - np.nanmin(data_norm) + 1e-10)
                    data_norm = (data_norm * 255).astype(np.uint8)
                    data_norm = np.nan_to_num(data_norm, nan=0)
                    
                    img_array = np.zeros((*data_norm.shape, 3), dtype=np.uint8)
                    img_array[:, :, 0] = data_norm
                    img_array[:, :, 1] = np.clip(data_norm - 64, 0, 255)
                    img_array[:, :, 2] = np.clip(data_norm - 128, 0, 255)
                    
                    img = Image.fromarray(img_array)
                    img.save(output_png)
        except:
            pass
        
        return True
    except:
        if output_fits.exists():
            output_fits.unlink()
        return False

def download_single_object(args):
    row, optical_dir, radio_dir, radio_png_dir, cache = args
    
    name = str(row['name']).replace(' ', '_').replace('/', '_')
    ra = float(row['ra'])
    dec = float(row['dec'])
    
    if cache.is_failed(ra, dec):
        return {'success': False, 'reason': 'cached'}
    
    optical_path = optical_dir / f"{name}.jpg"
    radio_fits = radio_dir / f"{name}.fits"
    radio_png = radio_png_dir / f"{name}.png"
    
    # Skip if exists
    if optical_path.exists() and radio_fits.exists() and radio_png.exists():
        if validate_optical(optical_path) and validate_radio(radio_fits):
            return {'success': True, 'reason': 'exists'}
    
    # Download
    optical_ok = download_optical(ra, dec, optical_path)
    if not optical_ok:
        cache.add_failed(ra, dec)
        return {'success': False, 'reason': 'optical_failed'}
    
    radio_ok = download_radio(ra, dec, radio_fits, radio_png)
    if not radio_ok:
        if optical_path.exists():
            optical_path.unlink()
        cache.add_failed(ra, dec)
        return {'success': False, 'reason': 'radio_failed'}
    
    return {'success': True, 'reason': 'downloaded'}

# =============================================================================
# MAIN DOWNLOAD
# =============================================================================

def main():
    # Setup directories
    optical_dir = base_dir / category / "optical"
    radio_dir = base_dir / category / "radio"
    radio_png_dir = base_dir / category / "radio_png"
    
    for d in [optical_dir, radio_dir, radio_png_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Cache
    cache_file = base_dir / category / f"{category}_cache.json"
    cache = DownloadCache(cache_file)
    
    # Run all queries
    print("\n📋 Running multiple queries to get more candidates...\n")
    
    catalogs = []
    catalogs.append(query_ellipticals_strict())
    catalogs.append(query_red_galaxies())
    catalogs.append(query_early_type())
    catalogs.append(query_non_starforming())
    
    # Combine and remove duplicates
    all_catalog = pd.concat(catalogs, ignore_index=True)
    all_catalog = all_catalog.drop_duplicates(subset=['ra', 'dec'], keep='first')
    
    print(f"\n📊 Combined catalog: {len(all_catalog)} unique candidates")
    print(f"🎯 Will download until we have {TARGET_TOTAL} total\n")
    
    # Count existing
    existing_files = len(list(optical_dir.glob('*.jpg')))
    print(f"✓ Found {existing_files} existing files")
    print(f"✓ Need {TARGET_TOTAL - existing_files} more\n")
    
    # Prepare for parallel download
    args_list = [
        (row, optical_dir, radio_dir, radio_png_dir, cache)
        for _, row in all_catalog.iterrows()
    ]
    
    success_count = existing_files
    failed_count = 0
    cached_skip = 0
    
    print("⚡ Starting parallel download with 16 threads...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_object, args): args for args in args_list}
        
        with tqdm(total=len(args_list), desc="ellipticals_boost") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['success']:
                    if result['reason'] == 'downloaded':
                        success_count += 1
                else:
                    if result['reason'] == 'cached':
                        cached_skip += 1
                    else:
                        failed_count += 1
                
                pbar.set_postfix({
                    'total': success_count,
                    'new': success_count - existing_files,
                    'failed': failed_count
                })
                pbar.update(1)
                
                if success_count >= TARGET_TOTAL:
                    for f in futures:
                        f.cancel()
                    break
    
    cache.save()
    
    print("\n" + "="*70)
    print("🎉 BOOST COMPLETE!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   • Started with: {existing_files}")
    print(f"   • Downloaded: {success_count - existing_files} new")
    print(f"   • Total now: {success_count}")
    print(f"   • Target: {TARGET_TOTAL}")
    print(f"   • Success rate: {(success_count - existing_files)/len(all_catalog)*100:.1f}%")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()