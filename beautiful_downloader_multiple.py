# =============================================================================
# OPTIMIZED PARALLEL DOWNLOADER - For High-Performance Machines
# Uses concurrent downloads + smart caching to avoid re-downloads
# Perfect for GPU workstations with lots of RAM!
# =============================================================================

import requests
from pathlib import Path
import time
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize, MinMaxInterval
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION - OPTIMIZED FOR HIGH-PERFORMANCE
# =============================================================================
base_dir = Path("data/beautiful_dataset_v2")
base_dir.mkdir(parents=True, exist_ok=True)

cutout_size_arcmin = 15
pix_scale = 1.5
size_pixels = int(cutout_size_arcmin * 60 / pix_scale)
n_per_class = 1000

# PERFORMANCE SETTINGS
MAX_WORKERS = 16  # Parallel download threads (your machine can handle this!)
BATCH_SIZE = 50  # Process in batches
NO_DELAY = True  # Remove delays for speed (your bandwidth can handle it)

lotss_url = "https://lofar-surveys.org/dr2-cutout.fits"
legacy_jpeg = "https://www.legacysurvey.org/viewer/jpeg-cutout/"

MIN_FLUX_RATIO = 0.01
MIN_OPTICAL_SIZE = 5000

# =============================================================================
# SMART CACHING - Avoid re-downloading same coordinates!
# =============================================================================

class DownloadCache:
    """Track what we've tried to avoid repeated failures"""
    
    def __init__(self, cache_file):
        self.cache_file = Path(cache_file)
        self.failed_coords = set()
        self.load()
    
    def load(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.failed_coords = set(map(tuple, data.get('failed', [])))
            except:
                pass
    
    def save(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump({
                'failed': list(self.failed_coords),
                'updated': datetime.now().isoformat()
            }, f)
    
    def is_failed(self, ra, dec):
        """Check if we've already failed this coordinate"""
        # Round to avoid floating point issues
        coord = (round(ra, 4), round(dec, 4))
        return coord in self.failed_coords
    
    def add_failed(self, ra, dec):
        """Mark coordinate as failed"""
        coord = (round(ra, 4), round(dec, 4))
        self.failed_coords.add(coord)
        
        # Save every 100 failures
        if len(self.failed_coords) % 100 == 0:
            self.save()

# =============================================================================
# COVERAGE CHECK
# =============================================================================

def is_in_good_coverage(ra, dec):
    """Check if coordinates are in BOTH LoTSS AND SDSS coverage"""
    if 120 <= ra <= 240 and 0 <= dec <= 60:
        return True
    if 330 <= ra <= 360 and 0 <= dec <= 15:
        return True
    if 0 <= ra <= 60 and 0 <= dec <= 15:
        return True
    return False

# =============================================================================
# QUERIES (Same as before but returning more candidates)
# =============================================================================

def query_spiral_galaxies(limit=5000):
    """Bright spiral galaxies"""
    print("🔍 Querying spiral galaxies from SDSS...")
    
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
        AND s.subclass = 'STARFORMING'
        AND s.z BETWEEN 0.01 AND 0.12
        AND p.r BETWEEN 13 AND 17
        AND p.petroR50_r > 3
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY p.petroR50_r DESC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Spiral_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} spirals")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_elliptical_galaxies(limit=5000):
    """Elliptical galaxies"""
    print("🔍 Querying elliptical galaxies from SDSS...")
    
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
        AND p.r BETWEEN 13 AND 17
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
            print(f"   ✓ Found {len(df)} ellipticals")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_starburst_galaxies(limit=5000):
    """Starburst galaxies"""
    print("🔍 Querying starburst galaxies from SDSS...")
    
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
        AND s.subclass = 'STARFORMING'
        AND s.z BETWEEN 0.02 AND 0.20
        AND p.r BETWEEN 14 AND 18
        AND (p.u - p.g) < 1.0
        AND p.petroR50_r BETWEEN 1.5 AND 4
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY (p.u - p.g) ASC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'Starburst_' + df['objID'].astype(str)
            print(f"   ✓ Found {len(df)} starbursts")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_radio_loud_agn(limit=5000):
    """Radio-loud AGN"""
    print("🔍 Querying radio-loud AGN from SDSS...")
    
    query = f"""
    SELECT TOP {limit}
        p.objID, p.ra, p.dec, p.i as mag,
        s.z as redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE 
        s.class = 'GALAXY'
        AND (s.subclass LIKE '%AGN%' OR s.subclass = 'BROADLINE')
        AND s.z BETWEEN 0.08 AND 0.40
        AND p.i BETWEEN 15 AND 18.5
        AND ((p.ra BETWEEN 120 AND 240 AND p.dec BETWEEN 0 AND 60)
             OR ((p.ra BETWEEN 0 AND 60 OR p.ra BETWEEN 330 AND 360) 
                 AND p.dec BETWEEN 0 AND 15))
    ORDER BY p.i ASC
    """
    
    try:
        result = SDSS.query_sql(query)
        if result and len(result) > 0:
            df = result.to_pandas()
            df['name'] = 'RadioAGN_' + df['objID'].astype(str)
            df['size'] = 2.5
            print(f"   ✓ Found {len(df)} AGN")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_fr2_radio_galaxies(limit=5000):
    """FR-II radio galaxies"""
    print("🔍 Querying FR-II radio galaxies from LoTSS...")
    
    try:
        v = Vizier(columns=['*'], row_limit=limit*2)
        v.ROW_LIMIT = limit * 2
        
        result = v.query_constraints(
            catalog='J/A+A/659/A1/catalog',
            Maj='>12',
            Total_flux='>40',
            S_Code='M',
            Dec='>25'
        )
        
        if result and len(result[0]) > 0:
            table = result[0]
            df = table.to_pandas()
            df = df.rename(columns={'RAJ2000': 'ra', 'DEJ2000': 'dec', 
                                   'Source_Name': 'name',
                                   'Maj': 'size'})
            
            df = df[df.apply(lambda row: is_in_good_coverage(row['ra'], row['dec']), axis=1)]
            
            df['name'] = 'FR2_' + df.index.astype(str)
            df['redshift'] = 0.5
            df['mag'] = 18.5
            
            print(f"   ✓ Found {len(df)} FR-II galaxies")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

# =============================================================================
# VALIDATION
# =============================================================================

def validate_radio(fits_path):
    """Validate radio FITS"""
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
    """Validate optical JPEG"""
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

# =============================================================================
# PARALLEL DOWNLOAD FUNCTIONS
# =============================================================================

def download_optical(ra, dec, output_path):
    """Download optical - tries SDSS first, then Legacy"""
    
    # Try SDSS first
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
    
    # Try Legacy Survey
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
    """Download radio"""
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
        with fits.open(output_fits, ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            
            if data.ndim == 4:
                data = data[0, 0]
            elif data.ndim == 3:
                data = data[0]
            
            norm = ImageNormalize(
                data, 
                interval=MinMaxInterval(),
                stretch=AsinhStretch(a=0.1)
            )
            
            plt.figure(figsize=(8, 8))
            plt.imshow(norm(data), cmap='inferno', origin='lower')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_png, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return True
    except:
        if output_fits.exists():
            output_fits.unlink()
        return False

def download_single_object(args):
    """Download one object (for parallel execution)"""
    row, optical_dir, radio_dir, radio_png_dir, cache = args
    
    name = str(row['name']).replace(' ', '_').replace('/', '_')
    ra = float(row['ra'])
    dec = float(row['dec'])
    
    # Check cache first
    if cache.is_failed(ra, dec):
        return {'success': False, 'reason': 'cached_failure', 'name': name}
    
    # File paths
    optical_path = optical_dir / f"{name}.jpg"
    radio_fits = radio_dir / f"{name}.fits"
    radio_png = radio_png_dir / f"{name}.png"
    
    # Skip if exists and valid
    if optical_path.exists() and radio_fits.exists() and radio_png.exists():
        if validate_optical(optical_path) and validate_radio(radio_fits):
            return {'success': True, 'reason': 'exists', 'name': name}
    
    # Download optical
    optical_ok = download_optical(ra, dec, optical_path)
    if not optical_ok:
        cache.add_failed(ra, dec)
        return {'success': False, 'reason': 'optical_failed', 'name': name}
    
    # Download radio
    radio_ok = download_radio(ra, dec, radio_fits, radio_png)
    if not radio_ok:
        if optical_path.exists():
            optical_path.unlink()
        cache.add_failed(ra, dec)
        return {'success': False, 'reason': 'radio_failed', 'name': name}
    
    return {'success': True, 'reason': 'downloaded', 'name': name}

# =============================================================================
# PARALLEL DOWNLOAD CATEGORY
# =============================================================================

def download_category_parallel(category, query_func, n_samples):
    """Download category using parallel threads"""
    
    print(f"\n{'='*70}")
    print(f"📸 DOWNLOADING: {category.upper()}")
    print(f"{'='*70}\n")
    
    # Create directories
    optical_dir = base_dir / category / "optical"
    radio_dir = base_dir / category / "radio"
    radio_png_dir = base_dir / category / "radio_png"
    
    for d in [optical_dir, radio_dir, radio_png_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize cache
    cache_file = base_dir / category / f"{category}_cache.json"
    cache = DownloadCache(cache_file)
    
    # Query catalog
    catalog = query_func(limit=n_samples * 10)  # 10x overfetch
    
    if len(catalog) == 0:
        print(f"❌ No objects found for {category}")
        return 0
    
    print(f"📋 Catalog: {len(catalog)} candidates")
    print(f"🎯 Target: {n_samples} pairs")
    print(f"⚡ Using {MAX_WORKERS} parallel threads\n")
    
    # Save catalog
    catalog_file = base_dir / category / f"{category}_catalog.csv"
    catalog.to_csv(catalog_file, index=False)
    
    # Prepare arguments for parallel processing
    args_list = [
        (row, optical_dir, radio_dir, radio_png_dir, cache)
        for _, row in catalog.iterrows()
    ]
    
    # Download in parallel with progress bar
    success_count = 0
    failed_count = 0
    cached_skip = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_object, args): args for args in args_list}
        
        with tqdm(total=len(args_list), desc=category) as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['success']:
                    success_count += 1
                else:
                    if result['reason'] == 'cached_failure':
                        cached_skip += 1
                    else:
                        failed_count += 1
                
                pbar.set_postfix({
                    'success': success_count,
                    'failed': failed_count,
                    'cached': cached_skip
                })
                pbar.update(1)
                
                # Stop if we have enough
                if success_count >= n_samples:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
    
    # Save final cache
    cache.save()
    
    print(f"\n✅ {category}: {success_count}/{n_samples} pairs downloaded")
    print(f"   Success rate: {success_count/len(catalog)*100:.1f}%")
    print(f"   Cached skips: {cached_skip}\n")
    
    return success_count

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run downloader"""
    
    print("\n" + "="*70)
    print("🚀 PARALLEL DOWNLOADER - HIGH PERFORMANCE MODE")
    print("="*70)
    print(f"\n⚡ PERFORMANCE SETTINGS:")
    print(f"   • Parallel threads: {MAX_WORKERS}")
    print(f"   • Smart caching: ENABLED (no re-downloads!)")
    print(f"   • Rate limiting: DISABLED (max speed)")
    print(f"\n📊 Configuration:")
    print(f"   • Samples per class: {n_per_class}")
    print(f"   • Total target: {n_per_class * 5} images (5 categories)")
    print(f"   • Expected time: 2-4 hours (much faster!)")
    print("="*70)
    
    categories = {
        'spiral_galaxies': query_spiral_galaxies,
        'elliptical_galaxies': query_elliptical_galaxies,
        'starburst_galaxies': query_starburst_galaxies,
        'radio_loud_agn': query_radio_loud_agn,
        'fr2_radio_galaxies': query_fr2_radio_galaxies,
    }
    
    total_success = 0
    start_time = time.time()
    
    for category, query_func in categories.items():
        try:
            count = download_category_parallel(category, query_func, n_per_class)
            total_success += count
        except Exception as e:
            print(f"\n❌ Error in {category}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Statistics:")
    print(f"   • Total pairs: {total_success}/{n_per_class * 5}")
    print(f"   • Overall success: {total_success/(n_per_class * 5)*100:.1f}%")
    print(f"   • Time taken: {elapsed/3600:.1f} hours")
    print(f"   • Speed: {total_success/(elapsed/60):.1f} pairs/minute")
    print(f"\n📁 Output: {base_dir}/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()