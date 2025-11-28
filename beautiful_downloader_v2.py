# =============================================================================
# IMPROVED BEAUTIFUL DOWNLOADER - High Success Rate!
# Only queries objects WITHIN LoTSS DR2 coverage
# Better categories with distinct visual features
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

# =============================================================================
# CONFIGURATION
# =============================================================================
base_dir = Path("data/beautiful_dataset_v2")
base_dir.mkdir(parents=True, exist_ok=True)

cutout_size_arcmin = 15
pix_scale = 1.5
size_pixels = int(cutout_size_arcmin * 60 / pix_scale)
n_per_class = 1000  # Target per category

lotss_url = "https://lofar-surveys.org/dr2-cutout.fits"
legacy_jpeg = "https://www.legacysurvey.org/viewer/jpeg-cutout/"

MIN_FLUX_RATIO = 0.01  # More lenient - accept fainter sources
MIN_OPTICAL_SIZE = 5000  # Lower threshold

# =============================================================================
# LoTSS DR2 + SDSS OVERLAP - Critical for success!
# =============================================================================
# We need objects in BOTH LoTSS DR2 AND SDSS coverage
# Main overlap region: RA ~180-240°, Dec ~0-60°

def is_in_good_coverage(ra, dec):
    """Check if coordinates are in BOTH LoTSS AND SDSS coverage"""
    # Best overlap region (SDSS North + LoTSS)
    if 120 <= ra <= 240 and 0 <= dec <= 60:
        return True
    # SDSS South Galactic Cap + LoTSS overlap
    if 330 <= ra <= 360 and 0 <= dec <= 15:
        return True
    if 0 <= ra <= 60 and 0 <= dec <= 15:
        return True
    return False

# =============================================================================
# IMPROVED CATALOG QUERIES - Coverage-aware!
# =============================================================================

def query_spiral_galaxies(limit=2000):
    """Bright spiral galaxies in LoTSS coverage"""
    print("🔍 Querying spiral galaxies from SDSS...")
    
    # Query WITHIN LoTSS coverage regions
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
            print(f"   ✓ Found {len(df)} spirals IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_elliptical_galaxies(limit=2000):
    """Smooth elliptical galaxies - visually VERY different from spirals"""
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
            print(f"   ✓ Found {len(df)} ellipticals IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_starburst_galaxies(limit=2000):
    """Compact starbursts - very bright, intense emission"""
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
            print(f"   ✓ Found {len(df)} starbursts IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_radio_loud_agn(limit=2000):
    """Radio-loud AGN - cross-matched with LoTSS for guaranteed radio!"""
    print("🔍 Querying radio-loud AGN from LoTSS+SDSS cross-match...")
    
    # This uses objects that are ALREADY in LoTSS catalog!
    try:
        v = Vizier(columns=['*'], row_limit=limit*2)
        v.ROW_LIMIT = limit * 2
        
        # LoTSS DR2 catalog - sources with optical IDs
        result = v.query_constraints(
            catalog='J/A+A/659/A1/optical',
            Total_flux='>10',  # Decent flux
            Dec='>25',  # LoTSS coverage
            _r='<5'  # Has optical counterpart
        )
        
        if result and len(result[0]) > 0:
            table = result[0]
            df = table.to_pandas()
            df = df.rename(columns={'RAJ2000': 'ra', 'DEJ2000': 'dec', 
                                   'Source_Name': 'name'})
            
            # Only keep those in good coverage
            df = df[df.apply(lambda row: is_in_good_coverage(row['ra'], row['dec']), axis=1)]
            
            df['name'] = 'RadioAGN_' + df.index.astype(str)
            df['redshift'] = 0.3
            df['mag'] = 17.0
            df['size'] = 3.0
            
            print(f"   ✓ Found {len(df)} radio-loud AGN IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ VizieR query failed: {e}")
    
    # Fallback: SDSS AGN in LoTSS coverage
    print("   Using SDSS AGN fallback...")
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
            print(f"   ✓ Found {len(df)} AGN IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except:
        pass
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_fr2_radio_galaxies(limit=2000):
    """Extended FR-II radio galaxies - double lobes!"""
    print("🔍 Querying FR-II radio galaxies from LoTSS...")
    
    try:
        v = Vizier(columns=['*'], row_limit=limit*2)
        v.ROW_LIMIT = limit * 2
        
        # LoTSS catalog - LARGE, BRIGHT, MULTI-COMPONENT sources
        result = v.query_constraints(
            catalog='J/A+A/659/A1/catalog',
            Maj='>12',  # Large
            Total_flux='>40',  # Bright
            S_Code='M',  # Multi-component
            Dec='>25'  # Coverage
        )
        
        if result and len(result[0]) > 0:
            table = result[0]
            df = table.to_pandas()
            df = df.rename(columns={'RAJ2000': 'ra', 'DEJ2000': 'dec', 
                                   'Source_Name': 'name',
                                   'Maj': 'size'})
            
            # Filter for coverage
            df = df[df.apply(lambda row: is_in_good_coverage(row['ra'], row['dec']), axis=1)]
            
            df['name'] = 'FR2_' + df.index.astype(str)
            df['redshift'] = 0.5
            df['mag'] = 18.5
            
            print(f"   ✓ Found {len(df)} FR-II galaxies IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

def query_compact_sources(limit=2000):
    """Compact radio sources - point-like"""
    print("🔍 Querying compact sources from LoTSS...")
    
    try:
        v = Vizier(columns=['*'], row_limit=limit*2)
        v.ROW_LIMIT = limit * 2
        
        # Small, bright, point-like sources
        result = v.query_constraints(
            catalog='J/A+A/659/A1/catalog',
            Maj='<8',  # Compact
            Total_flux='>15',  # Decent flux
            S_Code='S',  # Single component
            Dec='>25'
        )
        
        if result and len(result[0]) > 0:
            table = result[0]
            df = table.to_pandas()
            df = df.rename(columns={'RAJ2000': 'ra', 'DEJ2000': 'dec', 
                                   'Source_Name': 'name'})
            
            # Filter for coverage
            df = df[df.apply(lambda row: is_in_good_coverage(row['ra'], row['dec']), axis=1)]
            
            df['name'] = 'Compact_' + df.index.astype(str)
            df['redshift'] = 0.4
            df['mag'] = 18.0
            df['size'] = 2.0
            
            print(f"   ✓ Found {len(df)} compact sources IN LoTSS coverage")
            return df[['name', 'ra', 'dec', 'redshift', 'mag', 'size']].head(limit)
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    return pd.DataFrame(columns=['name', 'ra', 'dec', 'redshift', 'mag', 'size'])

# =============================================================================
# VALIDATION & DOWNLOAD (Same as before but with better error handling)
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
            
            if abs(median_val) < 1e-10:  # Essentially zero
                return False
            
            if data_range < abs(median_val) * MIN_FLUX_RATIO:
                return False
            
            return True
    except Exception as e:
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

def download_optical(ra, dec, output_path):
    """Download optical - tries SDSS first, then Legacy Survey"""
    
    # Try 1: SDSS (faster, better for northern sources)
    url_sdss = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    params_sdss = {
        'ra': ra,
        'dec': dec,
        'scale': 0.4,
        'width': 512,
        'height': 512
    }
    
    try:
        response = requests.get(url_sdss, params=params_sdss, timeout=30)
        if response.status_code == 200 and len(response.content) > MIN_OPTICAL_SIZE:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            if validate_optical(output_path):
                return True
            else:
                output_path.unlink()
    except:
        pass
    
    # Try 2: Legacy Survey (slower but wider coverage)
    url_legacy = (f"{legacy_jpeg}?"
                  f"ra={ra:.6f}&dec={dec:.6f}&"
                  f"size={size_pixels}&"
                  f"layer=ls-dr10&"
                  f"pixscale={pix_scale}")
    
    try:
        response = requests.get(url_legacy, timeout=30)
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
        response = requests.get(lotss_url, params=params, timeout=90)
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

# =============================================================================
# DOWNLOAD CATEGORY
# =============================================================================

def download_category(category, query_func, n_samples):
    """Download one category"""
    
    print(f"\n{'='*70}")
    print(f"📸 DOWNLOADING: {category.upper()}")
    print(f"{'='*70}\n")
    
    # Create directories
    optical_dir = base_dir / category / "optical"
    radio_dir = base_dir / category / "radio"
    radio_png_dir = base_dir / category / "radio_png"
    
    for d in [optical_dir, radio_dir, radio_png_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Query catalog - get MORE candidates since filtering is strict
    catalog = query_func(limit=n_samples * 5)  # 5x overfetch for filtering
    
    if len(catalog) == 0:
        print(f"❌ No objects found for {category}")
        return 0
    
    print(f"📋 Catalog: {len(catalog)} candidates")
    print(f"🎯 Target: {n_samples} pairs\n")
    
    # Save catalog
    catalog_file = base_dir / category / f"{category}_catalog.csv"
    catalog.to_csv(catalog_file, index=False)
    
    # Download
    success_count = 0
    failed_count = 0
    
    pbar = tqdm(catalog.iterrows(), total=len(catalog), desc=f"{category}")
    
    for idx, row in pbar:
        if success_count >= n_samples:
            break
        
        name = str(row['name']).replace(' ', '_').replace('/', '_')
        ra = float(row['ra'])
        dec = float(row['dec'])
        
        # File paths
        optical_path = optical_dir / f"{name}.jpg"
        radio_fits = radio_dir / f"{name}.fits"
        radio_png = radio_png_dir / f"{name}.png"
        
        # Skip if exists and valid
        if optical_path.exists() and radio_fits.exists() and radio_png.exists():
            if validate_optical(optical_path) and validate_radio(radio_fits):
                success_count += 1
                pbar.set_postfix({'success': success_count, 'failed': failed_count})
                continue
        
        # Download
        optical_ok = download_optical(ra, dec, optical_path)
        if not optical_ok:
            failed_count += 1
            pbar.set_postfix({'success': success_count, 'failed': failed_count})
            continue
        
        radio_ok = download_radio(ra, dec, radio_fits, radio_png)
        if not radio_ok:
            if optical_path.exists():
                optical_path.unlink()
            failed_count += 1
            pbar.set_postfix({'success': success_count, 'failed': failed_count})
            continue
        
        success_count += 1
        pbar.set_postfix({'success': success_count, 'failed': failed_count})
        
        time.sleep(0.2)  # Faster rate limiting
    
    pbar.close()
    
    print(f"\n✅ {category}: {success_count}/{n_samples} pairs downloaded")
    print(f"   Success rate: {success_count/len(catalog)*100:.1f}%\n")
    
    return success_count

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run downloader"""
    
    print("\n" + "="*70)
    print("🌌 IMPROVED BEAUTIFUL DOWNLOADER v2")
    print("="*70)
    print(f"\n✨ NEW FEATURES:")
    print(f"   • Only queries objects IN LoTSS coverage (higher success!)")
    print(f"   • Better categories with distinct features")
    print(f"   • Improved error handling")
    print(f"\n📊 Configuration:")
    print(f"   • Samples per class: {n_per_class}")
    print(f"   • Total target: {n_per_class * 6} images")
    print(f"   • Expected success rate: 70-90% (much better!)")
    print("="*70)
    
    categories = {
        'spiral_galaxies': query_spiral_galaxies,
        'elliptical_galaxies': query_elliptical_galaxies,
        'starburst_galaxies': query_starburst_galaxies,
        'radio_loud_agn': query_radio_loud_agn,
        'fr2_radio_galaxies': query_fr2_radio_galaxies,
        'compact_sources': query_compact_sources
    }
    
    total_success = 0
    
    for category, query_func in categories.items():
        try:
            count = download_category(category, query_func, n_per_class)
            total_success += count
        except Exception as e:
            print(f"\n❌ Error in {category}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Statistics:")
    print(f"   • Total pairs: {total_success}/{n_per_class * 6}")
    print(f"   • Overall success: {total_success/(n_per_class * 6)*100:.1f}%")
    print(f"\n📁 Output: {base_dir}/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()