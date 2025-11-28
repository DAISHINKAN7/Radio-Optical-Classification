# =============================================================================
# TEST V2 - New categories with better success rate!
# =============================================================================

# Just change the config from the main script
import sys
sys.path.insert(0, '.')

from beautiful_downloader_v2 import *

# Override for testing
n_per_class = 10
base_dir = Path("data/test_v2")
base_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("🧪 TEST V2 - IMPROVED CATEGORIES")
print("="*70)
print(f"\n✨ What's New:")
print(f"   • Only queries LoTSS coverage area (higher success!)")
print(f"   • 6 distinct categories with unique features")
print(f"   • Better error handling")
print(f"\n📊 Test: {n_per_class} samples per category (60 total)")
print(f"   Expected time: ~10-15 minutes")
print(f"   Expected success: 70-90% (much better!)")
print("="*70)

input("\n⏸  Press ENTER to start...")

categories = {
    'spiral_galaxies': query_spiral_galaxies,
    'elliptical_galaxies': query_elliptical_galaxies,
    'starburst_galaxies': query_starburst_galaxies,
    'radio_loud_agn': query_radio_loud_agn,
    'fr2_radio_galaxies': query_fr2_radio_galaxies,
    'compact_sources': query_compact_sources
}

total = 0

for category, func in categories.items():
    try:
        count = download_category(category, func, n_per_class)
        total += count
    except Exception as e:
        print(f"\n❌ Error in {category}: {e}")
        continue

print("\n" + "="*70)
print("🎉 TEST V2 COMPLETE!")
print("="*70)
print(f"\n✅ Downloaded: {total}/{n_per_class * 6} pairs")
print(f"📈 Success rate: {total/(n_per_class * 6)*100:.1f}%")
print(f"\n📁 Check: {base_dir}/")
print("\n💡 Categories:")
print("   1. Spiral galaxies - spiral arms, blue")
print("   2. Elliptical galaxies - smooth, red, round")
print("   3. Starburst galaxies - compact, very blue")
print("   4. Radio-loud AGN - active nucleus + radio")
print("   5. FR-II radio galaxies - double lobes")
print("   6. Compact sources - point-like")
print("\n🚀 If satisfied, run: python beautiful_downloader_v2.py")
print("="*70 + "\n")