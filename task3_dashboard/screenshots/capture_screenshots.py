import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import subprocess
import sys

# ── Config ──────────────────────────────────────────────
DASHBOARD_URL = "http://127.0.0.1:8050"
SCREENSHOTS_DIR = os.path.dirname(os.path.abspath(__file__))
WAIT = 4  # seconds to wait for charts to load
# ────────────────────────────────────────────────────────

def setup_driver():
    options = Options()
    options.add_argument("--window-size=1600,900")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # Remove headless so you can see it working
    # options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    return driver

def take_screenshot(driver, filename, scroll_y=0):
    if scroll_y > 0:
        driver.execute_script(f"window.scrollTo(0, {scroll_y});")
        time.sleep(1.5)
    path = os.path.join(SCREENSHOTS_DIR, filename)
    driver.save_screenshot(path)
    print(f"  ✅ Saved: {filename}")

def select_dropdown(driver, dropdown_id, value):
    try:
        # Try Dash dropdown
        dropdown = driver.find_element(By.ID, dropdown_id)
        dropdown.click()
        time.sleep(0.5)
        options = driver.find_elements(
            By.XPATH, f"//div[@id='{dropdown_id}']//div[contains(@class,'option')]"
        )
        for opt in options:
            if opt.text.strip() == value:
                opt.click()
                time.sleep(WAIT)
                return True
    except Exception as e:
        print(f"  ⚠️ Dropdown error: {e}")
    return False

def reset_filters(driver):
    """Reset all dropdowns to 'All'"""
    for dd_id in ["category-filter", "region-filter", "year-filter"]:
        select_dropdown(driver, dd_id, "All")
    time.sleep(WAIT)

def main():
    print("\n🚀 Starting Dashboard Screenshot Capture...")
    print("=" * 50)

    # Install selenium if not present
    try:
        import selenium
    except ImportError:
        print("Installing selenium...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium"])

    driver = setup_driver()

    try:
        # ── Open Dashboard ──────────────────────────────
        print("\n📂 Opening dashboard...")
        driver.get(DASHBOARD_URL)
        time.sleep(WAIT + 1)

        # ── Screenshot 1: Full Overview ─────────────────
        print("\n📸 Taking screenshots...")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        take_screenshot(driver, "01_dashboard_overview.png")

        # ── Screenshot 2: KPI Cards ─────────────────────
        driver.execute_script("window.scrollTo(0, 250);")
        time.sleep(1)
        take_screenshot(driver, "02_kpi_cards.png")

        # ── Screenshot 3: Charts Row 1 ──────────────────
        driver.execute_script("window.scrollTo(0, 450);")
        time.sleep(1)
        take_screenshot(driver, "03_sales_trend_and_category_pie.png")

        # ── Screenshot 4: Charts Row 2 ──────────────────
        driver.execute_script("window.scrollTo(0, 900);")
        time.sleep(1.5)
        take_screenshot(driver, "04_region_bar_and_scatter.png")

        # ── Screenshot 5: Charts Row 3 ──────────────────
        driver.execute_script("window.scrollTo(0, 1350);")
        time.sleep(1.5)
        take_screenshot(driver, "05_top_products_and_segment.png")

        # ── Screenshot 6: Filter - Technology ──────────
        print("\n🔽 Applying filters...")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        select_dropdown(driver, "category-filter", "Technology")
        take_screenshot(driver, "06_filter_technology.png")

        # ── Screenshot 7: Filter - Furniture ───────────
        select_dropdown(driver, "category-filter", "Furniture")
        take_screenshot(driver, "07_filter_furniture.png")

        # ── Screenshot 8: Filter - Office Supplies ──────
        select_dropdown(driver, "category-filter", "Office Supplies")
        take_screenshot(driver, "08_filter_office_supplies.png")

        # ── Screenshot 9: Filter - East Region ──────────
        reset_filters(driver)
        select_dropdown(driver, "region-filter", "East")
        take_screenshot(driver, "09_filter_region_east.png")

        # ── Screenshot 10: Filter - West Region ─────────
        select_dropdown(driver, "region-filter", "West")
        take_screenshot(driver, "10_filter_region_west.png")

        # ── Screenshot 11: Year Filter ───────────────────
        reset_filters(driver)
        select_dropdown(driver, "year-filter", "2017")
        take_screenshot(driver, "11_filter_year_2017.png")

        # ── Screenshot 12: Full page scrolled view ───────
        reset_filters(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        take_screenshot(driver, "12_full_page_bottom.png")

        # ── Done ─────────────────────────────────────────
        print("\n" + "=" * 50)
        print("✅ All screenshots captured successfully!")
        print(f"📁 Saved in: {SCREENSHOTS_DIR}")
        print("=" * 50)

        # List all saved files
        files = [f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith(".png")]
        files.sort()
        print(f"\n📸 Total screenshots: {len(files)}")
        for f in files:
            size = os.path.getsize(os.path.join(SCREENSHOTS_DIR, f))
            print(f"   {f} ({size/1024:.1f} KB)")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        time.sleep(2)
        driver.quit()
        print("\n🏁 Browser closed.")

if __name__ == "__main__":
    main()