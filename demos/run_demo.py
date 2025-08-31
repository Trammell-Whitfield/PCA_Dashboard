#!/usr/bin/env python3
"""
Interactive Demo Runner for PCA Stock Analysis Dashboard

Run this script to generate sample data, create visualizations,
and demonstrate the dashboard capabilities.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\n{step_num}. {description}")
    print("-" * 40)

def run_sample_data_generation():
    """Generate sample data and visualizations"""
    print_step(1, "Generating Sample Data & Visualizations")
    
    try:
        # Run sample data generator
        result = subprocess.run([
            sys.executable, 
            "sample_data.py"
        ], cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample data generated successfully!")
            print(result.stdout)
        else:
            print("❌ Error generating sample data:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")

def run_usage_demo():
    """Run the usage demonstration"""
    print_step(2, "Running Usage Demo")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "sample_usage.py"
        ], cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Error running usage demo:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Failed to run usage demo: {e}")

def check_dashboard_files():
    """Check if main dashboard files exist"""
    print_step(3, "Checking Dashboard Files")
    
    required_files = [
        "../main.py",
        "../GARCH_update.py", 
        "../LSTM_Pred.py",
        "../time_series_analysis.py"
    ]
    
    demo_dir = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = demo_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} required files are missing")
    else:
        print(f"\n✅ All dashboard files are present!")

def show_dashboard_startup():
    """Show how to start the dashboard"""
    print_step(4, "Starting the Dashboard")
    
    print("To run the full dashboard:")
    print("1. Navigate to the project root directory:")
    print("   cd /Users/elhoyabembe/Documents/GitHub/Investalogical_Final/PCA_Dashboard")
    print()
    print("2. Install required dependencies:")
    print("   pip install -r requirement.txt")
    print()
    print("3. Start the dashboard:")
    print("   python main.py")
    print()
    print("4. Open your web browser to:")
    print("   http://localhost:8050")
    print()
    print("💡 The dashboard will automatically load with sample data if needed")

def show_demo_files():
    """Show created demo files"""
    print_step(5, "Demo Files Created")
    
    demo_dir = Path(__file__).parent
    demo_files = [
        "README.md - Comprehensive demo documentation",
        "sample_usage.py - Programmatic usage examples", 
        "sample_data.py - Sample data generator",
        "run_demo.py - This interactive demo runner",
        "sample_stock_data.csv - Generated sample data (if created)",
        "sample_analysis_charts.png - Sample visualization charts (if created)"
    ]
    
    print("📁 Files in demos/ folder:")
    for file_desc in demo_files:
        file_name = file_desc.split(" - ")[0]
        file_path = demo_dir / file_name
        
        if file_path.exists():
            print(f"   ✅ {file_desc}")
        else:
            print(f"   📝 {file_desc}")

def main():
    """Run complete interactive demo"""
    print_header("PCA Stock Analysis Dashboard - Interactive Demo")
    
    print("🚀 Welcome to Investalogical's Advanced Analytics Platform!")
    print("This demo will showcase the dashboard capabilities and generate sample content.")
    
    # Menu options
    while True:
        print("\n" + "=" * 60)
        print("📋 Demo Options:")
        print("1. Generate sample data & visualizations")
        print("2. Run usage demonstration") 
        print("3. Check dashboard files")
        print("4. Show dashboard startup instructions")
        print("5. Show demo files")
        print("6. Run all demos")
        print("7. Exit")
        print("=" * 60)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            run_sample_data_generation()
        elif choice == "2":
            run_usage_demo()
        elif choice == "3":
            check_dashboard_files()
        elif choice == "4":
            show_dashboard_startup()
        elif choice == "5":
            show_demo_files()
        elif choice == "6":
            run_sample_data_generation()
            time.sleep(1)
            run_usage_demo() 
            time.sleep(1)
            check_dashboard_files()
            time.sleep(1)
            show_dashboard_startup()
            time.sleep(1)
            show_demo_files()
        elif choice == "7":
            print("\n👋 Thanks for exploring the PCA Dashboard demo!")
            print("🚀 Ready to analyze some stocks? Run: python main.py")
            break
        else:
            print("❌ Invalid option. Please choose 1-7.")

if __name__ == "__main__":
    main()