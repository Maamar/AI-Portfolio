"""
Portfolio Structure Creator

This script creates the complete folder structure for all 19 AI/ML projects.
No dependencies needed - uses only Python built-in libraries.

Run this in your portfolio root folder

Usage:
    python create_portfolio_structure.py
"""

import os
import sys
from pathlib import Path

# Define all 19 projects
PROJECTS = [
    "01-neural-networks-from-scratch",
    "02-classification-pipeline",
    "03-computer-vision-cnn",
    "04-nlp-text-classification",
    "05-rnn-lstm-sequences",
    "06-recommendation-system",
    "07-reinforcement-learning-ql",
    "08-gan-generative-models",
    "09-transfer-learning",
    "10-ensemble-methods",
    "11-autoencoder-anomaly",
    "12-attention-transformers",
    "13-time-series-forecasting",
    "14-bayesian-inference",
    "15-graph-neural-networks",
    "16-meta-learning",
    "17-federated-learning",
    "18-explainable-ai-xai",
    "19-capstone-humanoid-robotics"
]

# Subfolders for each project
SUBFOLDERS = [
    "src",
    "tests",
    "notebooks",
    "data",
    "models",
    "results",
    "assets"
]


def create_structure():
    """Create the complete folder structure."""
    
    print("=" * 70)
    print("🚀 PORTFOLIO STRUCTURE CREATOR")
    print("=" * 70)
    print()
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"📁 Working directory: {current_dir}")
    print()
    
    # Check if we're in the right place
    if not str(current_dir).endswith("AI-Portfolio"):
        print("⚠️  WARNING: You should run this from C:\\Users\\maama\\AI-Portfolio\\")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Cancelled.")
            return False
    
    print()
    print("Creating folder structure for all 19 projects...")
    print("-" * 70)
    print()
    
    created_count = 0
    error_count = 0
    
    # Create each project folder
    for project in PROJECTS:
        try:
            project_path = current_dir / project
            
            # Create project folder
            project_path.mkdir(exist_ok=True)
            
            # Create all subfolders
            for subfolder in SUBFOLDERS:
                subfolder_path = project_path / subfolder
                subfolder_path.mkdir(exist_ok=True)
            
            print(f"✅ {project}")
            created_count += 1
            
        except Exception as e:
            print(f"❌ {project} - Error: {e}")
            error_count += 1
    
    print()
    print("-" * 70)
    print()
    print(f"📊 SUMMARY:")
    print(f"   ✅ Created: {created_count} projects")
    if error_count > 0:
        print(f"   ❌ Errors: {error_count}")
    print()
    
    # Verify structure
    print("📁 VERIFICATION - Checking structure...")
    print()
    
    all_good = True
    for project in PROJECTS[:3]:  # Check first 3 as sample
        project_path = current_dir / project
        if project_path.exists():
            subfolders_exist = all(
                (project_path / subfolder).exists() 
                for subfolder in SUBFOLDERS
            )
            if subfolders_exist:
                print(f"✅ {project} - All subfolders OK")
            else:
                print(f"⚠️  {project} - Missing some subfolders")
                all_good = False
        else:
            print(f"❌ {project} - Not found")
            all_good = False
    
    print(f"   ... (and 16 more projects)")
    print()
    
    if all_good and error_count == 0:
        print("=" * 70)
        print("✨ SUCCESS! Portfolio structure created!")
        print("=" * 70)
        print()
        print("📝 Next steps:")
        print("   1. Check that all folders appear in VS Code Explorer")
        print("   2. Create virtual environment: python -m venv venv")
        print("   3. Activate it: venv\\Scripts\\Activate.ps1")
        print("   4. Install dependencies: pip install -r requirements.txt")
        print("   5. Add Python files to Project 1")
        print()
        return True
    else:
        print("=" * 70)
        print("⚠️  COMPLETED WITH WARNINGS")
        print("=" * 70)
        return False


def show_tree():
    """Show a sample tree structure."""
    print()
    print("📋 FOLDER STRUCTURE CREATED:")
    print()
    print("AI-Portfolio/")
    print("├── venv/                                    (Virtual environment)")
    print("├── 01-neural-networks-from-scratch/")
    print("│   ├── src/                                (Source code)")
    print("│   ├── tests/                              (Unit tests)")
    print("│   ├── notebooks/                          (Jupyter notebooks)")
    print("│   ├── data/                               (Datasets)")
    print("│   ├── models/                             (Trained models)")
    print("│   ├── results/                            (Output results)")
    print("│   └── assets/                             (Images, diagrams)")
    print("├── 02-classification-pipeline/")
    print("│   └── (same structure as above)")
    print("├── 03-computer-vision-cnn/")
    print("│   └── (same structure as above)")
    print("├── ... (04-18)")
    print("└── 19-capstone-humanoid-robotics/")
    print("    └── (same structure as above)")
    print()


def main():
    """Main function."""
    try:
        # Show what will be created
        print()
        print("This script will create:")
        print("  • 19 project folders")
        print("  • 7 subfolders in each project")
        print("  • Total: 133 folders")
        print()
        
        # Ask for confirmation
        response = input("Ready to create? (y/n): ").lower()
        if response != 'y':
            print("Cancelled.")
            return
        
        print()
        
        # Create structure
        success = create_structure()
        
        # Show tree
        show_tree()
        
        if success:
            print("🎉 All done! Check VS Code now!")
            input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()