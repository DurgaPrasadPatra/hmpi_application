#!/usr/bin/env python3
"""
Script to create the HMPI Application project structure
Run: python create_structure.py
"""

import os

def create_hmpi_structure():
    """Create the complete HMPI application folder structure"""
    
    # List of all directories to create
    directories = [
        "hmpi_application",
        "hmpi_application/modules",
        "hmpi_application/data",
        "hmpi_application/data/sample",
        "hmpi_application/data/standards", 
        "hmpi_application/data/uploads",
        "hmpi_application/assets",
        "hmpi_application/assets/images",
        "hmpi_application/assets/styles",
        "hmpi_application/assets/templates",
        "hmpi_application/tests",
        "hmpi_application/docs",
        "hmpi_application/scripts",
        "hmpi_application/deployment",
        "hmpi_application/deployment/docker",
        "hmpi_application/deployment/kubernetes", 
        "hmpi_application/deployment/streamlit"
    ]
    
    # List of all files to create
    files = [
        "hmpi_application/README.md",
        "hmpi_application/requirements.txt",
        "hmpi_application/setup.py",
        "hmpi_application/.gitignore",
        "hmpi_application/config.yaml",
        "hmpi_application/app.py",
        "hmpi_application/modules/__init__.py",
        "hmpi_application/modules/hmpi_calculator.py",
        "hmpi_application/modules/data_processor.py",
        "hmpi_application/modules/visualization.py",
        "hmpi_application/modules/report_generator.py",
        "hmpi_application/modules/geospatial_analyzer.py",
        "hmpi_application/modules/utils.py",
        "hmpi_application/data/sample/sample_groundwater_data.csv",
        "hmpi_application/data/sample/sample_with_coordinates.xlsx",
        "hmpi_application/data/sample/template.csv",
        "hmpi_application/data/standards/who_standards.json",
        "hmpi_application/data/standards/epa_standards.json",
        "hmpi_application/data/standards/local_standards.json",
        "hmpi_application/assets/styles/main.css",
        "hmpi_application/assets/templates/report_template.html",
        "hmpi_application/assets/templates/summary_template.md",
        "hmpi_application/tests/__init__.py",
        "hmpi_application/tests/test_hmpi_calculator.py",
        "hmpi_application/tests/test_data_processor.py",
        "hmpi_application/tests/test_visualization.py",
        "hmpi_application/tests/test_integration.py",
        "hmpi_application/docs/user_guide.md",
        "hmpi_application/docs/api_reference.md",
        "hmpi_application/docs/methodology.md",
        "hmpi_application/docs/deployment_guide.md",
        "hmpi_application/scripts/data_validation.py",
        "hmpi_application/scripts/batch_processing.py",
        "hmpi_application/scripts/database_setup.py",
        "hmpi_application/deployment/docker/Dockerfile",
        "hmpi_application/deployment/docker/docker-compose.yml",
        "hmpi_application/deployment/kubernetes/deployment.yaml",
        "hmpi_application/deployment/streamlit/config.toml"
    ]
    
    # Create directories
    print("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create files
    print("\nCreating files...")
    for file_path in files:
        with open(file_path, 'w') as f:
            f.write(f"# {os.path.basename(file_path)}\n")
        print(f"✓ Created: {file_path}")
    
    # Create empty .gitkeep files for empty directories
    empty_dirs = [
        "hmpi_application/data/uploads",
        "hmpi_application/assets/images"
    ]
    
    for directory in empty_dirs:
        gitkeep_path = os.path.join(directory, ".gitkeep")
        with open(gitkeep_path, 'w') as f:
            f.write("")
        print(f"✓ Created: {gitkeep_path}")
    
    print(f"\n🎉 HMPI application structure created successfully!")
    print(f"📁 Root directory: hmpi_application/")
    print(f"📄 Total directories: {len(directories)}")
    print(f"📄 Total files: {len(files) + len(empty_dirs)}")

if __name__ == "__main__":
    create_hmpi_structure()
