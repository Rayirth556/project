"""
simple_project_tree.py
=======================
Simple ASCII project tree structure.
"""

def create_project_tree():
    """Create simple project tree structure"""
    
    tree = """
PROJECT TREE STRUCTURE
================================================================================

project/
├── pdr_pipeline/                    # Core pipeline modules
│   ├── __init__.py
│   ├── layer_2_feature_engine.py    # Feature generation (31 features)
│   ├── layer_3_inference_engine.py  # Model inference & prediction
│   └── layer_4_model_risk_management.py  # Model monitoring
│
├── data/                            # Data storage
│   ├── processed/
│   │   ├── features.parquet         # Training features (1000 x 32)
│   │   ├── labels.parquet           # Training labels (1000 x 1)
│   │   └── metadata.json            # Dataset metadata
│   └── raw/                         # Raw data (if any)
│
├── models/                          # Trained models
│   ├── pdr_xgb_clean.json           # Original 28-feature model
│   ├── pdr_xgb_clean_calibration.json
│   ├── pdr_xgb_full_31_features.json  # New 31-feature model
│   ├── pdr_xgb_full_31_calibration.json
│   ├── feature_importance.json
│   ├── feature_importance_full_31.json
│   ├── training_report.txt
│   └── layer_3_validation_report.json
│
├── CORE SCRIPTS/                    # Main functionality
│   ├── simplified_risk_workflow.py  # End-to-end risk assessment
│   ├── demo_risk_workflow.py         # Demo of workflow
│   └── quick_train_31_model.py      # Train 31-feature model
│
├── DATA PROCESSING/                 # Data handling
│   ├── preprocess_transaction_data.py  # Process transactions_data.csv
│   ├── create_real_training_data.py  # Create proper training data
│   └── check_data_structure.py       # Analyze data structure
│
├── TESTING & VALIDATION/            # Testing framework
│   ├── test_real_world_data.py       # Test on real data
│   ├── quick_kaggle_test.py         # Quick Kaggle data test
│   ├── test_inference_pipeline.py   # Test inference pipeline
│   └── solidify_layer_3.py          # Layer 3 validation
│
├── MODEL TRAINING/                  # Training scripts
│   ├── train_real_world_model.py     # Original training
│   ├── train_clean_model.py          # Clean model training
│   └── cross_validation_clean_model.py  # Cross-validation
│
├── ANALYSIS & DEBUGGING/            # Analysis tools
│   ├── investigate_feature_leakage.py  # Feature leakage analysis
│   ├── debug_row.py                  # Debug individual rows
│   └── training_structure_analysis.py  # Training structure analysis
│
├── DOCUMENTATION/                   # Documentation
│   ├── REAL_WORLD_TESTING_GUIDE.md  # Testing guide
│   └── MODEL_CARD_AND_VALIDATION_REPORT.md  # Model documentation
│
├── UTILITIES/                       # Utility scripts
│   ├── download_datasets.py         # Download datasets
│   ├── generate_large_mock.py       # Generate mock data
│   └── project_tree_structure.py    # This script
│
├── DATA FILES/                      # Large data files
│   ├── transactions_data.csv         # 13M transactions (1.2GB)
│   └── processed.zip                 # Compressed processed data
│
├── CONFIGURATION/                    # Config files
│   ├── .gitignore
│   ├── barclays_project.code-workspace
│   └── error_log.json
│
└── MISC/                           # Miscellaneous
    ├── hello.py                     # Test script
    ├── error_log.txt               # Error logs
    └── __pycache__/                # Python cache

KEY COMPONENTS SUMMARY:
======================

1. INFERENCE PIPELINE:
   - Layer 2: Feature generation (31 features)
   - Layer 3: Model inference (XGBoost)
   - Layer 4: Model monitoring

2. DATA FLOW:
   transactions_data.csv → Feature Engine → Model → Risk Score

3. MODEL FILES:
   - pdr_xgb_full_31_features.json (current model)
   - Trained on 1000 synthetic samples
   - Uses 31 features with weight distribution

4. WORKFLOW:
   simplified_risk_workflow.py (end-to-end assessment)

CURRENT ISSUE:
=============
- Model trained on synthetic aggregated data
- Should be trained on individual user profiles
- Training-inference distribution mismatch
"""
    
    print(tree)

if __name__ == "__main__":
    create_project_tree()
