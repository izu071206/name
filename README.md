"""
Improved Training Script (FIXED)
Train tất cả models: Base models, Ensemble, Family Classifier
File: src/models/train_improved.py
"""

import sys
import os
import yaml
import joblib
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
try:
    from src.models.neural_network_model import NeuralNetworkModel
    NN_AVAILABLE = True
except (ImportError, OSError):
    NN_AVAILABLE = False
    
from src.models.ensemble_model import EnsembleModel
from src.models.family_classifier import MalwareFamilyClassifier, CombinedObfuscationFamilyModel
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load training configuration"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    return full_config.get('training', full_config)


def load_data():
    """Load processed data"""
    logger.info("Loading processed data...")
    
    train_path = os.path.join(project_root, "data/processed/train_features.pkl")
    val_path = os.path.join(project_root, "data/processed/val_features.pkl")
    test_path = os.path.join(project_root, "data/processed/test_features.pkl")
    metadata_path = os.path.join(project_root, "data/processed/sample_metadata.csv")
    
    # Load features
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    with open(val_path, 'rb') as f:
        X_val, y_val = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Load metadata for family info
    metadata = None
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


def train_base_models(X_train, y_train, X_val, y_val, config):
    """Train base models: Random Forest, XGBoost, Neural Network"""
    logger.info("="*60)
    logger.info("TRAINING BASE MODELS")
    logger.info("="*60)
    
    models = {}
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    # --- FIX START: VALIDATE CLASS DISTRIBUTION ---
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    
    logger.info("\nTraining set class distribution:")
    logger.info(f"  Class 0 (Benign):     {neg_count} samples")
    logger.info(f"  Class 1 (Obfuscated): {pos_count} samples")

    # Critical check: Must have both classes
    if pos_count == 0 or neg_count == 0:
        logger.error("\n" + "="*60)
        logger.error("CRITICAL ERROR: Training set is missing one or both classes!")
        logger.error("============================================================")
        if neg_count == 0:
             logger.error("❌ Training set: Missing class 0 (Benign)!")
        if pos_count == 0:
             logger.error("❌ Training set: Missing class 1 (Obfuscated)!")

        logger.error("\nPossible causes:")
        logger.error("1. Not enough samples in data/benign/ or data/obfuscated/")
        logger.error("2. Data split ratio is too extreme (check config/dataset_config.yaml)")
        logger.error("3. Feature extraction failed for one class (check previous logs)")
        logger.error("="*60)
        raise ValueError("Cannot train with incomplete training set. Need both Benign and Obfuscated samples.")

    # Calculate imbalance ratio for weighting
    imbalance_ratio = float(neg_count / pos_count) 
    logger.info(f"\nData Imbalance Ratio (Neg/Pos): {imbalance_ratio:.2f}")
    # --- FIX END ---
    
    # 1. Random Forest
    logger.info("\n[1/3] Training Random Forest...")
    rf_params = config.get('random_forest', {})
    rf_model = RandomForestModel(**rf_params)
    rf_model.train(X_train, y_train, X_val, y_val)
    rf_model.save(os.path.join(save_dir, "random_forest_model.pkl"))
    models['random_forest'] = rf_model
    logger.info("✓ Random Forest training completed")
    
    # 2. XGBoost
    logger.info("\n[2/3] Training XGBoost...")
    xgb_params = config.get('xgboost', {}).copy()
    if xgb_params.get('scale_pos_weight') == 'auto':
        xgb_params['scale_pos_weight'] = imbalance_ratio
    xgb_model = XGBoostModel(**xgb_params)
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_model.save(os.path.join(save_dir, "xgboost_model.json"))
    models['xgboost'] = xgb_model
    logger.info("✓ XGBoost training completed")
    
    # 3. Neural Network (if available)
    if NN_AVAILABLE:
        logger.info("\n[3/3] Training Neural Network...")
        nn_params = config.get('neural_network', {})
        nn_model = NeuralNetworkModel(**nn_params)
        nn_model.train(X_train, y_train, X_val, y_val)
        nn_model.save(os.path.join(save_dir, "neural_network_model.pt"))
        models['neural_network'] = nn_model
        logger.info("✓ Neural Network training completed")
    else:
        logger.warning("Neural Network not available (PyTorch issue)")
    
    return models


def train_ensemble_model(X_train, y_train, X_val, y_val, base_models):
    """Train Ensemble Model"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ENSEMBLE MODEL")
    logger.info("="*60)
    
    save_dir = os.path.join(project_root, "models")
    
    # Determine which base models are available
    available_models = ['rf', 'xgb']
    if 'neural_network' in base_models:
        available_models.append('nn')
    
    # Create ensemble with weighted voting
    logger.info(f"Creating ensemble with models: {available_models}")
    ensemble = EnsembleModel(
        strategy='weighted_voting',
        base_models=available_models,
        weights=None  # Use default weights
    )
    
    # Train ensemble
    logger.info("Training ensemble...")
    ensemble.train(X_train, y_train, X_val, y_val)
    
    # Save ensemble
    ensemble.save(os.path.join(save_dir, "ensemble_model.pkl"))
    logger.info("✓ Ensemble model saved")
    
    return ensemble


def train_family_classifier(X_train, y_train, X_val, y_val, metadata):
    """Train Malware Family Classifier"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING FAMILY CLASSIFIER")
    logger.info("="*60)
    
    if metadata is None:
        logger.warning("No metadata available for family classification")
        return None
    
    # Get family labels from metadata
    train_metadata = metadata[metadata['split'] == 'train']
    val_metadata = metadata[metadata['split'] == 'val']
    
    if 'family' not in train_metadata.columns:
        logger.warning("No family column in metadata")
        return None
    
    # Get families for train/val samples
    families_train = train_metadata['family'].values
    families_val = val_metadata['family'].values if len(val_metadata) > 0 else None
    
    # Check if we have enough families
    unique_families = np.unique(families_train)
    if len(unique_families) < 2:
        logger.warning(f"Only {len(unique_families)} family found. Need at least 2 for classification.")
        return None
    
    logger.info(f"Training family classifier with {len(unique_families)} families")
    
    # Create and train classifier
    family_classifier = MalwareFamilyClassifier(n_estimators=200, max_depth=30)
    family_classifier.train(
        X_train, families_train,
        X_val if families_val is not None else None,
        families_val
    )
    
    # Save classifier
    save_dir = os.path.join(project_root, "models")
    family_classifier.save(os.path.join(save_dir, "family_classifier.pkl"))
    logger.info("✓ Family classifier saved")
    
    return family_classifier


def evaluate_all_models(models, X_test, y_test, metadata=None):
    """Evaluate all models"""
    logger.info("\n" + "="*60)
    logger.info("EVALUATING MODELS")
    logger.info("="*60)
    
    eval_dir = os.path.join(project_root, "data/evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(output_dir=eval_dir)
    
    results = {}
    
    # Evaluate base models
    for name, model in models.items():
        if name == 'family_classifier':
            continue  # Skip family classifier for obfuscation metrics
        
        logger.info(f"\nEvaluating {name}...")
        
        if name == 'ensemble':
            # For ensemble, use the ensemble object directly
            metrics = evaluator.evaluate(model, X_test, y_test, model_name=name)
        else:
            # For base models, use the underlying sklearn/xgboost model
            metrics = evaluator.evaluate(model.model, X_test, y_test, model_name=name)
        
        results[name] = metrics
        
        logger.info(f"  Accuracy: {metrics['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['metrics']['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['metrics']['precision']:.4f}")
        logger.info(f"  Recall: {metrics['metrics']['recall']:.4f}")
    
    # Evaluate family classifier separately
    if 'family_classifier' in models and metadata is not None:
        logger.info("\nEvaluating Family Classifier...")
        test_metadata = metadata[metadata['split'] == 'test']
        
        if 'family' in test_metadata.columns and len(test_metadata) > 0:
            families_test = test_metadata['family'].values
            family_preds = models['family_classifier'].predict(X_test)
            family_acc = np.mean(family_preds == families_test)
            
            logger.info(f"  Family Classification Accuracy: {family_acc:.4f}")
            results['family_classifier'] = {'accuracy': family_acc}
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    best_model = None
    best_f1 = 0.0
    
    for name, metrics in results.items():
        if name == 'family_classifier':
            continue
        f1 = metrics['metrics']['f1_score']
        logger.info(f"{name:20s} | F1: {f1:.4f} | Acc: {metrics['metrics']['accuracy']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
    
    logger.info(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    return results


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("IMPROVED TRAINING PIPELINE (FIXED)")
    logger.info("="*60)
    
    # 1. Load configuration
    config = load_config("config/train_config.yaml")
    
    # 2. Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_data()
    
    # 3. Train base models
    base_models = train_base_models(X_train, y_train, X_val, y_val, config)
    
    # 4. Train ensemble model
    ensemble = train_ensemble_model(X_train, y_train, X_val, y_val, base_models)
    base_models['ensemble'] = ensemble
    
    # 5. Train family classifier
    family_classifier = train_family_classifier(X_train, y_train, X_val, y_val, metadata)
    if family_classifier:
        base_models['family_classifier'] = family_classifier
    
    # 6. Evaluate all models
    results = evaluate_all_models(base_models, X_test, y_test, metadata)
    
    # 7. Save combined model
    if family_classifier:
        logger.info("\nCreating combined obfuscation + family model...")
        combined = CombinedObfuscationFamilyModel(ensemble, family_classifier)
        save_dir = os.path.join(project_root, "models")
        joblib.dump(combined, os.path.join(save_dir, "combined_model.pkl"))
        logger.info("✓ Combined model saved")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    logger.info("\nTrained models:")
    logger.info("  - Random Forest: models/random_forest_model.pkl")
    logger.info("  - XGBoost: models/xgboost_model.json")
    if NN_AVAILABLE:
        logger.info("  - Neural Network: models/neural_network_model.pt")
    logger.info("  - Ensemble: models/ensemble_model.pkl")
    if family_classifier:
        logger.info("  - Family Classifier: models/family_classifier.pkl")
        logger.info("  - Combined Model: models/combined_model.pkl")
    
    logger.info("\nEvaluation results: data/evaluation_results/")


if __name__ == "__main__":
    main()
