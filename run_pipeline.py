"""Run the unified pipeline with minimal logging"""
import sys
import os

# Suppress verbose logging
os.environ['LOGURU_LEVEL'] = 'WARNING'

from src.data_pipeline.unified_pipeline import UnifiedDataPipeline

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = UnifiedDataPipeline()
    
    print("Running unified data pipeline...")
    
    # Create feature matrix
    features = pipeline.create_feature_matrix()
    
    if not features.empty:
        # Generate signals
        signals = pipeline.generate_signals(features)
        
        # Save processed data
        pipeline.save_processed_data(features, signals)
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"✓ Features shape: {features.shape}")
        print(f"✓ Current signal: {signals['final_signal'].iloc[-1]}")
        print(f"✓ Signal strength: {signals['signal_strength'].iloc[-1]:.2f}")
        print(f"✓ Latest price: ${signals['price'].iloc[-1]:,.0f}")
        print(f"\nProcessed data saved to data/processed/")