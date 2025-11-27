"""
Simple script to check statistics about is_dead and is_eating in metadata CSV files.
"""

import pandas as pd
import argparse
import os

def analyze_metadata(metadata_file):
    """Analyze metadata CSV and print statistics."""
    
    if not os.path.exists(metadata_file):
        print(f"Error: File not found: {metadata_file}")
        return
    
    print(f"Loading metadata from: {metadata_file}")
    df = pd.read_csv(metadata_file)
    
    print(f"\nTotal rows: {len(df):,}")
    
    # Count is_dead
    if 'is_dead' in df.columns:
        # Handle both boolean and string representations
        is_dead_true = df['is_dead'].astype(str).str.lower().isin(['true', '1', '1.0']).sum()
        is_dead_false = df['is_dead'].astype(str).str.lower().isin(['false', '0', '0.0']).sum()
        is_dead_total = is_dead_true + is_dead_false
        
        print(f"\n--- is_dead Statistics ---")
        print(f"  True:  {is_dead_true:,} ({is_dead_true/len(df)*100:.2f}%)")
        print(f"  False: {is_dead_false:,} ({is_dead_false/len(df)*100:.2f}%)")
    else:
        print("\n⚠️  Column 'is_dead' not found in metadata")
    
    # Count is_eating
    if 'is_eating' in df.columns:
        # Handle both boolean and string representations
        is_eating_true = df['is_eating'].astype(str).str.lower().isin(['true', '1', '1.0']).sum()
        is_eating_false = df['is_eating'].astype(str).str.lower().isin(['false', '0', '0.0']).sum()
        is_eating_total = is_eating_true + is_eating_false
        
        print(f"\n--- is_eating Statistics ---")
        print(f"  True:  {is_eating_true:,} ({is_eating_true/len(df)*100:.2f}%)")
        print(f"  False: {is_eating_false:,} ({is_eating_false/len(df)*100:.2f}%)")
    else:
        print("\n⚠️  Column 'is_eating' not found in metadata")
    
    # Additional stats
    if 'is_dead' in df.columns and 'is_eating' in df.columns:
        # Count frames where both are true (shouldn't happen, but check)
        both_true = (
            df['is_dead'].astype(str).str.lower().isin(['true', '1', '1.0']) &
            df['is_eating'].astype(str).str.lower().isin(['true', '1', '1.0'])
        ).sum()
        
        if both_true > 0:
            print(f"\n⚠️  Warning: {both_true} rows have both is_dead=True and is_eating=True")
    
    # Episode statistics
    if 'episode_id' in df.columns:
        num_episodes = df['episode_id'].nunique()
        print(f"\n--- Episode Statistics ---")
        print(f"  Total episodes: {num_episodes:,}")
        print(f"  Average frames per episode: {len(df)/num_episodes:.1f}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description="Check statistics in metadata CSV files")
    parser.add_argument("--metadata", type=str, default="data_v5/metadata.csv",
                       help="Path to metadata CSV file")
    args = parser.parse_args()
    
    analyze_metadata(args.metadata)

if __name__ == "__main__":
    main()

