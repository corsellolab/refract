import os
import sys
import argparse

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(refract_path)

from refract.visualize import summarize_model_results

def main():
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    summarize_model_results(args.model_dir, args.output_dir)

if __name__ == "__main__":
    main() 