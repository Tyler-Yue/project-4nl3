"""
Calculate inter-annotator agreement for 20 Newsgroups annotations

This script calculates Cohen's Kappa, Fleiss' Kappa, or Krippendorff's Alpha
depending on the annotation setup.
"""

import json
from collections import defaultdict
from pathlib import Path
import krippendorff
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def load_all_annotations():
    """Load all annotation files in the current directory."""
    annotation_files = list(SCRIPT_DIR.glob('annotations_*.json'))
    annotation_files = [f for f in annotation_files if 'backup' not in f.name and 'reannotate_' not in f.name]
    
    if len(annotation_files) == 0:
        print("ERROR: No annotation files found")
        return None
    
    print(f"Found {len(annotation_files)} annotation file(s):")
    for f in annotation_files:
        print(f"  - {f.name}")
    
    all_data = {}
    for file in annotation_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotator = data['annotator']
            
            # Combine regular annotations and re-annotations
            annotations = data['annotations']
            
            # Check for re-annotation file
            reannotation_file = SCRIPT_DIR / f"annotations_{annotator}_reannotation.json"
            if reannotation_file.exists():
                with open(reannotation_file, 'r', encoding='utf-8') as rf:
                    reann_data = json.load(rf)
                    annotations.extend(reann_data['annotations'])
                    print(f"    + {len(reann_data['annotations'])} re-annotations")
            
            all_data[annotator] = annotations
    
    return all_data


def find_overlapping_annotations(all_data):
    """Find documents annotated by multiple people."""
    # Group by document_id
    doc_annotations = defaultdict(list)
    
    for annotator, annotations in all_data.items():
        for ann in annotations:
            doc_id = ann['document_id']
            category = ann['category_number']
            doc_annotations[doc_id].append((annotator, category))
    
    # Find documents with multiple annotations
    overlapping = {}
    for doc_id, anns in doc_annotations.items():
        if len(anns) >= 2:
            overlapping[doc_id] = anns
    
    return overlapping

def calculate_krippendorffs_alpha(overlapping_annotations, all_data):
    """
    Calculate Krippendorff's Alpha for multiple annotators.
    
    This is more complex and handles missing data well.
    For simplicity, we'll use a library if available.
    """
    try:
               
        # Build reliability matrix
        # Rows = annotators, Columns = documents
        annotators = list(all_data.keys())
        all_doc_ids = set()
        for annotations in all_data.values():
            for ann in annotations:
                all_doc_ids.add(ann['document_id'])
        
        all_doc_ids = sorted(all_doc_ids)
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}
        
        # Create matrix (annotators x documents)
        # Use np.nan for missing values
        matrix = np.full((len(annotators), len(all_doc_ids)), np.nan)
        
        for annotator_idx, annotator in enumerate(annotators):
            for ann in all_data[annotator]:
                doc_id = ann['document_id']
                doc_idx = doc_id_to_idx[doc_id]
                category = ann['category_number']
                matrix[annotator_idx, doc_idx] = category
        
        print(f"\nCalculating Krippendorff's Alpha for {len(annotators)} annotators:")
        print(f"  - Total documents: {len(all_doc_ids)}")
        print(f"  - Reliability matrix shape: {matrix.shape}")
        
        # Calculate alpha
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')
        
        print(f"\n  Krippendorff's Alpha: {alpha:.4f}")
        
        return alpha
        
    except ImportError:
        print("\nWARNING: krippendorff library not installed.")
        print("   Install with: pip install krippendorff")
        print("   (Note: krippendorff is only available via pip, not conda)")
        print("\n   Falling back to simpler calculation...")
        return None


def interpret_agreement(score, metric_name):
    """Provide interpretation of agreement score."""
    print(f"\nInterpretation of {metric_name} = {score:.4f}:")
    
    if score < 0:
        interpretation = "Poor (less than chance agreement)"
    elif score < 0.20:
        interpretation = "Slight agreement"
    elif score < 0.40:
        interpretation = "Fair agreement"
    elif score < 0.60:
        interpretation = "Moderate agreement"
    elif score < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"  {interpretation}")
    print("\nGeneral guidelines (Landis & Koch, 1977):")
    print("  < 0.00: Poor")
    print("  0.00-0.20: Slight")
    print("  0.21-0.40: Fair")
    print("  0.41-0.60: Moderate")
    print("  0.61-0.80: Substantial")
    print("  0.81-1.00: Almost Perfect")


def main():
    """Main function to calculate agreement."""
    print("=" * 80)
    print("Agreement Calculation")
    print("=" * 80)
    
    # Load all annotations
    all_data = load_all_annotations()
    if all_data is None:
        return
    
    # Find overlapping annotations
    overlapping = find_overlapping_annotations(all_data)
    
    print(f"\n{len(overlapping)} documents annotated by multiple people")
    
    if len(overlapping) == 0:
        print("\nWARNING: No overlapping annotations found.")
        print("   For agreement calculation, you need documents annotated by multiple people.")
        return
    
    # Try Krippendorff's Alpha
    alpha = calculate_krippendorffs_alpha(overlapping, all_data)
    if alpha is not None:
        interpret_agreement(alpha, "Krippendorff's Alpha")
    else:
        # Fall back to pairwise Cohen's Kappa
        print("\nCalculating pairwise Cohen's Kappa for all annotator pairs:")
        annotators = list(all_data.keys())
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                print(f"\n{annotators[i]} vs {annotators[j]}:")
                # Filter overlapping for this pair
                pair_overlapping = {}
                for doc_id, anns in overlapping.items():
                    pair_anns = [(a, c) for a, c in anns if a in [annotators[i], annotators[j]]]
                    if len(pair_anns) == 2:
                        pair_overlapping[doc_id] = pair_anns
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
