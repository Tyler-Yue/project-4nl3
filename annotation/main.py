"""
CSV Annotation Tool - Adjusted for Specific Dataset Columns
Columns: class_index, question_title, question_content, best_answer
"""

import json
import os
import sys
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# UPDATED CATEGORIES (1-10)
CATEGORIES = {
    1: "Society & Culture",
    2: "Science & Mathematics",
    3: "Health",
    4: "Education & Reference",
    5: "Computers & Internet",
    6: "Sports",
    7: "Business & Finance",
    8: "Entertainment & Music",
    9: "Family & Relationships",
    10: "Politics & Government"
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_categories():
    print("\nCATEGORIES:")
    # Display in two columns for better use of space
    items = list(CATEGORIES.items())
    for i in range(0, len(items), 2):
        left = f"{items[i][0]}. {items[i][1]}"
        right = f"{items[i+1][0]}. {items[i+1][1]}" if i+1 < len(items) else ""
        print(f"  {left:<30} {right}")

def display_document(doc_id, title, content, answer, total_docs, annotated_count):
    clear_screen()
    print(f"ANNOTATION TOOL - Doc ID: {doc_id} (Progress: {annotated_count}/{total_docs})")
    print("=" * 80)
    print(f"TITLE: {title}")
    print("-" * 80)
    print(f"QUESTION:\n{content}")
    print("-" * 80)
    print(f"BEST ANSWER:\n{answer}")
    print("=" * 80)

def get_annotation():
    display_categories()
    print("\nCOMMANDS: [1-10] = Category | 's' = Skip | 'q' = Save & Quit")
    
    while True:
        response = input("\nSelect Category: ").strip().lower()
        if response == 'q': return 'quit'
        if response == 's': return 'skip'
        
        try:
            choice = int(response)
            if 1 <= choice <= 10:
                return choice
            else:
                print("ERROR: Please enter 1 to 10.")
        except ValueError:
            print("ERROR: Invalid input.")

def load_data():
    # Looks for 'data.csv' in the same folder as the script
    csv_path = SCRIPT_DIR / "data.csv"
    
    if not csv_path.exists():
        print(f"ERROR: 'data.csv' not found in {SCRIPT_DIR}")
        sys.exit(1)
        
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Verify columns
        required = ['class_index', 'question_title', 'question_content', 'best_answer']
        if not all(col in df.columns for col in required):
            print(f"ERROR: CSV must have columns: {required}")
            sys.exit(1)

        documents = []
        for i, row in df.iterrows():
            documents.append({
                'id': i,
                'title': row['question_title'],
                'content': row['question_content'],
                'answer': row['best_answer'],
                'original_label': str(row['class_index'])
            })
        
        return documents
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def load_annotations(annotator_name):
    filename = SCRIPT_DIR / f"annotations_{annotator_name}.json"
    if filename.exists():
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'annotator': annotator_name, 'created_at': datetime.now().isoformat(), 'annotations': []}

def save_annotations(annotation_data, annotator_name, mode='new'):
    suffix = '_reannotation' if mode == 'reannotate' else ''
    filename = SCRIPT_DIR / f"annotations_{annotator_name}{suffix}.json"
    annotation_data['last_updated'] = datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2)
    
    # Export CSV copy
    csv_filename = SCRIPT_DIR / f"annotations_{annotator_name}{suffix}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['document_id', 'label', 'category_name', 'timestamp'])
        for ann in annotation_data['annotations']:
            writer.writerow([ann['document_id'], ann['category_number'], ann['category_name'], ann['timestamp']])

def main():
    print("--- CUSTOM DATASET ANNOTATOR ---")
    name = input("Enter your name: ").strip().lower()
    if not name: return

    mode_choice = input("1. New Annotation\n2. Re-annotation\nChoose (1/2): ")
    mode = 'new' if mode_choice == '1' else 'reannotate'
    
    documents = load_data()
    annotation_data = load_annotations(name)
    
    # Simple logic to find next unannotated document
    annotated_ids = {a['document_id'] for a in annotation_data['annotations']}
    
    try:
        for doc in documents:
            if doc['id'] in annotated_ids:
                continue
                
            display_document(doc['id'], doc['title'], doc['content'], doc['answer'], len(documents), len(annotated_ids))
            
            res = get_annotation()
            if res == 'quit': break
            if res == 'skip': continue
            
            annotation_data['annotations'].append({
                'document_id': doc['id'],
                'category_number': res,
                'category_name': CATEGORIES[res],
                'timestamp': datetime.now().isoformat()
            })
            annotated_ids.add(doc['id'])
            save_annotations(annotation_data, name, mode)
            
    except KeyboardInterrupt:
        print("\nSession paused.")
    finally:
        save_annotations(annotation_data, name, mode)
        print(f"Saved {len(annotation_data['annotations'])} total annotations.")

if __name__ == "__main__":
    main()