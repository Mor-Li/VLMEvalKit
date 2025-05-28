#!/usr/bin/env python3
# compare_MMSI_bench_fixed.py - Fixed version for model comparison
import os
import argparse
import pandas as pd
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vlmeval.dataset.mmsi_bench import MMSIBenchDataset

class MMSIBenchComparer:
    def __init__(self, model_names):
        self.model_names = model_names
        self.dataset = MMSIBenchDataset(dataset='MMSI_Bench')
        self.models_data = {}
        self.load_model_results()
        self.indices = []
        
    def load_model_results(self):
        """Load all model results"""
        for model in self.model_names:
            file_path = f'outputs/{model}/{model}_MMSI_Bench_score.xlsx'
            if os.path.exists(file_path):
                self.models_data[model] = pd.read_excel(file_path)
                print(f"Loaded {model} data: {len(self.models_data[model])} entries")
            else:
                print(f"Warning: File not found: {file_path}")
        
        # Get all categories for filtering
        if len(self.models_data) > 0:
            first_model = list(self.models_data.keys())[0]
            # Filter out NaN values and then sort
            categories = [cat for cat in self.models_data[first_model]['category'].unique() 
                          if isinstance(cat, str) and cat != 'nan']
            self.categories = sorted(categories)
            
            # Get all available indices
            self.indices = sorted(self.models_data[first_model]['index'].unique())
        else:
            self.categories = []
    
    def get_image_paths(self, idx):
        """Get image paths for a given question index"""
        try:
            line = self.dataset.data[self.dataset.data['index'] == idx].iloc[0]
            paths = self.dataset.dump_image(line)
            if isinstance(paths, list):
                return paths
            elif paths:
                return [paths]
            else:
                return []
        except Exception as e:
            print(f"Error getting images for index {idx}: {e}")
            return []
    
    def create_ui(self):
        """Create Gradio interface"""
        with gr.Blocks(title="MMSI Bench Model Comparison") as ui:
            gr.Markdown("## MMSI Bench Model Comparison")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Simple numeric input for index
                    question_index = gr.Number(label="Question Index", value=self.indices[0] if self.indices else 0, precision=0)
                    category_dropdown = gr.Dropdown(choices=["All"] + self.categories, label="Filter by Category", value="All")
                    
                    # Display questions for current category
                    available_questions = gr.DataFrame(label="Available Questions")
                    
                    nav_row = gr.Row()
                    with nav_row:
                        prev_btn = gr.Button("← Previous")
                        next_btn = gr.Button("Next →")
                    
                    stats_md = gr.Markdown(self.get_stats_by_category())
                
                with gr.Column(scale=2):
                    # Output components
                    image_gallery = gr.Gallery(label="Question Images", columns=2, height="auto")
                    question_text = gr.Textbox(label="Question", lines=6, interactive=False)
                    answer_text = gr.Textbox(label="Correct Answer", interactive=False)
                    category_text = gr.Textbox(label="Category", interactive=False)
                    
                    with gr.Row():
                        model_comparison = gr.DataFrame(headers=["Model", "Prediction", "Correct?"], interactive=False)
            
            # Define functions to update UI
            def update_available_questions(category):
                if not self.models_data:
                    return pd.DataFrame()
                
                first_model = list(self.models_data.keys())[0]
                df = self.models_data[first_model]
                
                # Filter by category
                if category != "All":
                    df = df[df['category'] == category]
                
                # Create a simplified dataframe with index, category, and question preview
                result = []
                for _, row in df.iterrows():
                    question_preview = str(row['question'])[:50].replace('\n', ' ')
                    result.append({
                        "Index": row['index'], 
                        "Category": row['category'], 
                        "Question Preview": f"{question_preview}..."
                    })
                
                return pd.DataFrame(result)
            
            def update_question_display(idx):
                if idx is None:
                    return [], "", "", "", pd.DataFrame()
                
                try:
                    # Convert to int
                    idx = int(idx)
                    
                    # Get images
                    image_paths = self.get_image_paths(idx)
                    images = []
                    
                    for path in image_paths:
                        if os.path.exists(path):
                            try:
                                img = Image.open(path)
                                images.append(img)
                            except Exception as e:
                                print(f"Error opening image {path}: {e}")
                                continue
                    
                    # Get question data
                    first_model = list(self.models_data.keys())[0]
                    question_data = self.models_data[first_model][self.models_data[first_model]['index'] == idx]
                    
                    if question_data.empty:
                        return [], f"No data found for index {idx}", "", "", pd.DataFrame()
                    
                    question_data = question_data.iloc[0]
                    
                    # Create model comparison dataframe
                    comparison_data = []
                    for model in self.model_names:
                        if model in self.models_data:
                            model_row = self.models_data[model][self.models_data[model]['index'] == idx]
                            if not model_row.empty:
                                row = model_row.iloc[0]
                                prediction = row['prediction'] if pd.notna(row['prediction']) else "No prediction"
                                extracted = row['extracted_pred'] if pd.notna(row['extracted_pred']) else "-"
                                is_correct = "✓" if row['score'] == 1.0 else "✗"
                                comparison_data.append({
                                    "Model": model, 
                                    "Prediction": f"{extracted} ({prediction[:50]}{'...' if len(prediction) > 50 else ''})",
                                    "Correct?": is_correct
                                })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    return (
                        images,
                        question_data['question'],
                        question_data['answer'],
                        question_data['category'],
                        comparison_df
                    )
                except Exception as e:
                    print(f"Error processing question {idx}: {e}")
                    return [], f"Error: {str(e)}", "", "", pd.DataFrame()
            
            def get_next_question(idx, category):
                if not self.indices:
                    return 0
                
                # Get valid indices for the category
                valid_indices = self.get_category_indices(category)
                if not valid_indices:
                    return self.indices[0]
                
                # Find next index in the filtered list
                try:
                    current_idx = int(idx)
                    # Find the next index that is larger than the current one
                    next_indices = [i for i in valid_indices if i > current_idx]
                    if next_indices:
                        return min(next_indices)
                    else:
                        # Wrap around to the beginning if we're at the end
                        return min(valid_indices)
                except:
                    return min(valid_indices)
            
            def get_prev_question(idx, category):
                if not self.indices:
                    return 0
                
                # Get valid indices for the category
                valid_indices = self.get_category_indices(category)
                if not valid_indices:
                    return self.indices[0]
                
                # Find prev index in the filtered list
                try:
                    current_idx = int(idx)
                    # Find the previous index that is smaller than the current one
                    prev_indices = [i for i in valid_indices if i < current_idx]
                    if prev_indices:
                        return max(prev_indices)
                    else:
                        # Wrap around to the end if we're at the beginning
                        return max(valid_indices)
                except:
                    return max(valid_indices)
            
            # Connect UI components
            question_index.change(
                update_question_display,
                inputs=question_index,
                outputs=[image_gallery, question_text, answer_text, category_text, model_comparison]
            )
            
            category_dropdown.change(
                update_available_questions,
                inputs=category_dropdown,
                outputs=available_questions
            )
            
            next_btn.click(
                lambda idx, cat: get_next_question(idx, cat),
                inputs=[question_index, category_dropdown],
                outputs=question_index
            )
            
            prev_btn.click(
                lambda idx, cat: get_prev_question(idx, cat),
                inputs=[question_index, category_dropdown],
                outputs=question_index
            )
            
            # Initialize displays
            ui.load(
                update_available_questions,
                inputs=category_dropdown,
                outputs=available_questions
            )
            
            ui.load(
                update_question_display,
                inputs=question_index,
                outputs=[image_gallery, question_text, answer_text, category_text, model_comparison]
            )
            
        return ui
    
    def get_category_indices(self, category):
        """Get indices for a specific category"""
        if not self.models_data:
            return []
        
        first_model = list(self.models_data.keys())[0]
        df = self.models_data[first_model]
        
        # Filter by category
        if category != "All":
            df = df[df['category'] == category]
        
        return sorted(df['index'].unique())
    
    def get_stats_by_category(self):
        """Generate model accuracy statistics by category"""
        if not self.models_data:
            return "No model data available"
        
        stats_text = "### Model Performance by Category\n\n"
        
        # Overall accuracy for each model
        stats_text += "#### Overall Accuracy\n\n"
        stats_text += "| Model | Accuracy |\n"
        stats_text += "|-------|----------|\n"
        
        for model in self.model_names:
            if model in self.models_data:
                accuracy = self.models_data[model]['score'].mean() * 100
                stats_text += f"| {model} | {accuracy:.2f}% |\n"
        
        # Category-wise accuracy
        stats_text += "\n#### Category-wise Accuracy\n\n"
        
        # Create a table with models as columns and categories as rows
        stats_text += "| Category | " + " | ".join(self.model_names) + " |\n"
        stats_text += "|----------|" + "|".join(["-------" for _ in self.model_names]) + "|\n"
        
        for category in self.categories:
            stats_text += f"| {category} |"
            for model in self.model_names:
                if model in self.models_data:
                    cat_data = self.models_data[model][self.models_data[model]['category'] == category]
                    cat_accuracy = cat_data['score'].mean() * 100 if not cat_data.empty else 0
                    stats_text += f" {cat_accuracy:.2f}% |"
            stats_text += "\n"
            
        return stats_text

def parse_args():
    parser = argparse.ArgumentParser(description="Compare model predictions on MMSI_Bench")
    parser.add_argument("--models", type=str, nargs="+", required=True, 
                        help="List of model names to compare")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    comparer = MMSIBenchComparer(args.models)
    ui = comparer.create_ui()
    ui.launch(share=True, server_port=7868) 