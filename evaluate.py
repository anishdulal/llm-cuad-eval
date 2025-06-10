#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUAD-QA Evaluation Script for Llama 3.2 3B - ALIGNED WITH FINE-TUNING
Supports both original and fine-tuned models
Uses SQuAD-style evaluation metrics
Matches the exact prompt format and parameters used during fine-tuning
"""

import os
import re
import string
import json
import argparse
import torch
import time
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================== EVALUATION METRICS ==================

def normalize_answer(s):
    """Normalize answer for comparison (SQuAD-style)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    """Get normalized tokens from string"""
    if not s:
        return []
    return normalize_answer(s).split()

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth"""
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate max metric over all ground truths"""
    if not ground_truths:
        return 0.0
    scores = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores.append(score)
    return max(scores)

# ================== MODEL WRAPPER - ALIGNED WITH FINE-TUNING ==================

class CUADEvaluator:
    def __init__(self, model_path, max_seq_length=65536):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load model and tokenizer - ALIGNED WITH FINE-TUNING SCRIPT"""
        print(f"Loading model from: {self.model_path}")
        print(f"Max sequence length: {self.max_seq_length}")

        try:
            # Use same configuration as fine-tuning script
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto detection like in fine-tuning
                load_in_4bit=True,
                device_map="auto",
            )

            # Set pad_token_id like in fine-tuning script
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Enable inference mode - FOLLOWING UNSLOTH TUTORIAL
            FastLanguageModel.for_inference(self.model)
            self.model.eval()

            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def clean_response(self, response):
        """Clean and extract answer from model response - IMPROVED"""
        if not response:
            return "Not found"

        # Remove common prefixes that models add
        prefixes_to_remove = [
            "Based on the provided contract",
            "Upon reviewing the contract",
            "The part of this contract",
            "The parts of this contract",
            "After analyzing the contract",
            "Upon analyzing the contract",
            "In this contract",
            "The contract",
            "Looking at",
            "According to",
            "From the contract",
            "The relevant",
            "Here is",
            "Here are",
            "The answer is",
            "Answer:",
            "Response:",
            "The exact phrase",
            "Extract exact phrase",
        ]

        cleaned = response.strip()

        # Remove common prefixes (case insensitive)
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                # Remove common continuation patterns
                if cleaned.startswith(":"):
                    cleaned = cleaned[1:].strip()
                if cleaned.startswith(","):
                    cleaned = cleaned[1:].strip()
                break

        # Split by common separators and take the first substantive part
        separators = ['\n', '. ', ':', ' that should be reviewed', ' are:', ' is:']
        for sep in separators:
            if sep in cleaned:
                parts = cleaned.split(sep)
                if len(parts) > 1:
                    # Take the first non-empty meaningful part
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 3 and not part.lower().startswith(('the following', 'as follows')):
                            cleaned = part
                            break
                break

        # Final cleanup
        cleaned = cleaned.strip()

        # Remove quotes if they wrap the entire answer
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()

        # If still too verbose or empty, return "Not found"
        if not cleaned or len(cleaned) > 200 or cleaned.lower().startswith(('there is no', 'no specific', 'i did not find', 'not found')):
            return "Not found"

        return cleaned

    def generate_answer(self, question, context, max_new_tokens=64):
        """Generate answer - EXACTLY MATCHING FINE-TUNING PROMPT FORMAT"""

        # Context truncation - same logic as fine-tuning
        max_context_length = self.max_seq_length - 1000
        if len(context) > max_context_length:
            # Keep beginning and end of context
            half_length = max_context_length // 2
            context = context[:half_length] + "\n[...TRUNCATED...]\n" + context[-half_length:]

        # EXACT SAME PROMPT FORMAT AS FINE-TUNING SCRIPT
        prompt = f"You are a legal document analyzer. Extract exact phrases from legal documents to answer questions. Only provide the exact text/phrases from the document that answer the question. Do not add explanations or commentary. If the information is not found, respond with 'Not found'.\n\nDocument: {context}\n\nQuestion: {question}\n\nAnswer (extract exact phrase from document):"

        try:
            # Tokenize with same approach as fine-tuning script
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length - max_new_tokens
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=1.5,  # Same as fine-tuning script inference test
                    min_p=0.1,       # Same as fine-tuning script inference test
                    do_sample=True,  # Same as fine-tuning script
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part (skip the input prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][tokenized['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Apply post-processing
            cleaned_response = self.clean_response(generated_text)

            return cleaned_response

        except Exception as e:
            print(f"Error during generation: {e}")
            return "Error generating response"

# ================== EVALUATION FUNCTIONS ==================

def evaluate_dataset(evaluator, dataset, max_samples=None, save_predictions=True, output_dir="./"):
    """Evaluate model on dataset"""

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    predictions = {}
    detailed_results = []
    total_em = 0
    total_f1 = 0

    start_time = time.time()

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            question_id = example.get('id', f"question_{i}")
            question = example['question']
            context = example['context']

            # Handle answers in CUAD format - same as fine-tuning preprocessing
            if example['answers']['text'] and len(example['answers']['text']) > 0:
                # Take the first non-empty answer, strip whitespace
                ground_truths = [next((a.strip() for a in example['answers']['text'] if a.strip()), "Not found")]
                # If still empty after stripping, use "Not found"
                if not ground_truths[0]:
                    ground_truths = ["Not found"]
            else:
                ground_truths = ["Not found"]  # Consistent with fine-tuning preprocessing

            # Generate prediction
            prediction = evaluator.generate_answer(question, context)
            predictions[question_id] = prediction

            # Calculate metrics
            em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

            total_em += em
            total_f1 += f1

            # Store detailed result
            detailed_results.append({
                'id': question_id,
                'question': question[:200] + "..." if len(question) > 200 else question,
                'context_length': len(context),
                'ground_truths': ground_truths,
                'prediction': prediction,
                'exact_match': em,
                'f1_score': f1
            })

            if (i + 1) % 10 == 0:
                current_em = (total_em / (i + 1)) * 100
                current_f1 = (total_f1 / (i + 1)) * 100
                print(f"Progress: {i+1}/{len(dataset)} | EM: {current_em:.2f}% | F1: {current_f1:.2f}%")

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

    end_time = time.time()
    evaluation_time = end_time - start_time

    # Calculate final metrics
    num_evaluated = len(detailed_results)
    if num_evaluated == 0:
        print("❌ No examples were successfully evaluated!")
        return None

    final_em = (total_em / num_evaluated) * 100
    final_f1 = (total_f1 / num_evaluated) * 100

    # Prepare results
    results = {
        'model_path': evaluator.model_path,
        'dataset_size': num_evaluated,
        'exact_match': final_em,
        'f1_score': final_f1,
        'evaluation_time_seconds': evaluation_time,
        'samples_per_second': num_evaluated / evaluation_time,
        'timestamp': datetime.now().isoformat(),
        'detailed_results': detailed_results,
    }

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {evaluator.model_path}")
    print(f"Samples Evaluated: {num_evaluated}")
    print(f"Exact Match: {final_em:.2f}%")
    print(f"F1 Score: {final_f1:.2f}%")
    print(f"Evaluation Time: {evaluation_time/60:.2f} minutes")
    print(f"Speed: {num_evaluated/evaluation_time:.2f} samples/second")
    print("="*80)

    # Save results
    if save_predictions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(evaluator.model_path).replace("/", "_")

        # Save predictions
        pred_file = os.path.join(output_dir, f"predictions_{model_name}_{timestamp}.json")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        # Save detailed results
        results_file = os.path.join(output_dir, f"results_{model_name}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📁 Predictions saved to: {pred_file}")
        print(f"📁 Detailed results saved to: {results_file}")

    return results

# ================== MAIN FUNCTION ==================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.2 3B on CUAD dataset")
    parser.add_argument("--model_path",
                       default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                       help="HuggingFace model repo or local path to fine-tuned model")
    parser.add_argument("--finetuned_model_path",
                       default="./cuad_finetuned_llama3_2_3b",
                       help="Path to fine-tuned model for comparison")
    parser.add_argument("--split",
                       default="test",
                       choices=["train", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--max_samples",
                       type=int,
                       default=4182,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--evaluate_original",
                       action="store_true",
                       default=True,
                       help="Evaluate original model")
    parser.add_argument("--evaluate_finetuned",
                       action="store_true",
                       default=True,
                       help="Evaluate fine-tuned model")
    parser.add_argument("--output_dir",
                       default="./evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--max_seq_length",
                       type=int,
                       default=65536,
                       help="Maximum sequence length - should match fine-tuning")

    args, _ = parser.parse_known_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 CUAD EVALUATION SCRIPT - ALIGNED WITH FINE-TUNING")
    print("="*60)
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Output directory: {args.output_dir}")

    # Load dataset
    print(f"\n📂 Loading CUAD {args.split} dataset...")
    try:
        dataset = load_dataset("theatticusproject/cuad-qa", split=args.split, trust_remote_code=True)
        print(f"✅ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    results_summary = []

    # Evaluate original model
    if args.evaluate_original:
        print(f"\n🔄 EVALUATING ORIGINAL MODEL")
        print("="*50)

        evaluator_original = CUADEvaluator(
            model_path=args.model_path,
            max_seq_length=args.max_seq_length,
        )

        if evaluator_original.load_model():
            original_results = evaluate_dataset(
                evaluator_original,
                dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir
            )
            if original_results:
                results_summary.append(('Original', original_results))

        # Clear memory
        del evaluator_original
        torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    if args.evaluate_finetuned:
        print(f"\n🔄 EVALUATING FINE-TUNED MODEL")
        print("="*50)

        if not os.path.exists(args.finetuned_model_path):
            print(f"⚠️  Fine-tuned model not found at: {args.finetuned_model_path}")
            print("Skipping fine-tuned evaluation...")
        else:
            evaluator_finetuned = CUADEvaluator(
                model_path=args.finetuned_model_path,
                max_seq_length=args.max_seq_length,
            )

            if evaluator_finetuned.load_model():
                finetuned_results = evaluate_dataset(
                    evaluator_finetuned,
                    dataset,
                    max_samples=args.max_samples,
                    output_dir=args.output_dir
                )
                if finetuned_results:
                    results_summary.append(('Fine-tuned', finetuned_results))

            # Clear memory
            del evaluator_finetuned
            torch.cuda.empty_cache()

    # Compare results
    if len(results_summary) >= 2:
        print(f"\n📊 COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<15} {'Exact Match':<15} {'F1 Score':<15} {'Samples':<10} {'Time (min)':<12}")
        print("-" * 80)

        for model_name, results in results_summary:
            print(f"{model_name:<15} {results['exact_match']:<15.2f} "
                  f"{results['f1_score']:<15.2f} {results['dataset_size']:<10} "
                  f"{results['evaluation_time_seconds']/60:<12.2f}")

        # Calculate improvement
        if len(results_summary) == 2:
            original_em = results_summary[0][1]['exact_match']
            original_f1 = results_summary[0][1]['f1_score']
            finetuned_em = results_summary[1][1]['exact_match']
            finetuned_f1 = results_summary[1][1]['f1_score']

            em_improvement = finetuned_em - original_em
            f1_improvement = finetuned_f1 - original_f1

            print(f"\n🎯 IMPROVEMENT:")
            print(f"Exact Match: {em_improvement:+.2f} percentage points")
            print(f"F1 Score: {f1_improvement:+.2f} percentage points")

        print("="*80)

    print(f"\n✅ Evaluation completed! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()