from __future__ import annotations                                                                                                                                  
                                                                                                                                                                   
import argparse                                                                                                                                                     
import json                                                                                                                                                         
from pathlib import Path                                                                                                                                            
import numpy as np                                                                                                                                                  
                                                                                                                                                                   
from common.data import load_yaml, read_jsonl, repo_path, write_jsonl                                                                                               
from common.generation import batch_generate                                                                                                                        
from common.models import clear_gpu, load_policy, load_tokenizer                                                                                                    
from task5_feedback.rlaif import PairwiseAIJudge                                                                                                                    
from task5_feedback.rlvr import exact_reward, extract_designated_final                                                                                              
                                                                                                                                                                   
                                                                                                                                                                   
def policy_specs(cfg):                                                                                                                                              
   return {                                                                                                                                                        
       "sft": None,                                                                                                                                                
       "rlvr": cfg["policies"]["rlvr"],                                                                                                                            
       "rlaif": cfg["policies"]["rlaif"],                                                                                                                          
   }                                                                                                                                                               
                                                                                                                                                                   
                                                                                                                                                                   
def dataset_path(cfg, dataset: str):                                                                                                                                
   if dataset == "gsm":                                                                                                                                            
       return cfg["paths"]["gsm_eval"]                                                                                                                             
   if dataset == "transfer":                                                                                                                                       
       return cfg["paths"]["math_transfer_eval"]                                                                                                                   
   raise ValueError(dataset)                                                                                                                                       
                                                                                                                                                                   
                                                                                                                                                                   
def load_math_evaluation(config_path: str, dataset: str):                                                                                                           
   cfg = load_yaml(config_path)                                                                                                                                    
   rows = read_jsonl(dataset_path(cfg, dataset))                                                                                                                   
   tokenizer = load_tokenizer(cfg["base_model"])                                                                                                                   
   return cfg, rows, tokenizer                                                                                                                                     
                                                                                                                                                                   
                                                                                                                                                                   
def load_frozen_policy(cfg, name: str):                                                                                                                             
   specs = policy_specs(cfg)                                                                                                                                       
   if name not in specs:                                                                                                                                           
       raise KeyError(name)                                                                                                                                        
   return load_policy(cfg, adapter_path=specs[name], trainable=False)                                                                                              
                                                                                                                                                                   
                                                                                                                                                                   
def generate_for_policy(                                                                                                                                            
   cfg: dict,                                                                                                                                                      
   policy_name: str,                                                                                                                                               
   dataset: str,                                                                                                                                                   
   rows: list[dict],                                                                                                                                               
   tokenizer,                                                                                                                                                      
   batch_size: int = 8,                                                                                                                                            
   force: bool = False,                                                                                                                                            
) -> list[dict]:                                                                                                                                                    
   outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"                                                                                        
   outdir.mkdir(parents=True, exist_ok=True)                                                                                                                       
   out_file = outdir / f"generated_{dataset}_{policy_name}.jsonl"                                                                                                  
                                                                                                                                                                   
   total_prompts = len(rows)                                                                                                                                       
                                                                                                                                                                   
   if out_file.exists() and not force:                                                                                                                             
       try:                                                                                                                                                        
           existing = read_jsonl(out_file)                                                                                                                         
           if len(existing) == total_prompts:                                                                                                                      
               print(f"[Skip] '{policy_name}' on {dataset} already generated ({total_prompts}/{total_prompts}).", flush=True)                                      
               return existing                                                                                                                                     
       except Exception:                                                                                                                                           
           pass                                                                                                                                                    
                                                                                                                                                                   
   print(f"\n>>> Generating {dataset.upper()} responses for policy: {policy_name.upper()} <<<", flush=True)                                                        
   specs = policy_specs(cfg)                                                                                                                                       
   adapter = specs[policy_name]                                                                                                                                    
   if adapter:                                                                                                                                                     
       adapter_resolved = repo_path(adapter)                                                                                                                       
       if not adapter_resolved.exists():                                                                                                                           
           raise FileNotFoundError(f"Checkpoint for '{policy_name}' not found at {adapter_resolved}.")                                                             
       print(f"Loading checkpoint: {adapter_resolved}", flush=True)                                                                                                
   else:                                                                                                                                                           
       print(f"Using untouched base model: {cfg['base_model']}", flush=True)                                                                                       
                                                                                                                                                                   
   model = load_frozen_policy(cfg, policy_name)                                                                                                                    
   records = []                                                                                                                                                    
   max_new_tokens = int(cfg.get("math_max_new_tokens", 512))                                                                                                       
                                                                                                                                                                   
   for start in range(0, total_prompts, batch_size):                                                                                                               
       chunk = rows[start : start + batch_size]                                                                                                                    
       prompts = [r["messages"] for r in chunk]                                                                                                                    
       gen = batch_generate(                                                                                                                                       
           model,                                                                                                                                                  
           tokenizer,                                                                                                                                              
           prompts,                                                                                                                                                
           max_prompt_length=512,                                                                                                                                  
           max_new_tokens=max_new_tokens,                                                                                                                          
           temperature=0.0,                                                                                                                                        
           top_p=1.0,                                                                                                                                              
           do_sample=False,                                                                                                                                        
       )                                                                                                                                                           
                                                                                                                                                                   
       for raw_row, resp, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):                                                                          
           gold_str = str(raw_row.get("gold_final", "")).strip()                                                                                                   
           extracted = extract_designated_final(resp)                                                                                                              
           is_correct = bool(exact_reward(resp, gold_str) == 1.0)                                                                                                  
           is_format = bool(extracted is not None)                                                                                                                 
                                                                                                                                                                   
           record = {                                                                                                                                              
               "source_index": raw_row.get("source_index", raw_row.get("prompt_id", len(records))),                                                                
               "dataset": dataset,                                                                                                                                 
               "policy": policy_name,                                                                                                                              
               "question": raw_row.get("question", ""),                                                                                                            
               "gold_final": gold_str,                                                                                                                             
               "response": resp,                                                                                                                                   
               "response_tokens": int(n_tok),                                                                                                                      
               "extracted_final": extracted,                                                                                                                       
               "exact_correct": is_correct,                                                                                                                        
               "format_compliant": is_format,                                                                                                                      
           }                                                                                                                                                       
           records.append(record)                                                                                                                                  
                                                                                                                                                                   
       if (start // batch_size) % 10 == 0 or (start + batch_size >= total_prompts):                                                                                
           print(f"  Processed {min(start + len(chunk), total_prompts)}/{total_prompts} examples...", flush=True)                                                  
                                                                                                                                                                   
   write_jsonl(out_file, records)                                                                                                                                  
   print(f"Saved {len(records)} responses for '{policy_name}' to {out_file}", flush=True)                                                                          
   clear_gpu()                                                                                                                                                     
   return records                                                                                                                                                  
                                                                                                                                                                   
                                                                                                                                                                   
def run_math_evaluation(config_path: str = "configs/feedback.yaml", dataset: str = "gsm", batch_size: int = 8, force: bool = False):                                
   cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)                                                                                               
   outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"                                                                                        
   outdir.mkdir(parents=True, exist_ok=True)                                                                                                                       
                                                                                                                                                                   
   print("==================================================", flush=True)                                                                                         
   print(f"  Task 5: Math Evaluation on {dataset.upper()} ({len(rows)} examples)  ", flush=True)                                                                   
   print("==================================================", flush=True)                                                                                         
                                                                                                                                                                   
   # 1. Generate or load responses for all 3 policies                                                                                                              
   policy_results = {}                                                                                                                                             
   for pol in ["sft", "rlvr", "rlaif"]:                                                                                                                            
       policy_results[pol] = generate_for_policy(                                                                                                                  
           cfg, pol, dataset, rows, tokenizer, batch_size=batch_size, force=force                                                                                  
       )                                                                                                                                                           
                                                                                                                                                                   
   # 2. Compute individual policy metrics                                                                                                                          
   metrics = {}                                                                                                                                                    
   for pol, p_rows in policy_results.items():                                                                                                                      
       n = len(p_rows)                                                                                                                                             
       exact_acc = float(np.mean([1.0 if r["exact_correct"] else 0.0 for r in p_rows]))                                                                            
       format_comp = float(np.mean([1.0 if r["format_compliant"] else 0.0 for r in p_rows]))                                                                       
       lengths = [r["response_tokens"] for r in p_rows]                                                                                                            
       mean_len = float(np.mean(lengths)) if lengths else 0.0                                                                                                      
       std_len = float(np.std(lengths)) if lengths else 0.0                                                                                                        
                                                                                                                                                                   
       metrics[pol] = {                                                                                                                                            
           "exact_accuracy": round(exact_acc, 4),                                                                                                                  
           "format_compliance": round(format_comp, 4),                                                                                                             
           "mean_response_length": round(mean_len, 2),                                                                                                             
           "std_response_length": round(std_len, 2),                                                                                                               
           "total_examples": n,                                                                                                                                    
       }                                                                                                                                                           
                                                                                                                                                                   
   # 3. AI Pairwise Judging vs SFT baseline                                                                                                                        
   cache_path = outdir / f"pairwise_judge_cache_{dataset}.json"                                                                                                    
   print(f"\n>>> Running Pairwise AI Judge on {dataset.upper()} vs SFT Baseline <<<", flush=True)                                                                  
   judge = PairwiseAIJudge(cfg, cache_path=cache_path)                                                                                                             
                                                                                                                                                                   
   pairwise_results = {}                                                                                                                                           
   sft_rows = policy_results["sft"]                                                                                                                                
                                                                                                                                                                   
   for pol in ["rlvr", "rlaif"]:                                                                                                                                   
       p_rows = policy_results[pol]                                                                                                                                
       wins, ties, losses = 0, 0, 0                                                                                                                                
       verifier_judge_agreements = 0                                                                                                                               
       pair_details = []                                                                                                                                           
                                                                                                                                                                   
       for idx, (cand_row, sft_row) in enumerate(zip(p_rows, sft_rows)):                                                                                           
           prob = cand_row["question"]                                                                                                                             
           c_resp = cand_row["response"]                                                                                                                           
           s_resp = sft_row["response"]                                                                                                                            
                                                                                                                                                                   
           pref = judge.compare(prob, c_resp, s_resp)                                                                                                              
           if pref == "A":                                                                                                                                         
               wins += 1                                                                                                                                           
               j_score = 1.0                                                                                                                                       
           elif pref == "TIE":                                                                                                                                     
               ties += 1                                                                                                                                           
               j_score = 0.5                                                                                                                                       
           else:                                                                                                                                                   
               losses += 1                                                                                                                                         
               j_score = 0.0                                                                                                                                       
                                                                                                                                                                   
           # Exact verifier outcome comparison:                                                                                                                    
           # +1 if cand correct and sft wrong; -1 if cand wrong and sft correct; 0 if equal                                                                        
           c_ver = 1.0 if cand_row["exact_correct"] else 0.0                                                                                                       
           s_ver = 1.0 if sft_row["exact_correct"] else 0.0                                                                                                        
           v_delta = c_ver - s_ver                                                                                                                                 
                                                                                                                                                                   
           # Verifier-judge agreement definition:                                                                                                                  
           if (v_delta > 0 and pref == "A") or (v_delta < 0 and pref == "B") or (v_delta == 0 and pref == "TIE"):                                                  
               verifier_judge_agreements += 1                                                                                                                      
                                                                                                                                                                   
           if (idx + 1) % 50 == 0 or (idx + 1) == len(p_rows):                                                                                                     
               print(f"  Judged {idx + 1}/{len(p_rows)} pairs ({pol.upper()} vs SFT)...", flush=True)                                                              
                                                                                                                                                                   
           if idx < 5:                                                                                                                                             
               pair_details.append({                                                                                                                               
                   "problem_index": idx,                                                                                                                           
                   "judge_preference": pref,                                                                                                                       
                   "cand_correct": cand_row["exact_correct"],                                                                                                      
                   "sft_correct": sft_row["exact_correct"],                                                                                                        
                   "agreement": bool((v_delta > 0 and pref == "A") or (v_delta < 0 and pref == "B") or (v_delta == 0 and pref == "TIE")),                          
               })                                                                                                                                                  
                                                                                                                                                                   
       n_pairs = len(p_rows)                                                                                                                                       
       win_rate = float((wins + 0.5 * ties) / max(n_pairs, 1))                                                                                                     
       agree_rate = float(verifier_judge_agreements / max(n_pairs, 1))                                                                                             
                                                                                                                                                                   
       pairwise_results[pol] = {                                                                                                                                   
           "win_rate_vs_sft": round(win_rate, 4),                                                                                                                  
           "wins": wins,                                                                                                                                           
           "ties": ties,                                                                                                                                           
           "losses": losses,                                                                                                                                       
           "verifier_judge_agreement_rate": round(agree_rate, 4),                                                                                                  
           "sample_pairs": pair_details,                                                                                                                           
       }                                                                                                                                                           
       metrics[pol]["win_rate_vs_sft"] = round(win_rate, 4)                                                                                                        
       metrics[pol]["verifier_judge_agreement"] = round(agree_rate, 4)                                                                                             
                                                                                                                                                                   
   # SFT vs SFT reference                                                                                                                                          
   metrics["sft"]["win_rate_vs_sft"] = 0.5000                                                                                                                      
   metrics["sft"]["verifier_judge_agreement"] = 1.0000                                                                                                             
                                                                                                                                                                   
   clear_gpu()                                                                                                                                                     
                                                                                                                                                                   
   # 4. Out-of-Domain Transfer Drops (if dataset == "transfer")                                                                                                    
   transfer_drops = {}                                                                                                                                             
   if dataset == "transfer":                                                                                                                                       
       gsm_summary_file = outdir / "math_eval_gsm.json"                                                                                                            
       if gsm_summary_file.exists():                                                                                                                               
           try:                                                                                                                                                    
               gsm_data = json.loads(gsm_summary_file.read_text(encoding="utf-8"))                                                                                 
               gsm_metrics = gsm_data.get("metrics", {})                                                                                                           
               for pol in ["sft", "rlvr", "rlaif"]:                                                                                                                
                   if pol in gsm_metrics and pol in metrics:                                                                                                       
                       acc_in = gsm_metrics[pol]["exact_accuracy"]                                                                                                 
                       acc_out = metrics[pol]["exact_accuracy"]                                                                                                    
                       wr_in = gsm_metrics[pol].get("win_rate_vs_sft", 0.5)                                                                                        
                       wr_out = metrics[pol].get("win_rate_vs_sft", 0.5)                                                                                           
                       transfer_drops[pol] = {                                                                                                                     
                           "accuracy_drop": round(float(acc_in - acc_out), 4),                                                                                     
                           "win_rate_drop": round(float(wr_in - wr_out), 4),                                                                                       
                           "in_domain_accuracy": acc_in,                                                                                                           
                           "transfer_accuracy": acc_out,                                                                                                           
                           "in_domain_win_rate": wr_in,                                                                                                            
                           "transfer_win_rate": wr_out,                                                                                                            
                       }                                                                                                                                           
           except Exception as e:                                                                                                                                  
               print(f"Notice: Failed to calculate transfer drops from {gsm_summary_file}: {e}", flush=True)                                                       
                                                                                                                                                                   
   final_report = {                                                                                                                                                
       "dataset": dataset,                                                                                                                                         
       "total_examples": len(rows),                                                                                                                                
       "metrics": metrics,                                                                                                                                         
       "pairwise_vs_sft": pairwise_results,                                                                                                                        
       "transfer_drops": transfer_drops if transfer_drops else None,                                                                                               
   }                                                                                                                                                               
                                                                                                                                                                   
   out_file = outdir / f"math_eval_{dataset}.json"                                                                                                                 
   with open(out_file, "w", encoding="utf-8") as f:                                                                                                                
       json.dump(final_report, f, indent=2)                                                                                                                        
                                                                                                                                                                   
   # 5. Print summary table                                                                                                                                        
   print(f"\n--- {dataset.upper()} Mathematical Reasoning Evaluation Summary ---", flush=True)                                                                     
   print("| Policy | Exact Accuracy | Win Rate vs SFT | Format Compliance | Verifier-Judge Agreement | Mean Length |", flush=True)                                 
   print("| :--- | :--- | :--- | :--- | :--- | :--- |", flush=True)                                                                                                
   for pol in ["sft", "rlvr", "rlaif"]:                                                                                                                            
       m = metrics[pol]                                                                                                                                            
       print(                                                                                                                                                      
           f"| {pol.upper()} | {m['exact_accuracy']:.1%} | {m['win_rate_vs_sft']:.1%} | "                                                                          
           f"{m['format_compliance']:.1%} | {m.get('verifier_judge_agreement', 0.0):.1%} | {m['mean_response_length']:.1f} |",                                     
           flush=True,                                                                                                                                             
       )                                                                                                                                                           
                                                                                                                                                                   
   if transfer_drops:                                                                                                                                              
       print("\n--- Out-of-Domain (SVAMP) Generalization Drop (GSM8K -> SVAMP) ---", flush=True)                                                                   
       print("| Policy | In-Domain Acc | Transfer Acc | Accuracy Drop | In-Domain Win-Rate | Transfer Win-Rate | Win-Rate Drop |", flush=True)                     
       print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |", flush=True)                                                                                     
       for pol in ["sft", "rlvr", "rlaif"]:                                                                                                                        
           d = transfer_drops[pol]                                                                                                                                 
           print(                                                                                                                                                  
               f"| {pol.upper()} | {d['in_domain_accuracy']:.1%} | {d['transfer_accuracy']:.1%} | "                                                                
               f"{d['accuracy_drop']:.1%} | {d['in_domain_win_rate']:.1%} | {d['transfer_win_rate']:.1%} | {d['win_rate_drop']:.1%} |",                            
               flush=True,                                                                                                                                         
           )                                                                                                                                                       
                                                                                                                                                                   
   print(f"\nResults saved to {out_file}", flush=True)                                                                                                             
   return final_report                                                                                                                                             
                                                                                                                                                                   
                                                                                                                                                                   
def main():                                                                                                                                                         
   ap = argparse.ArgumentParser()                                                                                                                                  
   ap.add_argument("--config", default="configs/feedback.yaml")                                                                                                    
   ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")                                                                                        
   ap.add_argument("--batch-size", type=int, default=8)                                                                                                            
   ap.add_argument("--force", action="store_true", help="Force regenerate responses and re-score")                                                                 
   args = ap.parse_args()                                                                                                                                          
                                                                                                                                                                   
   run_math_evaluation(args.config, args.dataset, batch_size=args.batch_size, force=args.force)                                                                    
                                                                                                                                                                   
                                                                                                                                                                   
if __name__ == "__main__":                                                                                                                                          
   main()                    
