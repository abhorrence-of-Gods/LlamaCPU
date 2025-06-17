# evaluate.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
import random
import os
import re
import glob
from tqdm import tqdm
import gc

# 自分のファイルからインポート
from swm import HybridSWM
from model import Llama3_NSW_OEU
from main import NeuralCpuExpert 

# --- グローバル設定 ---
CHECKPOINT_DIR = "checkpoints_final_adder"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
PC_SLOT_ADDR = 0

def load_model_for_evaluation(checkpoint_dir: str, device: torch.device):
    """最新の学習済みモデルを評価モードでロードする"""
    print("--- Loading Model for Evaluation ---")
    
    target_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=target_dtype,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    config = {
        'swm_slots': 256, 'swm_slot_dim': 512, 'swm_ptr_dim': 64, 'swm_num_ptr': 4
    }
    swm = HybridSWM(batch_size=1, num_slots=config['swm_slots'], slot_dim=config['swm_slot_dim'], 
                    pointer_dim=config['swm_ptr_dim'], num_pointers=config['swm_num_ptr'], dtype=target_dtype).to(device)
    
    model = Llama3_NSW_OEU(base_model.config, swm, base_model=base_model)
    
    lora_dirs = glob.glob(os.path.join(checkpoint_dir, "lora_stage_*"))
    if not lora_dirs:
        raise FileNotFoundError(f"No LoRA checkpoints found in {checkpoint_dir}")
    latest_lora_path = max(lora_dirs, key=lambda p: int(re.search(r'lora_stage_(\d+)', os.path.basename(p)).group(1)))

    print(f"Loading LoRA weights from: {latest_lora_path}")
    model.llama = PeftModel.from_pretrained(model.llama.base_model, latest_lora_path)
    model.llama = model.llama.merge_and_unload()
    print("LoRA weights merged into the base model.")

    pt_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if pt_checkpoints:
        latest_pt_path = max(pt_checkpoints, key=lambda p: int(re.search(r'stage_(\d+)', os.path.basename(p)).group(1)))
        print(f"Loading head weights from: {latest_pt_path}")
        checkpoint = torch.load(latest_pt_path, map_location=device)
        model.load_state_dict(checkpoint['head_state_dict'], strict=False)
    
    model.to(device, dtype=target_dtype)
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer

def run_inference(model, tokenizer, num_digits: int, verbose: bool = False):
    """
    指定された桁数の問題に対して、モデルの自律的な推論を実行し、結果を評価する。
    """
    device = model.swm.memory.device
    target_dtype = model.swm.memory.dtype

    def spec_to_tensor(spec, swm):
        slot = torch.zeros(swm.slot_dim, device=device, dtype=target_dtype)
        slot[0] = spec['type_val']
        ptr_start_idx = swm.type_dim
        for i, ptr_addr in enumerate(spec['pointers']):
            if ptr_addr is None: continue
            ptr_key = swm.keys[ptr_addr]
            slot[ptr_start_idx + i*swm.pointer_dim : ptr_start_idx + (i+1)*swm.pointer_dim] = ptr_key
        if spec['content'] is not None:
            content_tensor = torch.tensor(spec['content'], device=device, dtype=target_dtype)
            slot[-swm.content_dim:] = content_tensor.expand(swm.content_dim)
        return slot

    swm_info = {'content_dim': model.swm.content_dim, 'pointer_dim': model.swm.pointer_dim, 'type_dim': model.swm.type_dim, 'slot_dim': model.swm.slot_dim}
    expert = NeuralCpuExpert()
    expert.set_stage(num_digits)
    prompt_str, plan_phase, _, initial_pc = expert.compile(swm_info)
    
    model.swm.reset()
    messages = [{"role": "user", "content": prompt_str}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        current_input_ids = prompt_ids
        past_key_values = None
        for action in plan_phase:
            outputs = model(current_input_ids, past_key_values)
            addr_prob = F.one_hot(torch.tensor([action['addr']]), num_classes=model.swm.num_slots).to(device, dtype=target_dtype)
            model.swm.write_to_address(addr_prob, outputs['predicted_slot'])
            past_key_values = outputs['past_key_values']

        pc_init_spec = expert._create_slot_spec(pointers=[initial_pc])
        pc_init_slot = spec_to_tensor(pc_init_spec, model.swm).unsqueeze(0)
        model.swm.write_to_address(F.one_hot(torch.tensor([PC_SLOT_ADDR]), num_classes=model.swm.num_slots).to(device, dtype=target_dtype), pc_init_slot)

        halted = False
        exec_history = []
        max_exec_steps = num_digits * 5 + 10
        
        for step in range(max_exec_steps):
            outputs = model(prompt_ids, None)
            pc_logits = outputs['pc_logits']
            pc_probs = F.softmax(pc_logits, dim=-1)
            predicted_pc_addr = torch.argmax(pc_probs, dim=-1).item()
            
            exec_history.append(f"Step {step}: Model chose to execute instruction at address [{predicted_pc_addr}]")
            
            executed_op_slot = model.execute_step(pc_probs)
            
            op_content_hash = hash("HALT") % 100 / 100.0
            executed_op_content = model.swm.decode_slot(executed_op_slot)['content'][0, 0].item()
            if abs(executed_op_content - op_content_hash) < 1e-4:
                exec_history.append("HALT instruction detected. Execution finished.")
                halted = True
                break

    if not halted:
        exec_history.append("Execution failed: Max steps reached without HALT.")

    # ★★★ 修正: expert.operator_library_spec を使用 ★★★
    num_ops = len(expert.operator_library_spec)
    result_start_addr = 1 + num_ops + num_digits * 2
    
    model_answer_str = ""
    for i in range(num_digits + 1):
        addr_to_read = result_start_addr + i
        if addr_to_read < model.swm.num_slots:
            slot_content = model.swm.decode_slot(model.swm.memory[0, addr_to_read])['content'][0, 0]
            # 正規化を元に戻す
            model_answer_str += str(int(round(slot_content.item() * 10)))
    
    model_answer_str = model_answer_str[::-1].lstrip('0') or '0'
    model_answer = int(model_answer_str)

    nums = [int(s) for s in re.findall(r'\d+', prompt_str)]
    correct_answer = nums[0] + nums[1]

    is_correct = (model_answer == correct_answer)

    if verbose or not is_correct:
        print("-" * 50)
        print(f"Problem: {prompt_str}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Model's Answer:   {model_answer} -> {'CORRECT' if is_correct else 'WRONG'}")
        if verbose:
            print("\nExecution Trace:")
            for line in exec_history:
                print(line)
        print("-" * 50)

    return is_correct

def main(num_trials: int = 100, digits_to_test: list = [5, 10, 20, 22]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model, tokenizer = load_model_for_evaluation(CHECKPOINT_DIR, device)
    except FileNotFoundError as e:
        print(e)
        return

    for num_digits in digits_to_test:
        print(f"\n===== Testing {num_digits}-digit addition ({num_trials} trials) =====")
        
        correct_count = 0
        
        for i in tqdm(range(num_trials), desc=f"Testing {num_digits} digits"):
            verbose_log = (i < 3)
            if run_inference(model, tokenizer, num_digits=num_digits, verbose=verbose_log):
                correct_count += 1
        
        accuracy = (correct_count / num_trials) * 100
        print(f"===== Result for {num_digits}-digit addition =====")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{num_trials})")
        print("=" * 45)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the Neural CPU Adder model.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials for each digit length.")
    parser.add_argument("--digits", type=str, default="5,10,20,22", help="Comma-separated list of digit lengths to test (e.g., '5,10,20,22').")
    
    args = parser.parse_args()
    digits_list = [int(d.strip()) for d in args.digits.split(',')]
    
    main(num_trials=args.trials, digits_to_test=digits_list)