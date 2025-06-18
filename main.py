import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from collections import deque
import random
import os
import re
import glob
import gc

from peft import get_peft_model, PeftModel, LoraConfig, TaskType

from swm import HybridSWM
from model import Llama3_NSW_OEU

try: import bitsandbytes.optim as bnb_optim
except ImportError: raise ImportError("Please install bitsandbytes: pip install bitsandbytes")

CHECKPOINT_DIR = "checkpoints_final_adder"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
PC_SLOT_ADDR = 0

#======================================================================
# ▼▼▼ チェックポイント管理機能 ▼▼▼
#======================================================================
def save_checkpoint(model, optimizer, scheduler, stage_idx, success):
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    filename = f"stage_{stage_idx}.pt"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    lora_dir = os.path.join(CHECKPOINT_DIR, f"lora_stage_{stage_idx}")
    model.llama.save_pretrained(lora_dir)
    
    head_state_dict = {k: v for k, v in model.state_dict().items() if "llama" not in k}
    state = {
        'stage_idx': stage_idx,
        'head_state_dict': head_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'success': success,
    }
    torch.save(state, filepath)
    status_str = "Completed" if success else "Attempted (In-Progress)"
    print(f"\n[Checkpoint] Saved Stage {stage_idx} ({status_str}) to {filepath}")

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR): return None, None
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "stage_*.pt"))
    if not checkpoints: return None, None
    
    latest_checkpoint_path = max(checkpoints, key=lambda p: int(re.search(r'stage_(\d+)', os.path.basename(p)).group(1)))
    
    print(f"[Checkpoint] Found latest checkpoint at {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    return latest_checkpoint_path, checkpoint

#======================================================================
# ▼▼▼ ニューラルCPU向け専門家コンパイラ ▼▼▼
#======================================================================
class NeuralCpuExpert:
    def __init__(self):
        self.stage = 1
        self.operator_library_spec = {}
    def set_stage(self, stage: int): self.stage = stage
    def _create_slot_spec(self, type_val=0.0, pointers=[], content=None):
        return {'type_val': type_val, 'pointers': pointers, 'content': content}
    def compile(self, swm_info):
        num_digits = self.stage
        a, b = random.randint(0, 10**num_digits - 1), random.randint(0, 10**num_digits - 1)
        a_str, b_str = str(a).zfill(num_digits), str(b).zfill(num_digits)
        plan_phase, exec_trace = [], []
        next_free_slot = 1
        def allocate(n=1):
            nonlocal next_free_slot
            start = next_free_slot; next_free_slot += n
            return list(range(start, start + n))
        op_addrs = {}
        op_names = ["ADD", "MOD", "DIV", "COPY", "HALT"]
        for op_name in op_names:
            if op_name not in self.operator_library_spec:
                addr = allocate()[0]
                op_content = [hash(op_name) % 100 / 100.0] * swm_info['content_dim']
                self.operator_library_spec[op_name] = (addr, self._create_slot_spec(type_val=1.0, content=op_content))
            op_addrs[op_name] = self.operator_library_spec[op_name][0]
            plan_phase.append({'addr': op_addrs[op_name], 'slot_spec': self.operator_library_spec[op_name][1]})
        addrs_a, addrs_b = allocate(num_digits), allocate(num_digits)
        addrs_result = allocate(num_digits + 1)
        addr_carry, const_ten = allocate()[0], allocate()[0]
        temp_slot_1, temp_slot_2 = allocate()[0], allocate()[0]
        for i, d in enumerate(reversed(a_str)): plan_phase.append({'addr': addrs_a[i], 'slot_spec': self._create_slot_spec(content=[int(d)])})
        for i, d in enumerate(reversed(b_str)): plan_phase.append({'addr': addrs_b[i], 'slot_spec': self._create_slot_spec(content=[int(d)])})
        for addr in addrs_result: plan_phase.append({'addr': addr, 'slot_spec': self._create_slot_spec(content=[0])})
        plan_phase.append({'addr': addr_carry, 'slot_spec': self._create_slot_spec(content=[0])})
        plan_phase.append({'addr': const_ten, 'slot_spec': self._create_slot_spec(content=[10])})
        plan_phase.append({'addr': temp_slot_1, 'slot_spec': self._create_slot_spec(content=[0])})
        plan_phase.append({'addr': temp_slot_2, 'slot_spec': self._create_slot_spec(content=[0])})
        prog_start_addr = next_free_slot
        for i in range(num_digits):
            plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[addrs_a[i], addr_carry, temp_slot_1])})
            plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[temp_slot_1, addrs_b[i], temp_slot_2])})
            plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[temp_slot_2, const_ten, addrs_result[i]])})
            plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[temp_slot_2, const_ten, addr_carry])})
        plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[addr_carry, None, addrs_result[num_digits]])})
        plan_phase.append({'addr': allocate()[0], 'slot_spec': self._create_slot_spec(type_val=1.0, pointers=[])})
        for i in range(num_digits):
            exec_trace.extend([
                {'pc_state': prog_start_addr + i*4 + 0}, {'pc_state': prog_start_addr + i*4 + 1},
                {'pc_state': prog_start_addr + i*4 + 2}, {'pc_state': prog_start_addr + i*4 + 3},
            ])
        exec_trace.append({'pc_state': prog_start_addr + num_digits*4})
        exec_trace.append({'pc_state': prog_start_addr + num_digits*4 + 1})
        prompt = f"Autonomously calculate {a_str} + {b_str}"
        return prompt, plan_phase, exec_trace, prog_start_addr

#======================================================================
# ▼▼▼ 学習ループ ▼▼▼
#======================================================================
def train_stage(model, tokenizer, optimizer, scheduler, device, config):
    swm_info = {'content_dim': model.swm.content_dim, 'pointer_dim': model.swm.pointer_dim, 'type_dim': model.swm.type_dim, 'slot_dim': model.swm.slot_dim}
    expert = NeuralCpuExpert()
    expert.set_stage(config['current_stage'])
    
    loss_fct_mse = nn.MSELoss().to(device)
    loss_fct_pc = nn.CrossEntropyLoss().to(device)

    model.train()
    recent_losses = deque(maxlen=config['check_interval'])
    target_dtype = torch.bfloat16

    def spec_to_tensor(spec, swm, for_loss=False):
        slot = torch.zeros(swm.slot_dim, device=device, dtype=target_dtype)
        slot[0] = spec['type_val']
        ptr_start_idx = swm.type_dim
        for i, ptr_addr in enumerate(spec['pointers']):
            if ptr_addr is None: continue
            ptr_key = swm.keys[ptr_addr]
            if for_loss: ptr_key = ptr_key.detach()
            slot[ptr_start_idx + i*swm.pointer_dim : ptr_start_idx + (i+1)*swm.pointer_dim] = ptr_key
        if spec['content'] is not None:
            normalized_content = [c / config['content_norm_factor'] for c in spec['content']]
            content_tensor = torch.tensor(normalized_content, device=device, dtype=target_dtype)
            slot[-swm.content_dim:] = content_tensor.expand(swm.content_dim)
        return slot
        
    for epoch in range(1, config['max_epochs_per_stage'] + 1):
        optimizer.zero_grad()
        prompt_str, plan_phase, exec_trace, initial_pc = expert.compile(swm_info)
        messages = [{"role": "user", "content": prompt_str}]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        model.swm.reset()
        
        total_plan_loss_type = 0
        total_plan_loss_pointers = 0
        total_plan_loss_content = 0
        total_pc_loss = 0
        
        # --- PLANフェーズ ---
        outputs = model(prompt_ids, past_key_values=None)
        
        for action in plan_phase:
            target_slot = spec_to_tensor(action['slot_spec'], model.swm, for_loss=True).unsqueeze(0)
            
            predicted_parts = model.swm.decode_slot(outputs['predicted_slot'])
            target_parts = model.swm.decode_slot(target_slot)

            loss_type = loss_fct_mse(predicted_parts['type'], target_parts['type'])
            total_plan_loss_type += loss_type

            loss_pointers = 0
            for pred_ptr, target_ptr in zip(predicted_parts['pointers'], target_parts['pointers']):
                loss_pointers += loss_fct_mse(pred_ptr, target_ptr)
            total_plan_loss_pointers += loss_pointers / len(predicted_parts['pointers'])

            loss_content = loss_fct_mse(predicted_parts['content'], target_parts['content'])
            total_plan_loss_content += loss_content
            
            with torch.no_grad():
                addr_idx_gpu = torch.tensor([action['addr']], device=device, dtype=torch.long)
                addr_prob = F.one_hot(addr_idx_gpu, num_classes=model.swm.num_slots).to(dtype=target_dtype)
                slot_for_write = spec_to_tensor(action['slot_spec'], model.swm).unsqueeze(0)
                model.swm.write_to_address(addr_prob, slot_for_write)

        # --- EXECフェーズ ---
        progress_ratio = min(epoch / config['annealing_epochs'], 1.0)
        with torch.no_grad():
            pc_init_spec = expert._create_slot_spec(pointers=[initial_pc])
            pc_init_slot = spec_to_tensor(pc_init_spec, model.swm).unsqueeze(0)
            model.swm.write_to_address(F.one_hot(torch.tensor([PC_SLOT_ADDR]), num_classes=model.swm.num_slots).to(device, dtype=target_dtype), pc_init_slot)
        
        current_exec_input_ids = prompt_ids
        past_key_values_exec = None
        for step in range(len(exec_trace)):
            trace = exec_trace[step]
            expert_pc_addr = torch.tensor([trace['pc_state']], device=device)
            outputs = model(current_exec_input_ids, past_key_values=past_key_values_exec)
            total_pc_loss += loss_fct_pc(outputs['pc_logits'], expert_pc_addr)
            past_key_values_exec = outputs['past_key_values']
            current_exec_input_ids = prompt_ids[:, -1:]
            use_self_regression = random.random() < progress_ratio
            pc_probs = F.softmax(outputs['pc_logits'], dim=-1) if use_self_regression else F.one_hot(expert_pc_addr, num_classes=model.swm.num_slots).to(dtype=target_dtype)
            with torch.no_grad(): model.execute_step(pc_probs)
        
        weighted_plan_loss = (config['plan_loss_weight_type'] * total_plan_loss_type) + \
                             (config['plan_loss_weight_pointers'] * total_plan_loss_pointers) + \
                             (config['plan_loss_weight_content'] * total_plan_loss_content)
        
        weighted_loss = weighted_plan_loss + (config['pc_loss_weight'] * total_pc_loss)
        weighted_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        unweighted_plan_loss = total_plan_loss_type.item() + total_plan_loss_pointers.item() + total_plan_loss_content.item()
        unweighted_total_loss = unweighted_plan_loss + total_pc_loss.item()
        recent_losses.append(unweighted_total_loss)

        if epoch % config['log_interval'] == 0:
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            print(f"  Stage {expert.stage} | Epoch {epoch}/{config['max_epochs_per_stage']} | Avg Loss: {avg_loss:.4f} "
                  f"[Plan(T/P/C): {total_plan_loss_type.item():.4f}/{total_plan_loss_pointers.item():.4f}/{total_plan_loss_content.item():.4f}, PC: {total_pc_loss.item():.4f}] | SR Ratio: {progress_ratio:.2f}")

        if epoch % config['check_interval'] == 0 and recent_losses:
            if sum(recent_losses) / len(recent_losses) < config['loss_threshold']:
                print(f"  Competence reached for Stage {expert.stage}. Advancing...")
                return True
        
        gc.collect(); torch.cuda.empty_cache()
    
    print(f"  Max epochs reached for Stage {expert.stage}. Competence not reached.")
    return False

#======================================================================
# ▼▼▼ メイン実行ブロック ▼▼▼
#======================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.bfloat16
    config = {
        'lr': 2e-5, 'max_epochs_per_stage': 10000, 'log_interval': 100,
        'check_interval': 1000, 'loss_threshold': 0.05,
        'annealing_epochs': 20000,
        'swm_slots': 256, 'swm_slot_dim': 512, 'swm_ptr_dim': 64, 'swm_num_ptr': 4,
        'max_digits': 20,
        'content_norm_factor': 10.0,
        
        'plan_loss_weight_type': 1.0,
        'plan_loss_weight_pointers': 10.0,
        'plan_loss_weight_content': 5.0,
        'pc_loss_weight': 1.0,
    }

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=target_dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="cpu", torch_dtype=target_dtype)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    swm = HybridSWM(batch_size=1, num_slots=config['swm_slots'], slot_dim=config['swm_slot_dim'], 
                    pointer_dim=config['swm_ptr_dim'], num_pointers=config['swm_num_ptr'], dtype=target_dtype)
    
    model = Llama3_NSW_OEU(base_model.config, swm, base_model=base_model)
    
    start_stage_idx = 0
    latest_ckpt_path, checkpoint = find_latest_checkpoint()
    
    if checkpoint:
        last_attempted_stage = checkpoint.get('stage_idx', -1)
        lora_path = os.path.join(CHECKPOINT_DIR, f"lora_stage_{last_attempted_stage}")
        print(f"[Checkpoint] Loading LoRA weights from {lora_path}")
        peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
        model.llama = peft_model
        print("[Checkpoint] PEFT model loaded from checkpoint.")
    else:
        print("[System] No checkpoint found. Initializing new PEFT model.")
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)
        model.llama = peft_model

    model.llama.print_trainable_parameters()

    optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=config['lr'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=config['max_epochs_per_stage'])

    if checkpoint:
        last_attempted_stage = checkpoint.get('stage_idx', -1)
        was_successful = checkpoint.get('success', False)
        
        model.load_state_dict(checkpoint['head_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[Checkpoint] Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("[Checkpoint] Scheduler state loaded.")
        
        if was_successful:
            start_stage_idx = last_attempted_stage + 1
            print(f"[Checkpoint] Stage {last_attempted_stage} was successful. Resuming from new Stage {start_stage_idx}.")
        else:
            start_stage_idx = last_attempted_stage
            print(f"[Checkpoint] Stage {last_attempted_stage} was not completed. Retrying Stage {start_stage_idx}.")

    model.to(device, dtype=target_dtype)
    print(f"Model moved to {device} with dtype {target_dtype}.")
    
    for stage_idx in range(start_stage_idx, config['max_digits']):
        current_digits = stage_idx + 1
        print(f"\n{'='*30}\n{'='*7} ATTEMPTING STAGE {stage_idx} ({current_digits} Digits) {'='*7}\n{'='*30}\n")
        
        stage_config = config.copy()
        stage_config['current_stage'] = current_digits
        
        success = train_stage(model, tokenizer, optimizer, scheduler, device, stage_config)
        
        save_checkpoint(model, optimizer, scheduler, stage_idx, success)
        
        if not success: 
            print(f"\nTraining stopped at Stage {stage_idx} because it did not reach competence within the epoch limit.")
            print("A checkpoint has been saved. You can re-run the script to resume training from this stage.")
            break
    
    print(f"\nFull curriculum finished or stopped.")

if __name__ == "__main__":
    main()
