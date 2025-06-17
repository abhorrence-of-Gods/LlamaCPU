import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from swm import HybridSWM

PC_SLOT_ADDR = 0

class OperatorExecutionUnit(nn.Module):
    def __init__(self, slot_dim, d_model, num_heads=4):
        super().__init__()
        self.proj_operator = nn.Linear(slot_dim, d_model)
        self.proj_args = nn.Linear(slot_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, slot_dim)
        )

    def forward(self, operator_slot, arg_slots):
        op_query = self.proj_operator(operator_slot).unsqueeze(1)
        if not arg_slots:
            return torch.zeros_like(operator_slot)
        arg_kv = self.proj_args(torch.stack(arg_slots, dim=1))
        
        attended_context, _ = self.cross_attention(op_query, arg_kv, arg_kv)
        update_value = self.ffn(attended_context.squeeze(1))
        return update_value

class Llama3_NSW_OEU(nn.Module):
    def __init__(self, config: PretrainedConfig, swm: HybridSWM, base_model: PreTrainedModel):
        super().__init__()
        self.swm = swm
        self.d_model = config.hidden_size
        self.llama = base_model
        
        # ★★★ 修正: lm_headの取得ロジックを完全に削除します。
        # forwardメソッドでモデル出力から直接logitsを取得するため、この属性は不要です。
        
        self.slot_synthesizer = nn.Sequential(
            nn.Linear(self.d_model, swm.slot_dim),
            nn.Tanh()
        )
        
        self.pc_controller = nn.Linear(self.d_model, self.swm.num_slots)
        self.oeu = OperatorExecutionUnit(swm.slot_dim, self.d_model)

    def forward(self, input_ids, past_key_values=None, attention_mask=None):
        # LLMのforwardパスを実行
        gpt_outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            use_cache=True,
            output_hidden_states=True
        )
        
        # ★★★ 修正: gpt_outputsから直接logitsを取得します。
        # これにより、PEFT化されているかどうかにかかわらず、常に正しいロジットが得られ、
        # 非効率な再計算や属性エラーを回避できます。
        token_logits = gpt_outputs.logits
        
        # 最後の隠れ層の状態を取得（これは他のヘッドで利用するため必要）
        last_hidden_state = gpt_outputs.hidden_states[-1]
        h_t = last_hidden_state[:, -1, :].contiguous()

        # 他のカスタムヘッドの計算
        predicted_slot_vec = self.slot_synthesizer(h_t)
        pc_logits = self.pc_controller(h_t)

        return {
            "token_logits": token_logits,
            "predicted_slot": predicted_slot_vec,
            "pc_logits": pc_logits,
            "past_key_values": gpt_outputs.past_key_values,
        }
        
    def execute_step(self, pc_address_probs):
        # (このメソッドは変更なし)
        current_pc_addr = torch.argmax(pc_address_probs, dim=-1).item()

        operator_slot = torch.bmm(pc_address_probs.unsqueeze(1), self.swm.memory).squeeze(1)
        op_parts = self.swm.decode_slot(operator_slot)
        
        arg_slots = []
        for i in range(2): 
            if op_parts['pointers'][i].abs().sum() > 1e-6:
                arg_slot, _ = self.swm.read_from_pointer(op_parts['pointers'][i])
                arg_slots.append(arg_slot)
            
        if arg_slots:
            update_value = self.oeu(operator_slot, arg_slots)
            write_pointer = op_parts['pointers'][2]
            if write_pointer.abs().sum() > 1e-6:
                write_addr_scores = torch.matmul(write_pointer, self.swm.keys.T)
                write_addr_probs = F.softmax(write_addr_scores / (self.swm.pointer_dim**0.5), dim=-1)
                self.swm.write_to_address(write_addr_probs, update_value)

        next_pc_addr = current_pc_addr + 1
        if next_pc_addr >= self.swm.num_slots:
            next_pc_addr = self.swm.num_slots - 1 

        next_pc_pointer = self.swm.keys[next_pc_addr].unsqueeze(0)
        
        pc_slot_content = torch.zeros(1, self.swm.slot_dim, device=self.swm.memory.device, dtype=self.swm.memory.dtype)
        pc_slot_content[0, self.swm.type_dim : self.swm.type_dim + self.swm.pointer_dim] = next_pc_pointer

        pc_addr_prob = F.one_hot(torch.tensor([PC_SLOT_ADDR]), num_classes=self.swm.num_slots).to(self.swm.memory.device, dtype=self.swm.memory.dtype)
        self.swm.write_to_address(pc_addr_prob, pc_slot_content)

        return operator_slot