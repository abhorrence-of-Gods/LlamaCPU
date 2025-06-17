# swm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSWM(nn.Module):
    """
    ユニバーサル・ハイブリッドSWM。データ、プログラム（スキル）、状態変数を統一的に格納する。
    """
    # ★★★ 修正: __init__メソッドのシグネチャに dtype を追加 ★★★
    def __init__(self, batch_size, num_slots, slot_dim, num_pointers=4, pointer_dim=32, dtype=torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        self.type_dim = 1
        self.num_pointers = num_pointers
        self.pointer_dim = pointer_dim
        
        self.content_dim = self.slot_dim - self.type_dim - (self.num_pointers * self.pointer_dim)
        if self.content_dim <= 0:
            raise ValueError("slot_dim is too small for the defined structure.")

        # ★★★ 修正: 受け取ったdtypeを使ってバッファを初期化 ★★★
        self.register_buffer('memory', torch.zeros(batch_size, num_slots, slot_dim, dtype=dtype))

        # key_projも同じデータ型に変換
        self.key_proj = nn.Linear(self.num_slots, self.pointer_dim, bias=False).to(dtype=dtype)
        # eyeで生成するテンソルも同じデータ型に
        identity_matrix = torch.eye(self.num_slots, dtype=dtype)
        self.register_buffer('keys', self.key_proj(identity_matrix))

    def reset(self, batch_size=None):
        if batch_size is not None: self.batch_size = batch_size
        self.memory.zero_()

    def decode_slot(self, slot_vector):
        parts = {}
        batch_dim = slot_vector.shape[0] if slot_vector.dim() > 1 else 1
        
        current_idx = 0
        parts['type'] = slot_vector.view(batch_dim, -1)[:, current_idx:current_idx+self.type_dim]
        current_idx += self.type_dim
        parts['pointers'] = [
            slot_vector.view(batch_dim, -1)[:, current_idx + i*self.pointer_dim : current_idx + (i+1)*self.pointer_dim]
            for i in range(self.num_pointers)
        ]
        current_idx += self.num_pointers * self.pointer_dim
        parts['content'] = slot_vector.view(batch_dim, -1)[:, current_idx:]
        return parts

    def read_from_pointer(self, pointer_vec):
        attention_scores = torch.matmul(pointer_vec, self.keys.T)
        attention_probs = F.softmax(attention_scores / (self.pointer_dim**0.5), dim=-1)
        read_value = torch.bmm(attention_probs.unsqueeze(1), self.memory).squeeze(1)
        return read_value, attention_probs

    def write_to_address(self, addr_probs, value_to_write, gate=1.0):
        update_term = torch.bmm(addr_probs.unsqueeze(2), value_to_write.unsqueeze(1))
        if isinstance(gate, torch.Tensor):
            gate = gate.unsqueeze(1)
        
        forget_gate = 1.0 - (gate * addr_probs.unsqueeze(-1))
        self.memory = self.memory * forget_gate + update_term
        return self.memory