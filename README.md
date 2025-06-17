# LlamaCPU
An LLM-driven Neuro-Symbolic Processor

llamaCPU: A Neuro-Symbolic Architecture with a Differentiable LLM-based CPU

![alt text](https://img.shields.io/badge/build-passing-brightgreen)


![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)


![alt text](https://img.shields.io/badge/python-3.10+-blue.svg)

llamaCPU is an experimental neuro-symbolic architecture that re-imagines a Large Language Model (Llama 3) as a general-purpose, differentiable Central Processing Unit (CPU). It learns to execute algorithmic tasks by manipulating a structured, external memory, drawing a direct analogy to the von Neumann architecture.

The core idea is to move beyond using LLMs as monolithic problem-solvers and instead leverage them as the core component of a learnable, structured reasoning system.

Core Concept

The system is designed as a differentiable counterpart to a classic computer:

LLM as a Neural CPU: The Large Language Model (in this case, meta-llama/Meta-Llama-3-8B-Instruct) acts as the central controller. It interprets high-level instructions and drives the step-by-step execution of a program.

HybridSWM as Differentiable RAM: A HybridSWM (Hybrid Short-term Working Memory) module serves as an external, differentiable memory space. It's not just a key-value store; its slots are structured to hold types, content, and pointers, enabling the creation of complex data structures.

OEU as a Neural ALU: An OperatorExecutionUnit (OEU) functions as a learnable Arithmetic Logic Unit. Instead of using fixed operations (+, -), it's a neural module that learns to perform computations based on operator and operand embeddings.

Architecture

The system operates in a two-phase process: PLAN and EXEC.

Generated code
User Prompt ("Calculate 18 + 7")
           │
           ▼
┌──────────────────────┐
│     LLM (as CPU)     │
└──────────────────────┘
           │
           ├───► Phase 1: PLAN (Compile)
           │     The LLM acts as a "compiler," translating the user prompt
           │     into a low-level program and data representation within the SWM.
           │
           │          ┌──────────────────────────────────┐
           │          │      HybridSWM (as RAM)          │
           │          │ ┌───────┐      ┌───────┐         │
           │          │ │ Slot  │ ...  │ Slot  │ ...     │
           │          │ │ type  │      │ type  │         │
           │          │ │ ptrs  │      │ ptrs  │         │
           │          │ │ data  │      │ data  │         │
           │          │ └───────┘      └───────┘         │
           │          └──────────────────────────────────┘
           │
           └───► Phase 2: EXEC (Execute)
                 The LLM acts as a "processor," sequentially executing
                 the program stored in the SWM.

                 For each step:
                   1. Read Program Counter (PC) from SWM.
                   2. Fetch instruction from SWM using PC.
                   3. Read operands from SWM using pointers in the instruction.
                   4. Pass operator & operands to OEU (ALU) for computation.
                   5. Write result back to SWM.
                   6. Increment PC.

How It Works

PLAN Phase (Compilation): Given a high-level task like "Autonomously calculate 25 + 42", the LLM first populates the HybridSWM. It creates slots for the input numbers (2, 5, 4, 2), the result placeholders, constants (like 10), and the actual program instructions (a sequence of pointers representing ADD, COPY, etc.). This entire memory layout is the "compiled" program.

EXEC Phase (Execution): The LLM's role shifts. It now focuses solely on driving the execution. In a loop, it:
a. Predicts the current program counter (PC) address.
b. Reads the instruction from that address in the HybridSWM.
c. The instruction contains pointers to its arguments (e.g., a pointer to the digit '5' and a pointer to the 'carry' bit).
d. These arguments are passed to the OperatorExecutionUnit (OEU), which computes a result.
e. The result is written back to the memory location specified by the instruction's output pointer.
f. The PC is incremented, and the loop continues until a HALT instruction is reached.

The entire process, from prediction to memory access to computation, is end-to-end differentiable, allowing the system to learn not just what to do, but how to perform the computation itself.

Key Features

Differentiable von Neumann Architecture: A novel implementation of the classic CPU-RAM model within a neural framework.

Two-Phase Computation: A clean separation between high-level task planning (PLAN) and low-level algorithmic execution (EXEC), which stabilizes learning.

Structured, Differentiable Memory: The HybridSWM supports pointers, allowing for the creation and manipulation of data structures, a critical component for symbolic reasoning.

Learned Neural Operators: The OEU learns to perform semantic operations, moving beyond fixed, hard-coded functions.

Curriculum Learning: The model is trained on a curriculum, starting with single-digit addition and progressively increasing the difficulty. This is essential for mastering complex, multi-step algorithms.

Novelty & Comparison to Existing Work

vs. DNC/NTM: While inspired by differentiable memory, llamaCPU uses a modern, pre-trained LLM as its controller, endowing the system with vast world knowledge and superior instruction-following capabilities from the start.

vs. Tool-formers/ReAct: These models call external, non-differentiable tools (like a calculator API). In llamaCPU, the memory and ALU are internal, differentiable components. The gradient flows through the computation process itself, allowing the model to learn how to calculate, not just when to call a black-box tool.

vs. LLM Code Generation: Code-generating models output symbolic code that is executed by a separate, deterministic interpreter. llamaCPU is the interpreter. The entire execution trace happens within the neural network, making the process more flexible and learnable.

Getting Started
Prerequisites

Python 3.10+

PyTorch

Transformers, PEFT, bitsandbytes

An environment with a CUDA-enabled GPU is highly recommended.

Installation

Clone the repository:

Generated bash
git clone https://github.com/your-username/llamaCPU.git
cd llamaCPU
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install the required packages:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(You will need to create a requirements.txt file with packages like torch, transformers, peft, bitsandbytes, accelerate)

Configuration

Key parameters can be adjusted at the top of main.py:

MODEL_ID: The Hugging Face model to use as the CPU.

CHECKPOINT_DIR: Directory to save and load training checkpoints.

config: A dictionary containing hyperparameters like learning rate, loss weights, and SWM dimensions.

Training

To start or resume training, simply run:

Generated bash
python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The script will automatically find the latest checkpoint and resume from the appropriate stage of the curriculum.

Roadmap & Future Work

Complex Algorithms: Expand the curriculum to include tasks like sorting, searching, and basic string manipulation.

Control Flow: Implement learnable JUMP and IF/THEN instructions to enable loops and conditional logic.

Richer Data Structures: Explicitly train the model to build and traverse linked lists, trees, or graphs within the SWM.

Exploring Different Backbones: Test the architecture with other capable LLMs.

License

This project is licensed under the MIT License. See the LICENSE file for details.
