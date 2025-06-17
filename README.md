# LlamaCPU: A Neuro-Symbolic Architecture with a Large Language Model as a Differentiable CPU

  <!-- TODO: Replace with an actual architecture diagram URL -->

## Overview

**LlamaCPU** is a novel neuro-symbolic architecture that re-imagines a Large Language Model (LLM) not just as a task-solver, but as the central processing unit (CPU) of a fully differentiable computer. Inspired by the von Neumann architecture, this project separates the system into a neural CPU (Llama 3), a differentiable RAM (a structured external memory), and a neural ALU (an operator execution unit), all trainable end-to-end.

The core idea is to teach the LLM to perform complex, multi-step algorithmic tasks by compiling high-level instructions into a low-level program in memory, and then executing it step-by-step.

![Image](https://github.com/user-attachments/assets/7ff44ed7-7546-4bf9-9066-37fdebf409cd)

## Key Features

- **LLM as a Differentiable CPU**: Leverages the vast pre-trained knowledge and reasoning capabilities of Llama 3 to control the entire computational process.
- **Differentiable RAM with Pointers**: Implements a `HybridSWM` (Hybrid Slot-based Working Memory) that acts as a structured, addressable memory. Crucially, it supports pointers, enabling the creation of complex data structures and indirect addressing.
- **Two-Phase "Plan & Execute" Process**:
    1.  **Plan (Compile) Phase**: The LLM acts as a "compiler," translating a high-level task (e.g., "calculate 123 + 456") into a sequence of low-level instructions and data representations within the SWM.
    2.  **Execute Phase**: The LLM switches its role to a "processor," sequentially fetching and executing instructions from memory, guided by a program counter.
- **End-to-End Differentiability**: Unlike tool-using LLMs that call external, non-differentiable APIs, every component in LlamaCPU is differentiable. Gradients flow through the entire execution trace, allowing the model to learn *how* to perform computations, not just what the final answer is.
- **Curriculum Learning**: The model learns progressively, starting with simple tasks (e.g., single-digit addition) and gradually moving to more complex ones, ensuring stable and robust learning.

## Architecture

The LlamaCPU system consists of three main components:

1.  **Llama 3 Model (`Llama3_NSW_OEU`)**: The core controller.
    - In the **Plan Phase**, it synthesizes entire memory slots (containing types, pointers, and content) to set up the program and data.
    - In the **Execute Phase**, it predicts the next instruction to execute based on the current program counter.

2.  **HybridSWM (Differentiable RAM)**: A slot-based memory module.
    - Each slot is a vector with dedicated parts for a type, multiple pointers, and content.
    - Pointers are represented as keys that can attend over all memory slot addresses, enabling differentiable read and write operations.

3.  **Operator Execution Unit (OEU)**: A small, specialized neural network that acts as a Neural ALU.
    - It takes an operator slot and its argument slots (read from memory via pointers) as input.
    - It performs the actual computation (e.g., addition) and produces a result to be written back to memory.

![Image](https://github.com/user-attachments/assets/f0787e0a-c45e-4276-ad5b-9ab586664fb9)

## Current Status

The model is currently being trained on multi-digit addition using a curriculum learning approach. The logs show that the architecture is functioning correctly and the loss is steadily decreasing, demonstrating its ability to learn algorithmic procedures.

## Setup and Usage

1.  **Prerequisites**:
    - Python 3.10+
    - PyTorch
    - Transformers, PEFT, BitsAndBytes, Accelerate

    Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Hugging Face Access**:
    Ensure you are logged into your Hugging Face account and have access to the `meta-llama/Meta-Llama-3-8B-Instruct` model.
    ```bash
    huggingface-cli login
    ```

3.  **Training**:
    To start or resume training, simply run:
    ```bash
    python main.py
    ```
    The script will automatically find the latest checkpoint in the `checkpoints_final_adder/` directory and resume training from the last attempted stage. If no checkpoint is found, it will start from Stage 0 (1-digit addition).

## Future Work

- [ ] **Expand Algorithmic Capabilities**: Train the model on more complex algorithms like sorting (e.g., Bubble Sort), searching, and basic string manipulation.
- [ ] **Enhance the OEU**: Develop more sophisticated OEUs capable of handling a wider range of logical and arithmetic operations.
- [ ] **Autonomous Memory Management**: Explore methods for the LLM to dynamically allocate and de-allocate memory slots as needed.
- [ ] **Generalization**: Test the model's ability to generalize to tasks and problem sizes it has not seen during training.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
