#!/usr/bin/env python3
import sys
import time
import numpy as np
import omni_bindings
from transformers import AutoTokenizer

IM_END = "<|im_end|>"

class OmniChat:
    def __init__(self, model_path: str, max_seq_len: int = 2048):
        print("Loading model...")
        self.runtime = omni_bindings.OmniRuntime(model_path, max_seq_len=max_seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Base")
        self.conversation_history = []
        self.max_history = 5
        print("Model loaded!")
    
    def generate_response(self, user_input: str, max_tokens: int = 256, temperature: float = 0.3):
        try:
            self.conversation_history.append({"role": "user", "content": user_input})
            
            tokens = self.tokenizer.apply_chat_template(
                self.conversation_history,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )[0].tolist()
            
            if len(tokens) > 1500:
                print("Warning: Conversation too long, clearing history...")
                self.conversation_history = [{"role": "user", "content": user_input}]
                tokens = self.tokenizer.apply_chat_template(
                    self.conversation_history,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True,
                )[0].tolist()
            
            print(f"[{len(tokens)} tokens]", end=" ", flush=True)
            self.runtime.reset_kv_cache()
            
            start = time.time()
            output_tokens = self._generate_with_stopping(tokens, max_tokens, temperature)
            elapsed = time.time() - start
            
            response_tokens = output_tokens[len(tokens):]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            self.conversation_history.append({"role": "assistant", "content": response})
            
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            
            return response, elapsed, len(response_tokens)
            
        except Exception as e:
            print(f"Error: {e}")
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return "", 0, 0
    
    def _generate_with_stopping(self, prompt_tokens, max_tokens, temperature):
        eos_id = self.tokenizer.eos_token_id or 2
        im_end_id = self.tokenizer.convert_tokens_to_ids(IM_END)
        
        logits = self.runtime.forward(prompt_tokens)
        output_tokens = list(prompt_tokens)
        
        for _ in range(max_tokens):
            next_token = self._sample_with_min_p(logits, temperature)
            output_tokens.append(next_token)
            
            if next_token == eos_id or next_token == im_end_id:
                break
            
            logits = self.runtime.forward([next_token])
        
        return output_tokens
    
    def _sample_with_min_p(self, logits, temperature=0.3, min_p=0.15):
        if temperature < 1e-6:
            return int(np.argmax(logits))
        
        logits = logits.astype(np.float64) / temperature
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        max_prob = np.max(probs)
        threshold = min_p * max_prob
        mask = probs >= threshold
        
        if not np.any(mask):
            return int(np.argmax(probs))
        
        filtered_probs = probs * mask
        filtered_probs = filtered_probs / np.sum(filtered_probs)
        
        return int(np.random.choice(len(filtered_probs), p=filtered_probs))
    
    def clear_history(self):
        self.conversation_history = []
        print("History cleared.")
    
    def close(self):
        self.runtime.close()


def main():
    print("=" * 60)
    print("Omni Core - Interactive Chatbot")
    print("=" * 60)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../lfm2_new.omni"
    
    try:
        chat = OmniChat(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print("Type your message and press Enter. Commands: /clear, /quit\n")
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/quit", "/exit"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                chat.clear_history()
                continue
            
            response, elapsed, n_tokens = chat.generate_response(user_input)
            if response:
                speed = n_tokens / elapsed if elapsed > 0 else 0
                print(f"\nAssistant: {response}")
                print(f"[{n_tokens} tokens, {elapsed:.1f}s, {speed:.1f} tok/s]\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        chat.close()


if __name__ == "__main__":
    main()
