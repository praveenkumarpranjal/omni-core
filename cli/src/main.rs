mod ffi;

use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::time::Instant;
use tokenizers::Tokenizer;
use rand::Rng;

#[derive(Parser)]
#[command(name = "omni")]
#[command(about = "Fastest CPU inference engine for LLMs on Apple Silicon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat with the model
    Chat {
        /// Path to .omni model file
        #[arg(short, long, default_value = "../lfm2_new.omni")]
        model: String,
        /// Path to tokenizer (HuggingFace model ID or local path)
        #[arg(short, long, default_value = "LiquidAI/LFM2.5-1.2B-Base")]
        tokenizer: String,
        /// Maximum tokens to generate per response
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Temperature for sampling
        #[arg(long, default_value_t = 0.3)]
        temperature: f32,
        /// Min-p threshold for sampling
        #[arg(long, default_value_t = 0.15)]
        min_p: f32,
    },
    /// Run inference on a model
    Run {
        /// Path to .omni model file
        model: String,
        /// Input prompt (comma-separated token IDs)
        prompt: String,
        /// Maximum tokens to generate
        #[arg(short, long, default_value_t = 100)]
        max_tokens: usize,
        /// Temperature for sampling
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,
    },
    /// Show model information
    Info {
        /// Path to .omni model file
        model: String,
    },
    /// Benchmark model performance
    Bench {
        /// Path to .omni model file
        model: String,
        /// Number of tokens to generate
        #[arg(short, long, default_value_t = 100)]
        tokens: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat {
            model,
            tokenizer,
            max_tokens,
            temperature,
            min_p,
        } => {
            run_chat(&model, &tokenizer, max_tokens, temperature, min_p);
        }
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
        } => {
            run_inference(&model, &prompt, max_tokens, temperature);
        }
        Commands::Info { model } => {
            show_info(&model);
        }
        Commands::Bench { model, tokens } => {
            benchmark(&model, tokens);
        }
    }
}

fn run_chat(
    model_path: &str,
    tokenizer_path: &str,
    max_tokens: usize,
    temperature: f32,
    min_p: f32,
) {
    println!("============================================================");
    println!("Omni Core - Interactive Chatbot (Rust)");
    println!("============================================================");
    
    // Load tokenizer
    println!("\nLoading tokenizer from {}...", tokenizer_path);
    let tokenizer = match Tokenizer::from_file(tokenizer_path) {
        Ok(t) => t,
        Err(_) => {
            // Try loading from HuggingFace
            eprintln!("Note: Tokenizer must be a local file path (e.g., tokenizer.json)");
            eprintln!("Download from: https://huggingface.co/{}/resolve/main/tokenizer.json", tokenizer_path);
            std::process::exit(1);
        }
    };
    
    // Load model
    println!("Loading model from {}...", model_path);
    let model = match ffi::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };
    
    let ctx = match ffi::Context::new(&model, 2048, 128256) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error creating context: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("✓ Model loaded\n");
    println!("Type your message and press Enter. Commands: /clear, /quit\n");
    
    let mut conversation: Vec<(String, String)> = Vec::new();
    let max_history = 5;
    
    loop {
        // Get user input
        print!("You: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        // Handle commands
        if input.eq_ignore_ascii_case("/quit") || input.eq_ignore_ascii_case("/exit") {
            println!("Goodbye!");
            break;
        }
        
        if input.eq_ignore_ascii_case("/clear") {
            conversation.clear();
            println!("History cleared.\n");
            continue;
        }
        
        // Generate response
        match generate_response(
            &ctx,
            &tokenizer,
            &mut conversation,
            input,
            max_tokens,
            temperature,
            min_p,
            max_history,
        ) {
            Ok((response, elapsed, n_tokens)) => {
                let speed = n_tokens as f32 / elapsed;
                println!("\nAssistant: {}", response);
                println!("[{} tokens, {:.1}s, {:.1} tok/s]\n", n_tokens, elapsed, speed);
            }
            Err(e) => {
                eprintln!("Error generating response: {}", e);
                if !conversation.is_empty() && conversation.last().unwrap().0 == "user" {
                    conversation.pop();
                }
            }
        }
    }
}

fn generate_response(
    ctx: &ffi::Context,
    tokenizer: &Tokenizer,
    conversation: &mut Vec<(String, String)>,
    user_input: &str,
    max_tokens: usize,
    temperature: f32,
    min_p: f32,
    max_history: usize,
) -> Result<(String, f32, usize), String> {
    // Add user message to conversation
    conversation.push(("user".to_string(), user_input.to_string()));
    
    // Build chat template
    let mut prompt = String::new();
    for (role, content) in conversation.iter() {
        if role == "user" {
            prompt.push_str("<|startoftext|><|im_start|>user\n");
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
        } else {
            prompt.push_str("<|im_start|>assistant\n");
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    
    // Tokenize
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("Tokenization error: {}", e))?;
    
    let tokens: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    
    // Check length
    if tokens.len() > 1500 {
        println!("Warning: Conversation too long, clearing history...");
        conversation.clear();
        conversation.push(("user".to_string(), user_input.to_string()));
        return generate_response(ctx, tokenizer, conversation, user_input, max_tokens, temperature, min_p, max_history);
    }
    
    print!("[{} tokens] ", tokens.len());
    io::stdout().flush().unwrap();
    
    // Reset KV cache and run inference
    ctx.reset_kv_cache();
    
    let start = Instant::now();
    let output_tokens = generate_with_stopping(ctx, tokenizer, &tokens, max_tokens, temperature, min_p)?;
    let elapsed = start.elapsed().as_secs_f32();
    
    // Decode response
    let response_tokens: Vec<u32> = output_tokens[tokens.len()..]
        .iter()
        .map(|&t| t as u32)
        .collect();
    
    let response = tokenizer
        .decode(&response_tokens, true)
        .map_err(|e| format!("Decoding error: {}", e))?
        .trim()
        .to_string();
    
    // Add assistant response to conversation
    conversation.push(("assistant".to_string(), response.clone()));
    
    // Trim conversation history
    if conversation.len() > max_history * 2 {
        conversation.drain(0..(conversation.len() - max_history * 2));
    }
    
    Ok((response, elapsed, response_tokens.len()))
}

fn generate_with_stopping(
    ctx: &ffi::Context,
    tokenizer: &Tokenizer,
    prompt_tokens: &[i32],
    max_tokens: usize,
    temperature: f32,
    min_p: f32,
) -> Result<Vec<i32>, String> {
    // Get special token IDs
    let eos_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(2) as i32;
    let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(7) as i32;
    
    // Forward pass on prompt
    let mut logits = ctx.forward(prompt_tokens);
    let mut output_tokens = prompt_tokens.to_vec();
    
    // Generate tokens
    for _ in 0..max_tokens {
        let next_token = sample_with_min_p(&logits, temperature, min_p);
        output_tokens.push(next_token);
        
        // Check for stop tokens
        if next_token == eos_id || next_token == im_end_id {
            break;
        }
        
        // Forward pass on single token
        logits = ctx.forward(&[next_token]);
    }
    
    Ok(output_tokens)
}

fn sample_with_min_p(logits: &[f32], temperature: f32, min_p: f32) -> i32 {
    if temperature < 1e-6 {
        // Greedy sampling
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i32)
            .unwrap_or(0);
    }
    
    // Apply temperature
    let scaled_logits: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    
    // Softmax
    let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled_logits
        .iter()
        .map(|&l| (l - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    
    // Apply min-p filtering
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    
    // Filter and renormalize
    let mut filtered_probs = probs.clone();
    filtered_probs.iter_mut().enumerate().for_each(|(_i, p)| {
        if *p < threshold {
            *p = 0.0;
        }
    });
    
    let filtered_sum: f32 = filtered_probs.iter().sum();
    if filtered_sum > 0.0 {
        filtered_probs.iter_mut().for_each(|p| *p /= filtered_sum);
    } else {
        // Fallback to original probs if all filtered out
        filtered_probs = probs;
    }
    
    // Sample from distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    
    for (idx, &prob) in filtered_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= r {
            return idx as i32;
        }
    }
    
    // Fallback
    (filtered_probs.len() - 1) as i32
}

fn run_inference(model_path: &str, prompt: &str, max_tokens: usize, temperature: f32) {
    println!("Loading model: {}", model_path);
    
    let model = match ffi::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    let ctx = match ffi::Context::new(&model, 2048, 128256) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    // Parse prompt as comma-separated token IDs
    let tokens: Vec<i32> = prompt
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    if tokens.is_empty() {
        eprintln!("Error: Invalid prompt format. Use comma-separated token IDs (e.g., '1,1098,5706')");
        std::process::exit(1);
    }
    
    println!("Prompt tokens: {:?}", tokens);
    println!("Generating {} tokens...\n", max_tokens);
    
    let start = Instant::now();
    let output = ctx.generate(&tokens, max_tokens, temperature);
    let elapsed = start.elapsed();
    
    println!("Generated tokens: {:?}", output);
    println!("\nGeneration complete:");
    println!("  Total tokens: {}", output.len());
    println!("  Time: {:.2}s", elapsed.as_secs_f32());
    println!("  Speed: {:.1} tok/s", output.len() as f32 / elapsed.as_secs_f32());
}

fn show_info(model_path: &str) {
    println!("Loading model: {}", model_path);
    
    let model = match ffi::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    model.print_info();
}

fn benchmark(model_path: &str, n_tokens: usize) {
    println!("Benchmarking: {}", model_path);
    println!("Generating {} tokens...\n", n_tokens);
    
    let model = match ffi::Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    let ctx = match ffi::Context::new(&model, 2048, 128256) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    // Test prompt: "The capital of France is"
    let prompt = vec![1, 1098, 5706, 803, 4481, 856];
    
    // Warmup
    println!("Warming up...");
    let _ = ctx.forward(&prompt);
    
    // Benchmark
    println!("Running benchmark...");
    let start = Instant::now();
    let output = ctx.generate(&prompt, n_tokens, 0.7);
    let elapsed = start.elapsed();
    
    println!("\nBenchmark Results:");
    println!("  Model: {}", model_path);
    println!("  Tokens generated: {}", output.len());
    println!("  Time: {:.2}s", elapsed.as_secs_f32());
    println!("  Speed: {:.1} tok/s", output.len() as f32 / elapsed.as_secs_f32());
    println!("  Avg time per token: {:.1}ms", elapsed.as_millis() as f32 / output.len() as f32);
}
