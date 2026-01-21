mod ffi;

use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "omni")]
#[command(about = "Fastest CPU inference engine for LLMs on Apple Silicon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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
