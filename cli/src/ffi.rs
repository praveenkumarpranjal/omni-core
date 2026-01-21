// FFI bindings to libomni.dylib (C++ library)

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

// Opaque pointers to C++ structs
#[repr(C)]
pub struct OmniModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OmniContext {
    _private: [u8; 0],
}

// External C functions from libomni.dylib
#[link(name = "omni")]
extern "C" {
    // Model loading
    pub fn omni_load_model(path: *const c_char) -> *mut OmniModel;
    pub fn omni_free_model(model: *mut OmniModel);
    
    // Context creation
    pub fn omni_create_context(model: *mut OmniModel, max_seq_len: c_int) -> *mut OmniContext;
    pub fn omni_free_context(ctx: *mut OmniContext);
    pub fn omni_reset_kv_cache(ctx: *mut OmniContext);
    
    // Inference
    pub fn omni_forward(
        ctx: *mut OmniContext,
        tokens: *const c_int,
        n_tokens: c_int,
        logits: *mut c_float,
    );
    
    // Generation
    pub fn omni_generate(
        ctx: *mut OmniContext,
        prompt: *const c_int,
        prompt_len: c_int,
        output: *mut c_int,
        max_tokens: c_int,
        temperature: c_float,
    ) -> c_int;
    
    // Sampling
    pub fn omni_sample(
        logits: *const c_float,
        vocab_size: c_int,
        temperature: c_float,
        top_p: c_float,
    ) -> c_int;
    
    // Info
    pub fn omni_print_info(model: *mut OmniModel);
}

// Safe Rust wrapper
pub struct Model {
    ptr: *mut OmniModel,
}

impl Model {
    pub fn load(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| format!("Invalid path: {}", e))?;
        
        unsafe {
            let ptr = omni_load_model(c_path.as_ptr());
            if ptr.is_null() {
                return Err(format!("Failed to load model: {}", path));
            }
            Ok(Model { ptr })
        }
    }
    
    pub fn print_info(&self) {
        unsafe {
            omni_print_info(self.ptr);
        }
    }
    
    pub fn as_ptr(&self) -> *mut OmniModel {
        self.ptr
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            omni_free_model(self.ptr);
        }
    }
}

pub struct Context {
    ptr: *mut OmniContext,
    vocab_size: usize,
}

impl Context {
    pub fn new(model: &Model, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        unsafe {
            let ptr = omni_create_context(model.as_ptr(), max_seq_len as c_int);
            if ptr.is_null() {
                return Err("Failed to create context".to_string());
            }
            Ok(Context { ptr, vocab_size })
        }
    }
    
    pub fn reset_kv_cache(&self) {
        unsafe {
            omni_reset_kv_cache(self.ptr);
        }
    }
    
    pub fn forward(&self, tokens: &[i32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];
        
        unsafe {
            omni_forward(
                self.ptr,
                tokens.as_ptr() as *const c_int,
                tokens.len() as c_int,
                logits.as_mut_ptr(),
            );
        }
        
        logits
    }
    
    pub fn generate(
        &self,
        prompt: &[i32],
        max_tokens: usize,
        temperature: f32,
    ) -> Vec<i32> {
        let mut output = vec![0i32; prompt.len() + max_tokens];
        
        unsafe {
            let total_len = omni_generate(
                self.ptr,
                prompt.as_ptr() as *const c_int,
                prompt.len() as c_int,
                output.as_mut_ptr() as *mut c_int,
                max_tokens as c_int,
                temperature,
            );
            
            output.truncate(total_len as usize);
        }
        
        output
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            omni_free_context(self.ptr);
        }
    }
}

// Thread-safe wrappers
unsafe impl Send for Model {}
unsafe impl Send for Context {}
