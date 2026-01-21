fn main() {
    // Tell cargo to look for libomni.dylib in ../core/build
    println!("cargo:rustc-link-search=native=../core/build");
    
    // Tell cargo to link against libomni
    println!("cargo:rustc-link-lib=dylib=omni");
    
    // Rerun if the library changes
    println!("cargo:rerun-if-changed=../core/build/libomni.dylib");
}
