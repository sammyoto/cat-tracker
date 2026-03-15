use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{ep::CPU};

fn main() {
    println!("Forcing CPU-only execution...");
    let session = Session::builder()
        .unwrap()
        // This explicitly tells it: Don't look for GPUs
        .with_execution_providers([CPU::default().build()])
        .unwrap()
        .commit_from_file("./src/assets/yolov8n.onnx");

    match session {
        Ok(_) => println!("Success!"),
        Err(e) => println!("Error: {:?}", e),
    }
}