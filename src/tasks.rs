use bincode::{Decode, Encode, error::DecodeError};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};
use cu29::payload::CuArray;

// video 4 linux imports
use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

// onnx runtime imports
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use ndarray::{Array, Axis, s};

// 640 * 480 * 2 = 614400  bytes for YUYV
pub const FRAME_SIZE: usize = 640 * 480 * 2;

#[derive(Default, Debug, Clone, Encode, Serialize, Deserialize, Reflect)]
pub struct CameraFrame {
    pub data: CuArray<u8, FRAME_SIZE>,
    pub width: u32,
    pub height: u32,
}

impl bincode::Decode<()> for CameraFrame {
    fn decode<D: bincode::de::Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            data: bincode::Decode::decode(decoder)?,
            width: bincode::Decode::decode(decoder)?,
            height: bincode::Decode::decode(decoder)?,
        })
    }
}

#[derive(Default, Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub struct CatDetection {
    pub found: bool,
    pub confidence: f32,
    pub center_x: u32,
    pub center_y: u32,
}

// Defines a source (ie. driver)
#[derive(Default, Reflect)]
pub struct CameraSource {
    #[reflect(ignore)]
    pub stream: Option<v4l::io::mmap::Stream<'static>>,
}

// Needs to be fully implemented if you want to have a stateful task.
impl Freezable for CameraSource {}

impl CuSrcTask for CameraSource {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(CameraFrame);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        // Create a new capture device with a few extra parameters
        let mut dev = Device::new(0).expect("Failed to open device");

        // Let's say we want to explicitly request another format
        let mut fmt = dev.format().expect("Failed to read format");
        fmt.width = 640;
        fmt.height = 480;
        fmt.fourcc = FourCC::new(b"YUYV");
        let fmt = dev.set_format(&fmt).map_err(|_| CuError::from("Camera doesn't support YUYV"))?;

        let stream = Stream::with_buffers(&mut dev, Type::VideoCapture, 4)
        .expect("Failed to create buffer stream");

        let fmt = dev.format().unwrap();
        println!("Camera is actually using: {} at {}x{}", fmt.fourcc, fmt.width, fmt.height);

        Ok(Self { stream: Some(stream) })
    }

    fn process(&mut self, _clock: &RobotClock, output: &mut Self::Output<'_>) -> CuResult<()> {
        let (data, _meta) = self.stream.as_mut().unwrap().next().map_err(|_| CuError::from("Camera timeout"))?;

        let mut frame = CameraFrame {
            data: CuArray::new(),
            width: 640,
            height: 480,
        };

        frame.data.fill_from_iter(data[..FRAME_SIZE].iter().copied());

        output.set_payload(frame);
        Ok(())
    }
}

#[derive(Default, Reflect)]
pub struct CatDetector {
    pub session: Option<Session>,
}

impl Freezable for CatDetector {}

impl CuTask for CatDetector {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CameraFrame);
    type Output<'m> = output_msg!(CatDetection);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    {   
        println!("Initializing ONNX session...");

        let session = Session::builder()
        .map_err(|e| CuError::from(format!("Failed to create ONNX builder: {}", e)))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| CuError::from(format!("Failed to set optimization: {}", e)))?
        .with_intra_threads(1)
        .map_err(|e| CuError::from(format!("Failed to set threads: {}", e)))?
        .commit_from_file("src/assets/yolov8n.onnx")
        .map_err(|e| CuError::from(format!("Failed to load model file: {}", e)))?;

        for input in session.inputs() {
            println!("Input name: {}", input.name());
            println!("Input shape: {:?}", input.dtype());
        }

        Ok(Self { session: Some(session) })
    }

    fn process(&mut self, _clock: &RobotClock, _input: &Self::Input<'_>, output: &mut Self::Output<'_>) -> CuResult<()> {
        let raw_data: &[u8] = _input.payload().unwrap().data.as_slice();
        let processed_data = process_to_320_tensor(raw_data);

        // 2. Prepare for ONNX
        let input = Value::from_array(
            ndarray::Array4::from_shape_vec((1, 3, 320, 320), processed_data).unwrap()
        ).map_err(|e| CuError::from(format!("Input error: {}", e)))?;

        // 3. Get Session
        let session = self.session.as_mut().unwrap();

        // 4. Run inference
        let outputs = session.run(ort::inputs!["images" => input]).unwrap();
        let parsed_outputs = outputs["output0"].try_extract_array::<f32>().unwrap().t().into_owned();

        let view = parsed_outputs.view();

        let mut best_cat_score = 0.5; // Only care about cats with > 50% confidence
        let mut cat_coords = (0.0, 0.0); // (x, y)
        let mut found_cat = false;

        const CAT_CLASS_INDEX: usize = 15; // COCO index for cat

        // 2. Iterate through the 2100 candidates
        for col in 0..2100 {
            let score = view[[col, 4 + CAT_CLASS_INDEX, 0]];
            
            if score > best_cat_score {
                best_cat_score = score;
                found_cat = true;
                
                // These coordinates are in the 320x320 space
                let x = view[[col, 0, 0]];
                let y = view[[col, 1, 0]];
                
                // Adjust for the 40-pixel vertical letterbox we added earlier!
                cat_coords = (x, y - 40.0); 
            }
        }

        // 3. Update your CatDetection payload
        let mut detection = CatDetection::default();

        if found_cat {
            detection.found = true;
            detection.confidence = best_cat_score;
            detection.center_x = cat_coords.0 as u32;
            detection.center_y = cat_coords.1 as u32;
            println!("Cat found! x: {}, y: {} (Conf: {:.2})", detection.center_x, detection.center_y, best_cat_score);
        }

        output.set_payload(detection);

        Ok(())
    }
}

#[derive(Default, Reflect)]
pub struct ServoSink {}

impl Freezable for ServoSink {}

impl CuSinkTask for ServoSink {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CatDetection);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn process(&mut self, _clock: &RobotClock, input: &Self::Input<'_>) -> CuResult<()> {
        // Placeholder — just print out the cat detection
        println!("Cat detection: {:?}", input.payload());
        Ok(())
    }
}

fn process_to_320_tensor(yuyv_640: &[u8]) -> Vec<f32> {
    const IN_W: usize = 640;
    const IN_H: usize = 480;
    const OUT_SIZE: usize = 320;
    
    // Create a buffer for 320x320x3 (Planar: RRR...GGG...BBB...)
    let mut out_tensor = vec![0.0f32; OUT_SIZE * OUT_SIZE * 3];
    
    // We want to center our 320x240 image inside the 320x320 square (letterboxing)
    // This leaves 40 pixels of black padding at the top and bottom.
    let vertical_offset = 40; 

    for y in 0..240 { // We take every 2nd row from the 480 height
        let in_row_idx = y * 2 * IN_W * 2; // Row * 2 (skip) * Width * 2 (YUYV bytes)
        let out_y = y + vertical_offset;

        for x in 0..160 { // Process 2 pixels at a time (YUYV group)
            // Skip every other YUYV block to go from 640 to 320 width
            let in_idx = in_row_idx + (x * 2 * 4); // x * 2 (skip) * 4 bytes per group

            let y0 = yuyv_640[in_idx] as f32;
            let u  = yuyv_640[in_idx + 1] as f32 - 128.0;
            let y1 = yuyv_640[in_idx + 2] as f32;
            let v  = yuyv_640[in_idx + 3] as f32 - 128.0;

            // Pixel 1 RGB
            let r0 = (y0 + 1.402 * v).clamp(0.0, 255.0) / 255.0;
            let g0 = (y0 - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0) / 255.0;
            let b0 = (y0 + 1.772 * u).clamp(0.0, 255.0) / 255.0;

            // Pixel 2 RGB
            let r1 = (y1 + 1.402 * v).clamp(0.0, 255.0) / 255.0;
            let g1 = (y1 - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0) / 255.0;
            let b1 = (y1 + 1.772 * u).clamp(0.0, 255.0) / 255.0;

            // Place into Planar Tensor layout [R, G, B]
            let out_x_left = x * 2;
            let out_x_right = x * 2 + 1;

            // Red plane
            out_tensor[out_y * OUT_SIZE + out_x_left] = r0;
            out_tensor[out_y * OUT_SIZE + out_x_right] = r1;
            // Green plane
            out_tensor[OUT_SIZE * OUT_SIZE + out_y * OUT_SIZE + out_x_left] = g0;
            out_tensor[OUT_SIZE * OUT_SIZE + out_y * OUT_SIZE + out_x_right] = g1;
            // Blue plane
            out_tensor[2 * OUT_SIZE * OUT_SIZE + out_y * OUT_SIZE + out_x_left] = b0;
            out_tensor[2 * OUT_SIZE * OUT_SIZE + out_y * OUT_SIZE + out_x_right] = b1;
        }
    }
    out_tensor
}