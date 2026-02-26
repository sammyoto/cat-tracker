use bincode::{Decode, Encode, error::DecodeError};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};
use cu29::payload::CuArray;

// 320 * 320 * 3 = 307200 bytes for RGB24
pub const FRAME_SIZE: usize = 320 * 320 * 3;

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
pub struct CameraSource {}

// Needs to be fully implemented if you want to have a stateful task.
impl Freezable for CameraSource {}

impl CuSrcTask for CameraSource {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(CameraFrame);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        Ok(Self {})
    }

    fn process(&mut self, _clock: &RobotClock, output: &mut Self::Output<'_>) -> CuResult<()> {
        let mut frame = CameraFrame {
        data: CuArray::new(),
        width: 320,
        height: 320,
        };
        // fill with placeholder data — real camera bytes go here later
        frame.data.fill_from_iter(std::iter::repeat(0u8).take(FRAME_SIZE));
        output.set_payload(frame);
        Ok(())
    }
}

#[derive(Default, Reflect)]
pub struct CatDetector {}

impl Freezable for CatDetector {}

impl CuTask for CatDetector {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(CameraFrame);
    type Output<'m> = output_msg!(CatDetection);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self>
    {
        Ok(Self {})
    }

    fn process(&mut self, _clock: &RobotClock, _input: &Self::Input<'_>, output: &mut Self::Output<'_>) -> CuResult<()> {
        // Placeholder — just say we found a cat in the center with 0.5 confidence
        output.set_payload(CatDetection {
            found: true,
            confidence: 0.5,
            center_x: 160,
            center_y: 160,
        });
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