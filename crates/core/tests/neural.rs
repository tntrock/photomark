//! End-to-end tests of the neural layer. They need the model files, so they pass
//! trivially when the models are not installed (run `photomark-cli fetch-models`).

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, Rgba, RgbaImage};
use photomark_core::{
    detect_layers, embed_layers, models, DetectOptions, EmbedOptions, NeuralEngine, WatermarkKey,
};

const ID: &[u8; 8] = b"ALLEN-26";
const KEY: &str = "neural-test";

fn engine() -> Option<NeuralEngine> {
    let Some(dir) = models::find() else {
        eprintln!("neural models not installed; skipping");
        return None;
    };
    Some(NeuralEngine::load(&dir).expect("models load"))
}

/// Soft colour blobs with some grain: enough structure for the encoder to hide in.
fn photo(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        let wave = |a: f32, b: f32, c: f32| ((fx * a + fy * b + c).sin() * 0.5 + 0.5) * 200.0;
        let grain = ((x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77)) >> 28) as f32;
        Rgba([
            (20.0 + wave(9.0, 4.0, 0.3) + grain) as u8,
            (20.0 + wave(3.0, 11.0, 1.7) + grain) as u8,
            (20.0 + wave(6.0, 7.0, 4.1) + grain) as u8,
            255,
        ])
    })
}

fn jpeg(img: &RgbaImage, quality: u8) -> RgbaImage {
    let rgb = DynamicImage::ImageRgba8(img.clone()).into_rgb8();
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, quality)
        .encode_image(&rgb)
        .unwrap();
    image::load(Cursor::new(bytes), image::ImageFormat::Jpeg)
        .unwrap()
        .into_rgba8()
}

#[test]
fn neural_layer_survives_what_the_tile_layer_cannot() {
    let Some(engine) = engine() else { return };
    let key = WatermarkKey::new(KEY);
    let original = photo(1200, 900);
    let mut marked = original.clone();
    let report = embed_layers(
        &mut marked,
        &key,
        ID,
        &EmbedOptions::default(),
        Some(&engine),
    )
    .unwrap();
    assert!(report.neural);
    assert!(report.psnr > 33.0, "psnr {}", report.psnr);

    let options = DetectOptions::default();
    let read = |img: &RgbaImage, key: &WatermarkKey| {
        detect_layers(img, key, &options, Some(&engine)).unwrap()
    };

    let clean = read(&marked, &key);
    assert_eq!(clean.neural.map(|n| n.payload), Some(*ID));
    assert_eq!(clean.tile.payload, Some(*ID));

    // Far too small and too compressed for the tile layer.
    let tiny = jpeg(
        &imageops::resize(&marked, 240, 180, FilterType::Triangle),
        85,
    );
    let found = read(&tiny, &key);
    assert_eq!(found.payload, Some(*ID));
    assert!(found.neural.is_some());

    let turned = read(&imageops::rotate90(&jpeg(&marked, 30)), &key);
    assert_eq!(
        turned.neural.map(|n| (n.payload, n.quarter_turns)),
        Some((*ID, 1))
    );

    // A heavy crop is the tile layer's job.
    let cropped = imageops::crop_imm(&marked, 500, 300, 420, 380).to_image();
    let found = read(&cropped, &key);
    assert_eq!(found.payload, Some(*ID));

    assert!(read(&original, &key).neural.is_none());
    assert!(read(&marked, &WatermarkKey::new("someone else"))
        .neural
        .is_none());
}
