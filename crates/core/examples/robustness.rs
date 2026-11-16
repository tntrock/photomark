//! Embed a watermark into a photo, attack it in various ways and report what survives.
//!
//! ```text
//! cargo run --release -p photomark-core --example robustness -- photo.jpg [strength]
//! ```

use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, Rgba, RgbaImage};
use photomark_core::{
    detect, embed, io, payload_to_text, DetectOptions, EmbedOptions, WatermarkKey,
};

const ID: &[u8; 8] = b"TEST-ID1";

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

/// Crop to `fraction` of the area, off-centre so the tile grid is misaligned.
fn crop(img: &RgbaImage, fraction: f32) -> RgbaImage {
    let side = fraction.sqrt();
    let (w, h) = img.dimensions();
    let (cw, ch) = ((w as f32 * side) as u32, (h as f32 * side) as u32);
    let (x, y) = ((w - cw) / 3 + 5, (h - ch) * 2 / 3 + 3);
    imageops::crop_imm(img, x.min(w - cw), y.min(h - ch), cw, ch).to_image()
}

fn scale(img: &RgbaImage, factor: f32) -> RgbaImage {
    let (w, h) = img.dimensions();
    imageops::resize(
        img,
        (w as f32 * factor).round() as u32,
        (h as f32 * factor).round() as u32,
        if factor < 1.0 {
            FilterType::Triangle
        } else {
            FilterType::CatmullRom
        },
    )
}

/// The image shown scaled inside a window, surrounded by flat UI and a title bar.
fn screenshot(img: &RgbaImage, factor: f32) -> RgbaImage {
    let shown = scale(img, factor);
    let (w, h) = shown.dimensions();
    let mut canvas = RgbaImage::from_pixel(w + 137, h + 91, Rgba([32, 32, 36, 255]));
    for y in 0..40 {
        for x in 0..canvas.width() {
            canvas.put_pixel(x, y, Rgba([240, 240, 240, 255]));
        }
    }
    imageops::overlay(&mut canvas, &shown, 71, 47);
    canvas
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: robustness <photo> [strength]");
    let strength: f32 = args.next().map_or(1.0, |s| s.parse().unwrap());

    let original = io::load(Path::new(&path)).unwrap().pixels;
    let key = WatermarkKey::new("robustness");
    let mut marked = original.clone();
    let options = EmbedOptions {
        strength,
        cell: None,
    };
    let report = embed(&mut marked, &key, ID, &options);
    println!(
        "{path}: {}x{}, cell {} px, PSNR {:.1} dB",
        marked.width(),
        marked.height(),
        report.cell,
        report.psnr
    );

    let cases: Vec<(&str, RgbaImage)> = vec![
        ("untouched", marked.clone()),
        ("jpeg q85", jpeg(&marked, 85)),
        ("jpeg q60", jpeg(&marked, 60)),
        ("crop to 25%", crop(&marked, 0.25)),
        ("crop to 10%", crop(&marked, 0.10)),
        ("crop to 5%", crop(&marked, 0.05)),
        ("crop to 2%", crop(&marked, 0.02)),
        ("scale 0.5", scale(&marked, 0.5)),
        ("scale 0.73", scale(&marked, 0.73)),
        ("scale 1.37", scale(&marked, 1.37)),
        ("crop 25% + jpeg q75", jpeg(&crop(&marked, 0.25), 75)),
        (
            "crop 20% + scale 0.6 + jpeg q75",
            jpeg(&scale(&crop(&marked, 0.2), 0.6), 75),
        ),
        (
            "crop 10% + scale 1.6 + jpeg q80",
            jpeg(&scale(&crop(&marked, 0.1), 1.6), 80),
        ),
        ("screenshot 0.45", screenshot(&marked, 0.45)),
        (
            "screenshot 0.62 + crop 30% + jpeg q80",
            jpeg(&crop(&screenshot(&marked, 0.62), 0.3), 80),
        ),
        ("rotate 90", imageops::rotate90(&marked)),
        (
            "mirror + crop 25% + jpeg q80",
            jpeg(&crop(&imageops::flip_horizontal(&marked), 0.25), 80),
        ),
    ];

    let detect_options = DetectOptions::default();
    let mut passed = 0;
    for (name, img) in &cases {
        let start = Instant::now();
        let found = detect(img, &key, &detect_options);
        let ok = found.detected && found.payload.as_ref() == Some(ID);
        passed += ok as usize;
        println!(
            "  {:<40} {:>4}x{:<4} {}  score {:>6.1}  period {:>6.1}  id {:<10} {:.1}s",
            name,
            img.width(),
            img.height(),
            if ok { "PASS" } else { "FAIL" },
            found.score,
            found.tile_period,
            found.payload.as_ref().map_or("-".into(), payload_to_text),
            start.elapsed().as_secs_f32()
        );
    }
    println!("  {passed}/{} attacks survived", cases.len());

    for (name, img, key) in [
        ("unmarked original", &original, &key),
        ("marked, wrong key", &marked, &WatermarkKey::new("wrong")),
    ] {
        let found = detect(img, key, &detect_options);
        println!(
            "  {:<40} {}  score {:.1}",
            name,
            if found.detected {
                "FALSE POSITIVE"
            } else {
                "clean"
            },
            found.score
        );
    }
}
