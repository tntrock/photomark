//! End-to-end tests on a synthetic, photo-like image.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, Rgba, RgbaImage};
use photomark_core::{detect, embed, DetectOptions, Detection, EmbedOptions, WatermarkKey};

const ID: &[u8; 8] = b"ALLEN-26";
const KEY: &str = "integration-test";

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut v = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    v ^= v >> 15;
    v = v.wrapping_mul(0x2C1B_3C6D);
    v ^= v >> 12;
    (v & 0xFFFF) as f32 / 65535.0
}

/// Smoothly interpolated lattice noise.
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (xi, yi) = (x0 as i32, y0 as i32);
    let top = hash(xi, yi, seed) * (1.0 - sx) + hash(xi + 1, yi, seed) * sx;
    let bottom = hash(xi, yi + 1, seed) * (1.0 - sx) + hash(xi + 1, yi + 1, seed) * sx;
    top * (1.0 - sy) + bottom * sy
}

/// A smooth "sky" above a textured "ground" with fine sensor-like noise.
fn photo(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32, y as f32);
        let horizon = h as f32 * (0.45 + 0.1 * value_noise(fx / 160.0, 0.0, 1));
        let mut level = 0.0;
        let mut scale = 96.0;
        for octave in 0..6 {
            level += value_noise(fx / scale, fy / scale, 10 + octave) / (1 << octave) as f32;
            scale /= 2.0;
        }
        level /= 2.0;
        let grain = (hash(x as i32, y as i32, 99) - 0.5) * 3.0;
        let [r, g, b] = if fy < horizon {
            let t = fy / horizon;
            [
                90.0 + 60.0 * t,
                140.0 + 50.0 * t,
                220.0 - 10.0 * t + 12.0 * level,
            ]
        } else {
            [
                40.0 + 150.0 * level,
                60.0 + 130.0 * level,
                30.0 + 90.0 * level,
            ]
        };
        Rgba([
            (r + grain).clamp(0.0, 255.0) as u8,
            (g + grain).clamp(0.0, 255.0) as u8,
            (b + grain).clamp(0.0, 255.0) as u8,
            255,
        ])
    })
}

fn marked() -> RgbaImage {
    let mut img = photo(960, 720);
    let report = embed(
        &mut img,
        &WatermarkKey::new(KEY),
        ID,
        &EmbedOptions::default(),
    );
    assert!(report.psnr > 38.0, "mark too visible: {} dB", report.psnr);
    img
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

fn find(img: &RgbaImage) -> Detection {
    detect(img, &WatermarkKey::new(KEY), &DetectOptions::default())
}

fn assert_found(img: &RgbaImage) -> Detection {
    let found = find(img);
    assert!(found.detected, "not detected: {found:?}");
    assert_eq!(found.payload.as_ref(), Some(ID), "{found:?}");
    found
}

#[test]
fn survives_untouched() {
    let found = assert_found(&marked());
    assert_eq!((found.quarter_turns, found.mirrored), (0, false));
    assert!((found.tile_period - 96.0).abs() < 0.5, "{found:?}");
}

#[test]
fn survives_jpeg() {
    assert_found(&jpeg(&marked(), 70));
}

#[test]
fn survives_heavy_crop() {
    // Keep about 9% of the area, at an offset that is not a multiple of the tile size.
    let crop = imageops::crop_imm(&marked(), 301, 187, 288, 216).to_image();
    assert_found(&jpeg(&crop, 80));
}

#[test]
fn survives_downscaled_crop() {
    let crop = imageops::crop_imm(&marked(), 123, 77, 640, 480).to_image();
    let small = imageops::resize(&crop, 419, 314, FilterType::Triangle);
    let found = assert_found(&jpeg(&small, 80));
    assert!(
        (found.tile_period - 96.0 * 419.0 / 640.0).abs() < 1.0,
        "{found:?}"
    );
}

#[test]
fn survives_upscaled_screenshot() {
    // The photo shown enlarged inside a window with flat borders, then captured.
    let shown = imageops::resize(&marked(), 1210, 907, FilterType::CatmullRom);
    let mut screen = RgbaImage::from_pixel(1400, 1000, Rgba([30, 30, 34, 255]));
    imageops::overlay(&mut screen, &shown, 95, 61);
    assert_found(&screen);
}

#[test]
fn survives_rotation_and_mirror() {
    let img = marked();
    let found = assert_found(&imageops::rotate90(&img));
    assert_eq!((found.quarter_turns, found.mirrored), (1, false));
    let found = assert_found(&imageops::rotate270(&img));
    assert_eq!((found.quarter_turns, found.mirrored), (3, false));
    let found = assert_found(&imageops::flip_horizontal(&img));
    assert!(found.mirrored);
}

#[test]
fn no_false_positive_on_unmarked_image() {
    let found = find(&photo(960, 720));
    assert!(!found.detected, "{found:?}");
    assert_eq!(found.payload, None);
}

#[test]
fn wrong_key_finds_nothing() {
    let found = detect(
        &marked(),
        &WatermarkKey::new("someone else's key"),
        &DetectOptions::default(),
    );
    assert!(!found.detected, "{found:?}");
}

#[test]
fn large_image_is_analysed_at_reduced_resolution() {
    let mut img = photo(1600, 1200);
    embed(
        &mut img,
        &WatermarkKey::new(KEY),
        ID,
        &EmbedOptions {
            strength: 1.0,
            cell: Some(8),
        },
    );
    let options = DetectOptions {
        max_work_pixels: 500_000,
    };
    let found = detect(&img, &WatermarkKey::new(KEY), &options);
    assert!(found.detected, "{found:?}");
    assert_eq!(found.payload.as_ref(), Some(ID));
    assert!((found.tile_period - 256.0).abs() < 1.0, "{found:?}");
}
