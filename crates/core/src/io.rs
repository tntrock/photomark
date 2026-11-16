//! Loading and saving image files.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageError, ImageReader, RgbaImage};

/// A decoded image, upright, with the colour profile it was tagged with.
pub struct Loaded {
    pub pixels: RgbaImage,
    pub icc_profile: Option<Vec<u8>>,
}

/// File extensions [`load`] understands.
pub const EXTENSIONS: [&str; 8] = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "jfif"];

/// Whether `path` has one of the supported image extensions.
pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Decode an image, applying its EXIF orientation so the pixels are upright.
pub fn load(path: &Path) -> Result<Loaded, ImageError> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let icc_profile = decoder.icc_profile()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(Loaded {
        pixels: image.into_rgba8(),
        icc_profile,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Png,
    /// JPEG with the given quality (1..=100).
    Jpeg(u8),
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg(_) => "jpg",
        }
    }
}

/// Encode `pixels` to `path`. JPEG output drops the alpha channel.
pub fn save(
    path: &Path,
    pixels: &RgbaImage,
    format: OutputFormat,
    icc_profile: Option<&[u8]>,
) -> Result<(), ImageError> {
    let writer = BufWriter::new(File::create(path)?);
    let (w, h) = pixels.dimensions();
    match format {
        OutputFormat::Png => {
            let mut encoder = PngEncoder::new(writer);
            if let Some(icc) = icc_profile {
                // A profile the encoder cannot store is not worth failing the save for.
                let _ = encoder.set_icc_profile(icc.to_vec());
            }
            encoder.write_image(pixels, w, h, image::ExtendedColorType::Rgba8)
        }
        OutputFormat::Jpeg(quality) => {
            let rgb = DynamicImage::ImageRgba8(pixels.clone()).into_rgb8();
            let mut encoder = JpegEncoder::new_with_quality(writer, quality.clamp(1, 100));
            if let Some(icc) = icc_profile {
                let _ = encoder.set_icc_profile(icc.to_vec());
            }
            encoder.write_image(&rgb, w, h, image::ExtendedColorType::Rgb8)
        }
    }
}
