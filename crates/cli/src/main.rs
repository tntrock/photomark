use std::path::{Path, PathBuf};
use std::process::ExitCode;

use photomark_core::io::{self, OutputFormat};
use photomark_core::{
    detect_layers, embed_layers, models, payload_from_text, payload_to_text, DetectOptions,
    EmbedOptions, NeuralEngine, WatermarkKey, PAYLOAD_BYTES,
};

const USAGE: &str = "\
photomark-cli - invisible, crop-resistant photo watermarks

USAGE:
    photomark-cli embed <INPUT> <OUTPUT> [--id TEXT] [--key KEY] [--strength N] [--cell PX] [--quality Q]
    photomark-cli detect <IMAGE> [--key KEY]
    photomark-cli fetch-models [DIR]

OPTIONS:
    --id TEXT       Text to embed, at most 8 bytes of UTF-8 (default: empty)
    --key KEY       Secret key; detection needs the same key (default: built-in key)
    --strength N    Watermark strength, 1.0 = default
    --cell PX       Cell size in pixels (default: chosen from the image size)
    --quality Q     JPEG quality when OUTPUT is .jpg (default: 92)
    --no-neural     Use only the tile layer, even if the neural models are installed

The output format follows the OUTPUT extension (.png or .jpg).
The neural (TrustMark) layer is used when its models are installed; `fetch-models`
downloads them (about 65 MB) to the per-user folder or to DIR.
`detect` exits with status 0 if a watermark was found and 1 if not.";

struct Args {
    positional: Vec<String>,
    id: String,
    key: String,
    strength: f32,
    cell: Option<u32>,
    quality: u8,
    no_neural: bool,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        positional: Vec::new(),
        id: String::new(),
        key: String::new(),
        strength: 1.0,
        cell: None,
        quality: 92,
        no_neural: false,
    };
    let mut args = args;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--id" => parsed.id = value("--id")?,
            "--key" => parsed.key = value("--key")?,
            "--strength" => {
                parsed.strength = value("--strength")?
                    .parse()
                    .map_err(|_| "--strength must be a number")?
            }
            "--cell" => {
                parsed.cell = Some(
                    value("--cell")?
                        .parse()
                        .map_err(|_| "--cell must be a whole number")?,
                )
            }
            "--quality" => {
                parsed.quality = value("--quality")?
                    .parse()
                    .map_err(|_| "--quality must be 1-100")?
            }
            "--no-neural" => parsed.no_neural = true,
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => parsed.positional.push(arg),
        }
    }
    Ok(parsed)
}

fn output_format(path: &Path, quality: u8) -> Result<OutputFormat, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Ok(OutputFormat::Png),
        "jpg" | "jpeg" => Ok(OutputFormat::Jpeg(quality)),
        _ => Err("output must end in .png or .jpg".into()),
    }
}

/// The neural engine, if wanted and installed. Problems are reported but not fatal:
/// the tile layer works on its own.
fn neural_engine(args: &Args) -> Option<NeuralEngine> {
    if args.no_neural {
        return None;
    }
    let Some(dir) = models::find() else {
        eprintln!(
            "note: neural models not installed, using the tile layer only (see fetch-models)"
        );
        return None;
    };
    match NeuralEngine::load(&dir) {
        Ok(engine) => Some(engine),
        Err(e) => {
            eprintln!("warning: neural layer unavailable: {e}");
            None
        }
    }
}

fn run_fetch(args: &Args) -> Result<ExitCode, String> {
    let dir = match args.positional.as_slice() {
        [_] => models::user_dir().ok_or("no per-user data folder; pass a DIR")?,
        [_, dir] => PathBuf::from(dir),
        _ => return Err("fetch-models takes at most one DIR".into()),
    };
    println!("downloading models to {}", dir.display());
    let mut shown = u64::MAX;
    models::fetch(&dir, |done| {
        let percent = done * 100 / models::DOWNLOAD_BYTES;
        if percent / 10 != shown {
            shown = percent / 10;
            println!("  {percent}%");
        }
    })
    .map_err(|e| format!("download failed: {e}"))?;
    if models::is_complete(&dir) {
        println!("models are ready");
    } else {
        println!(
            "models downloaded; this build also needs {} in that folder",
            models::RUNTIME_FILE
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn run_embed(args: &Args) -> Result<ExitCode, String> {
    let [_, input, output] = args.positional.as_slice() else {
        return Err("embed needs <INPUT> and <OUTPUT>".into());
    };
    let (input, output) = (PathBuf::from(input), PathBuf::from(output));
    let format = output_format(&output, args.quality)?;
    let payload = payload_from_text(&args.id).ok_or(format!(
        "--id must be at most {PAYLOAD_BYTES} bytes of UTF-8"
    ))?;

    let mut loaded = io::load(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let options = EmbedOptions {
        strength: args.strength,
        cell: args.cell,
    };
    let engine = neural_engine(args);
    let report = embed_layers(
        &mut loaded.pixels,
        &WatermarkKey::new(&args.key),
        &payload,
        &options,
        engine.as_ref(),
    )
    .map_err(|e| e.to_string())?;
    io::save(
        &output,
        &loaded.pixels,
        format,
        loaded.icc_profile.as_deref(),
    )
    .map_err(|e| format!("{}: {e}", output.display()))?;
    println!(
        "wrote {} ({}, cell {} px, PSNR {:.1} dB)",
        output.display(),
        if report.neural {
            "neural + tile layers"
        } else {
            "tile layer only"
        },
        report.cell,
        report.psnr
    );
    Ok(ExitCode::SUCCESS)
}

fn run_detect(args: &Args) -> Result<ExitCode, String> {
    let [_, input] = args.positional.as_slice() else {
        return Err("detect needs <IMAGE>".into());
    };
    let input = PathBuf::from(input);
    let loaded = io::load(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let engine = neural_engine(args);
    let found = detect_layers(
        &loaded.pixels,
        &WatermarkKey::new(&args.key),
        &DetectOptions::default(),
        engine.as_ref(),
    )
    .map_err(|e| e.to_string())?;

    if !found.detected {
        println!("no watermark found (best score {:.1})", found.tile.score);
        return Ok(ExitCode::from(1));
    }
    match &found.payload {
        Some(payload) => println!("watermark found, id: \"{}\"", payload_to_text(payload)),
        None => println!("watermark found, but the id could not be read"),
    }
    match &found.neural {
        Some(neural) => println!(
            "neural layer: found, {} bits corrected, rotated {} deg",
            neural.corrected,
            neural.quarter_turns as u32 * 90
        ),
        None if engine.is_some() => println!("neural layer: not found"),
        None => {}
    }
    let tile = &found.tile;
    if tile.detected {
        println!(
            "tile layer: found, score {:.1}, tile period {:.1} px, rotated {} deg{}",
            tile.score,
            tile.tile_period,
            tile.quarter_turns as u32 * 90,
            if tile.mirrored { ", mirrored" } else { "" }
        );
    } else {
        println!("tile layer: not found (best score {:.1})", tile.score);
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match args.positional.first().map(String::as_str) {
        Some("embed") => run_embed(&args),
        Some("detect") => run_detect(&args),
        Some("fetch-models") => run_fetch(&args),
        _ => {
            println!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    result.unwrap_or_else(|message| {
        eprintln!("error: {message}");
        ExitCode::from(2)
    })
}
