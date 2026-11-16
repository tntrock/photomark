use std::path::{Path, PathBuf};
use std::process::ExitCode;

use photomark_core::io::{self, OutputFormat};
use photomark_core::{
    detect, embed, payload_from_text, payload_to_text, DetectOptions, EmbedOptions, WatermarkKey,
    PAYLOAD_BYTES,
};

const USAGE: &str = "\
photomark-cli - invisible, crop-resistant photo watermarks

USAGE:
    photomark-cli embed <INPUT> <OUTPUT> [--id TEXT] [--key KEY] [--strength N] [--cell PX] [--quality Q]
    photomark-cli detect <IMAGE> [--key KEY]

OPTIONS:
    --id TEXT       Text to embed, at most 8 bytes of UTF-8 (default: empty)
    --key KEY       Secret key; detection needs the same key (default: built-in key)
    --strength N    Watermark strength, 1.0 = default
    --cell PX       Cell size in pixels (default: chosen from the image size)
    --quality Q     JPEG quality when OUTPUT is .jpg (default: 92)

The output format follows the OUTPUT extension (.png or .jpg).
`detect` exits with status 0 if a watermark was found and 1 if not.";

struct Args {
    positional: Vec<String>,
    id: String,
    key: String,
    strength: f32,
    cell: Option<u32>,
    quality: u8,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        positional: Vec::new(),
        id: String::new(),
        key: String::new(),
        strength: 1.0,
        cell: None,
        quality: 92,
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
    let report = embed(
        &mut loaded.pixels,
        &WatermarkKey::new(&args.key),
        &payload,
        &options,
    );
    io::save(
        &output,
        &loaded.pixels,
        format,
        loaded.icc_profile.as_deref(),
    )
    .map_err(|e| format!("{}: {e}", output.display()))?;
    println!(
        "wrote {} (cell {} px, PSNR {:.1} dB)",
        output.display(),
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
    let found = detect(
        &loaded.pixels,
        &WatermarkKey::new(&args.key),
        &DetectOptions::default(),
    );

    if !found.detected {
        println!("no watermark found (best score {:.1})", found.score);
        return Ok(ExitCode::from(1));
    }
    match &found.payload {
        Some(payload) => println!("watermark found, id: \"{}\"", payload_to_text(payload)),
        None => println!("watermark found, but the id could not be read"),
    }
    println!(
        "score {:.1}, tile period {:.1} px, rotated {} deg{}",
        found.score,
        found.tile_period,
        found.quarter_turns as u32 * 90,
        if found.mirrored { ", mirrored" } else { "" }
    );
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
