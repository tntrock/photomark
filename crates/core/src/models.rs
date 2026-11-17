//! Finding and downloading the neural model files.
//!
//! The TrustMark networks are about 65 MB, so they are not part of the program. They
//! are looked for in a `models` folder next to the executable or in the per-user data
//! folder, and can be downloaded there from Adobe's model host.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::neural::{DECODER_FILE, ENCODER_FILE};
pub use crate::neural::{RUNTIME_FILE, RUNTIME_IS_BUNDLED};

/// Environment variable that overrides where the models are looked for.
pub const DIR_VARIABLE: &str = "PHOTOMARK_MODELS";

const BASE_URL: &str = "https://cc-assets.netlify.app/watermarking/trustmark-models/";

struct ModelFile {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const FILES: [ModelFile; 2] = [
    ModelFile {
        name: ENCODER_FILE,
        bytes: 17_312_208,
        sha256: "19b3d1b25836130ffd78775a8f61539f993375d1823ef0e59ba5b8dffb4f892d",
    },
    ModelFile {
        name: DECODER_FILE,
        bytes: 47_401_222,
        sha256: "ee3268f057c9dabef680e169302f5973d0589feea86189ed229a896cc3aa88df",
    },
];

/// Total download size in bytes.
pub const DOWNLOAD_BYTES: u64 = FILES[0].bytes + FILES[1].bytes;

/// Whether `dir` holds everything [`crate::NeuralEngine::load`] needs.
pub fn is_complete(dir: &Path) -> bool {
    FILES.iter().all(|file| dir.join(file.name).is_file())
        && (RUNTIME_IS_BUNDLED || dir.join(RUNTIME_FILE).is_file())
}

/// Per-user folder the models are downloaded to.
pub fn user_dir() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(base.join("photomark").join("models"))
}

/// Places the models are looked for, in order of preference.
pub fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os(DIR_VARIABLE) {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("models")))
    {
        dirs.push(dir);
    }
    dirs.extend(user_dir());
    dirs
}

/// The first folder that holds a complete set of models.
pub fn find() -> Option<PathBuf> {
    search_dirs().into_iter().find(|dir| is_complete(dir))
}

/// Download the model files into `dir`, verifying their checksums. Files that are
/// already there are kept. `progress` is called with the bytes fetched so far out of
/// [`DOWNLOAD_BYTES`].
pub fn fetch(dir: &Path, mut progress: impl FnMut(u64)) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut done = 0u64;
    for file in &FILES {
        let target = dir.join(file.name);
        if !target.is_file() {
            let partial = dir.join(format!("{}.part", file.name));
            let result = download(file, &partial, done, &mut progress);
            if let Err(error) = result.and_then(|()| fs::rename(&partial, &target)) {
                let _ = fs::remove_file(&partial);
                return Err(error);
            }
        }
        done += file.bytes;
        progress(done);
    }
    Ok(())
}

fn download(
    file: &ModelFile,
    partial: &Path,
    done_before: u64,
    progress: &mut impl FnMut(u64),
) -> io::Result<()> {
    let response = ureq::get(&format!("{BASE_URL}{}", file.name))
        .call()
        .map_err(io::Error::other)?;
    let mut reader = response.into_reader().take(file.bytes + 1);
    let mut out = File::create(partial)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut received = 0u64;
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        out.write_all(&buffer[..n])?;
        hasher.update(&buffer[..n]);
        received += n as u64;
        progress(done_before + received.min(file.bytes));
    }
    out.flush()?;
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if received != file.bytes || digest != file.sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} does not match its expected checksum", file.name),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_folder_is_incomplete() {
        let dir = std::env::temp_dir().join(format!("photomark-models-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert!(!is_complete(&dir));
        fs::remove_dir_all(&dir).unwrap();
    }
}
