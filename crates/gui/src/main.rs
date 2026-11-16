#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use eframe::egui::{self, Color32, RichText};
use image::RgbaImage;
use photomark_core::io::{self, OutputFormat};
use photomark_core::{
    detect, embed, payload_from_text, payload_to_text, DetectOptions, Detection, EmbedOptions,
    WatermarkKey, PAYLOAD_BYTES,
};

/// Longest side of the preview texture shown on the detect tab.
const PREVIEW_SIDE: u32 = 1024;

const GREEN: Color32 = Color32::from_rgb(60, 170, 90);
const RED: Color32 = Color32::from_rgb(210, 80, 70);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([860.0, 680.0])
            .with_min_inner_size([640.0, 480.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Photomark",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

/// Use a system CJK font so the Traditional Chinese interface renders.
/// Returns false when none is installed, in which case the UI falls back to English.
fn install_cjk_font(ctx: &egui::Context) -> bool {
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
    for name in ["msjh.ttc", "msjh.ttf", "mingliu.ttc", "msyh.ttc"] {
        let Ok(bytes) = std::fs::read(Path::new(&windir).join("Fonts").join(name)) else {
            continue;
        };
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("cjk".to_owned(), egui::FontData::from_owned(bytes).into());
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .entry(family)
                .or_default()
                .push("cjk".to_owned());
        }
        ctx.set_fonts(fonts);
        return true;
    }
    false
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Embed,
    Detect,
}

#[derive(PartialEq, Clone, Copy)]
enum FormatChoice {
    /// JPEG sources stay JPEG, everything else becomes PNG.
    Auto,
    Jpeg,
    Png,
}

enum EmbedMsg {
    Line(String),
    Finished,
}

enum DetectMsg {
    Preview(egui::ColorImage),
    Finished(Result<Detection, String>),
}

enum DetectSource {
    File(PathBuf),
    Pixels(RgbaImage),
}

struct App {
    zh: bool,
    tab: Tab,
    key: String,

    files: Vec<PathBuf>,
    id: String,
    strength: f32,
    format: FormatChoice,
    quality: u8,
    out_dir: Option<PathBuf>,
    log: Vec<String>,
    embed_job: Option<Receiver<EmbedMsg>>,
    embed_done: usize,
    embed_total: usize,

    detect_label: String,
    preview: Option<egui::TextureHandle>,
    detect_job: Option<Receiver<DetectMsg>>,
    detect_result: Option<Result<Detection, String>>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let zh = install_cjk_font(&cc.egui_ctx);
        Self {
            zh,
            tab: Tab::Embed,
            key: String::new(),
            files: Vec::new(),
            id: String::new(),
            strength: 1.0,
            format: FormatChoice::Auto,
            quality: 92,
            out_dir: None,
            log: Vec::new(),
            embed_job: None,
            embed_done: 0,
            embed_total: 0,
            detect_label: String::new(),
            preview: None,
            detect_job: None,
            detect_result: None,
        }
    }

    /// Pick the Chinese or English wording.
    fn t(&self, zh: &'static str, en: &'static str) -> &'static str {
        if self.zh {
            zh
        } else {
            en
        }
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for path in paths {
            if path.is_dir() {
                let Ok(entries) = std::fs::read_dir(&path) else {
                    continue;
                };
                let mut found: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && io::is_supported(p))
                    .collect();
                found.sort();
                self.add_paths(found);
            } else if io::is_supported(&path) && !self.files.contains(&path) {
                self.files.push(path);
            }
        }
    }

    fn start_embed(&mut self, ctx: &egui::Context) {
        let Some(payload) = payload_from_text(&self.id) else {
            return;
        };
        let files = self.files.clone();
        let key = WatermarkKey::new(&self.key);
        let options = EmbedOptions {
            strength: self.strength,
            cell: None,
        };
        let (format, quality, out_dir, zh) =
            (self.format, self.quality, self.out_dir.clone(), self.zh);
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();

        self.log.clear();
        self.embed_done = 0;
        self.embed_total = files.len();
        self.embed_job = Some(rx);

        thread::spawn(move || {
            for input in files {
                let name = file_name(&input);
                let result = embed_file(
                    &input,
                    &key,
                    &payload,
                    &options,
                    format,
                    quality,
                    out_dir.as_deref(),
                );
                let line = match result {
                    Ok((output, psnr)) => {
                        format!("[OK] {name} → {} (PSNR {psnr:.1} dB)", output.display())
                    }
                    Err(e) if zh => format!("[失敗] {name}：{e}"),
                    Err(e) => format!("[FAILED] {name}: {e}"),
                };
                if tx.send(EmbedMsg::Line(line)).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
            let _ = tx.send(EmbedMsg::Finished);
            ctx.request_repaint();
        });
    }

    fn start_detect(&mut self, ctx: &egui::Context, source: DetectSource, label: String) {
        let key = WatermarkKey::new(&self.key);
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();

        self.detect_label = label;
        self.preview = None;
        self.detect_result = None;
        self.detect_job = Some(rx);

        thread::spawn(move || {
            let pixels = match source {
                DetectSource::Pixels(pixels) => pixels,
                DetectSource::File(path) => match io::load(&path) {
                    Ok(loaded) => loaded.pixels,
                    Err(e) => {
                        let _ = tx.send(DetectMsg::Finished(Err(e.to_string())));
                        ctx.request_repaint();
                        return;
                    }
                },
            };
            let _ = tx.send(DetectMsg::Preview(preview_image(&pixels)));
            ctx.request_repaint();
            let found = detect(&pixels, &key, &DetectOptions::default());
            let _ = tx.send(DetectMsg::Finished(Ok(found)));
            ctx.request_repaint();
        });
    }

    fn paste_from_clipboard(&mut self, ctx: &egui::Context) {
        let pasted = arboard::Clipboard::new()
            .and_then(|mut clipboard| clipboard.get_image())
            .ok()
            .and_then(|img| {
                RgbaImage::from_raw(img.width as u32, img.height as u32, img.bytes.into_owned())
            });
        match pasted {
            Some(pixels) => {
                let label = self.t("剪貼簿影像", "Clipboard image").to_owned();
                self.start_detect(ctx, DetectSource::Pixels(pixels), label);
            }
            None => {
                self.detect_label.clear();
                self.preview = None;
                self.detect_result = Some(Err(self
                    .t("剪貼簿裡沒有影像", "No image on the clipboard")
                    .to_owned()));
            }
        }
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.embed_job {
            let mut finished = false;
            for msg in rx.try_iter() {
                match msg {
                    EmbedMsg::Line(line) => {
                        self.log.push(line);
                        self.embed_done += 1;
                    }
                    EmbedMsg::Finished => finished = true,
                }
            }
            if finished {
                self.embed_job = None;
            }
        }
        if let Some(rx) = &self.detect_job {
            let mut finished = false;
            for msg in rx.try_iter() {
                match msg {
                    DetectMsg::Preview(image) => {
                        self.preview =
                            Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR));
                    }
                    DetectMsg::Finished(result) => {
                        self.detect_result = Some(result);
                        finished = true;
                    }
                }
            }
            if finished {
                self.detect_job = None;
            }
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let (dropped, pasted) = ctx.input(|i| {
            let dropped: Vec<PathBuf> = i
                .raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect();
            let pasted = i.events.iter().find_map(|e| match e {
                egui::Event::PasteImage(image) => Some(image.clone()),
                _ => None,
            });
            (dropped, pasted)
        });

        match self.tab {
            Tab::Embed if self.embed_job.is_none() => self.add_paths(dropped),
            Tab::Detect if self.detect_job.is_none() => {
                if let Some(path) = dropped.into_iter().find(|p| io::is_supported(p)) {
                    let label = file_name(&path);
                    self.start_detect(ctx, DetectSource::File(path), label);
                } else if let Some(image) = pasted {
                    let [w, h] = image.size;
                    let bytes = image
                        .pixels
                        .iter()
                        .flat_map(|c| c.to_srgba_unmultiplied())
                        .collect();
                    if let Some(pixels) = RgbaImage::from_raw(w as u32, h as u32, bytes) {
                        let label = self.t("剪貼簿影像", "Clipboard image").to_owned();
                        self.start_detect(ctx, DetectSource::Pixels(pixels), label);
                    }
                }
            }
            _ => {}
        }
    }

    fn key_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(self.t("金鑰", "Key"));
            let hint = self.t("留空則使用內建金鑰", "Empty = built-in key");
            ui.add(
                egui::TextEdit::singleline(&mut self.key)
                    .password(true)
                    .hint_text(hint)
                    .desired_width(220.0),
            );
            ui.weak(self.t(
                "檢測時必須使用與嵌入時相同的金鑰",
                "Detection needs the same key used for embedding",
            ));
        });
    }

    fn embed_tab(&mut self, ui: &mut egui::Ui) {
        let busy = self.embed_job.is_some();
        let ctx = ui.ctx().clone();

        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                if ui.button(self.t("加入照片…", "Add photos…")).clicked() {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter(self.t("影像", "Images"), &io::EXTENSIONS)
                        .pick_files()
                    {
                        self.add_paths(paths);
                    }
                }
                if ui.button(self.t("加入資料夾…", "Add folder…")).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.add_paths([dir]);
                    }
                }
                if ui.button(self.t("清空", "Clear")).clicked() {
                    self.files.clear();
                }
                ui.weak(self.t(
                    "也可以把照片或資料夾拖進視窗",
                    "You can also drop photos or folders here",
                ));
            });

            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("files")
                    .max_height(150.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.files.is_empty() {
                            ui.weak(self.t("尚未加入照片", "No photos yet"));
                        }
                        let mut remove = None;
                        for (i, path) in self.files.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if ui.small_button("×").clicked() {
                                    remove = Some(i);
                                }
                                ui.label(path.display().to_string());
                            });
                        }
                        if let Some(i) = remove {
                            self.files.remove(i);
                        }
                    });
            });

            ui.add_space(6.0);
            self.key_row(ui);

            let id_ok = payload_from_text(&self.id).is_some();
            ui.horizontal(|ui| {
                ui.label(self.t("識別碼", "ID"));
                let hint = self.t("例如 ALLEN-26", "e.g. ALLEN-26");
                ui.add(
                    egui::TextEdit::singleline(&mut self.id)
                        .hint_text(hint)
                        .desired_width(220.0),
                );
                let used = format!("{}/{PAYLOAD_BYTES} bytes", self.id.len());
                if id_ok {
                    ui.weak(used);
                } else {
                    ui.colored_label(RED, used);
                }
                ui.weak(self.t(
                    "（英數字各 1 byte，中文字各 3 bytes）",
                    "(UTF-8; ASCII characters take 1 byte each)",
                ));
            });

            ui.horizontal(|ui| {
                ui.label(self.t("強度", "Strength"));
                ui.add(egui::Slider::new(&mut self.strength, 0.5..=2.5).fixed_decimals(1));
                ui.weak(self.t(
                    "越高越耐裁切與壓縮，但越容易看見",
                    "Higher survives more abuse but is easier to see",
                ));
            });

            ui.horizontal(|ui| {
                ui.label(self.t("輸出格式", "Format"));
                let auto = self.t("依原檔", "Same as source");
                ui.selectable_value(&mut self.format, FormatChoice::Auto, auto);
                ui.selectable_value(&mut self.format, FormatChoice::Jpeg, "JPEG");
                ui.selectable_value(&mut self.format, FormatChoice::Png, "PNG");
                if self.format != FormatChoice::Png {
                    ui.label(self.t("JPEG 品質", "JPEG quality"));
                    ui.add(egui::Slider::new(&mut self.quality, 60..=100));
                }
            });

            ui.horizontal(|ui| {
                ui.label(self.t("輸出位置", "Output"));
                match &self.out_dir {
                    Some(dir) => ui.label(dir.display().to_string()),
                    None => ui.label(self.t(
                        "與原檔相同的資料夾（檔名加上 _wm）",
                        "Next to each source file (name gets _wm)",
                    )),
                };
                if ui.button(self.t("選擇…", "Choose…")).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.out_dir = Some(dir);
                    }
                }
                if self.out_dir.is_some() && ui.button(self.t("還原", "Reset")).clicked() {
                    self.out_dir = None;
                }
            });

            ui.add_space(6.0);
            let ready = !self.files.is_empty() && id_ok;
            let start = egui::Button::new(
                RichText::new(self.t("開始加上浮水印", "Add watermark")).strong(),
            );
            if ui.add_enabled(ready, start).clicked() {
                self.start_embed(&ctx);
            }
        });

        if self.embed_total > 0 {
            ui.add_space(6.0);
            let fraction = self.embed_done as f32 / self.embed_total as f32;
            ui.add(
                egui::ProgressBar::new(fraction)
                    .text(format!("{}/{}", self.embed_done, self.embed_total))
                    .animate(busy),
            );
            egui::ScrollArea::vertical()
                .id_salt("log")
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(line);
                    }
                });
        }
    }

    fn detect_tab(&mut self, ui: &mut egui::Ui) {
        let busy = self.detect_job.is_some();
        let ctx = ui.ctx().clone();

        ui.add_enabled_ui(!busy, |ui| {
            self.key_row(ui);
            ui.horizontal(|ui| {
                if ui.button(self.t("開啟影像…", "Open image…")).clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter(self.t("影像", "Images"), &io::EXTENSIONS)
                        .pick_file()
                    {
                        let label = file_name(&path);
                        self.start_detect(&ctx, DetectSource::File(path), label);
                    }
                }
                let paste = self.t("從剪貼簿貼上 (Ctrl+V)", "Paste from clipboard (Ctrl+V)");
                if ui.button(paste).clicked() {
                    self.paste_from_clipboard(&ctx);
                }
                ui.weak(self.t(
                    "也可以把影像拖進視窗；截圖後直接貼上即可",
                    "Or drop an image here; screenshots can be pasted directly",
                ));
            });
        });
        ui.separator();

        if busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(self.t("分析中…", "Analysing…"));
            });
        }
        match &self.detect_result {
            Some(Ok(found)) if found.detected => {
                ui.label(
                    RichText::new(self.t("偵測到浮水印", "Watermark found"))
                        .size(22.0)
                        .color(GREEN),
                );
                match &found.payload {
                    Some(payload) => {
                        ui.horizontal(|ui| {
                            ui.label(self.t("識別碼：", "ID:"));
                            ui.label(RichText::new(payload_to_text(payload)).size(20.0).strong());
                        });
                    }
                    None => {
                        ui.label(self.t(
                            "訊號存在，但影像損壞過多，無法讀出識別碼",
                            "The signal is there, but the image is too degraded to read the ID",
                        ));
                    }
                }
                let turned = match (found.quarter_turns, found.mirrored) {
                    (0, false) => String::new(),
                    (turns, false) if self.zh => format!("，旋轉 {}°", turns as u32 * 90),
                    (turns, true) if self.zh => format!("，旋轉 {}° 並鏡射", turns as u32 * 90),
                    (turns, false) => format!(", rotated {}°", turns as u32 * 90),
                    (turns, true) => format!(", rotated {}° and mirrored", turns as u32 * 90),
                };
                ui.weak(if self.zh {
                    format!(
                        "信心分數 {:.1}，圖樣週期 {:.1} px{turned}",
                        found.score, found.tile_period
                    )
                } else {
                    format!(
                        "score {:.1}, tile period {:.1} px{turned}",
                        found.score, found.tile_period
                    )
                });
            }
            Some(Ok(found)) => {
                ui.label(
                    RichText::new(self.t("未偵測到浮水印", "No watermark found"))
                        .size(22.0)
                        .color(RED),
                );
                ui.weak(if self.zh {
                    format!(
                        "最高分數 {:.1}。若確定有浮水印，請確認金鑰是否正確。",
                        found.score
                    )
                } else {
                    format!(
                        "best score {:.1}. Check that the key is right.",
                        found.score
                    )
                });
            }
            Some(Err(message)) => {
                ui.colored_label(RED, message);
            }
            None => {}
        }

        if let Some(texture) = &self.preview {
            ui.add_space(6.0);
            ui.weak(&self.detect_label);
            ui.add(
                egui::Image::new(texture)
                    .max_size(ui.available_size())
                    .maintain_aspect_ratio(true),
            );
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_jobs(&ctx);
        self.handle_input(&ctx);

        egui::Panel::top("tabs").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let embed = RichText::new(self.t("加浮水印", "Add watermark")).size(17.0);
                let detect = RichText::new(self.t("檢測浮水印", "Detect watermark")).size(17.0);
                ui.selectable_value(&mut self.tab, Tab::Embed, embed);
                ui.selectable_value(&mut self.tab, Tab::Detect, detect);
            });
            ui.add_space(4.0);
        });
        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Embed => self.embed_tab(ui),
            Tab::Detect => self.detect_tab(ui),
        });
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn preview_image(pixels: &RgbaImage) -> egui::ColorImage {
    let (w, h) = pixels.dimensions();
    let small;
    let shown = if w.max(h) > PREVIEW_SIDE {
        let scale = PREVIEW_SIDE as f32 / w.max(h) as f32;
        let side = |v: u32| ((v as f32 * scale).round() as u32).max(1);
        small = image::imageops::thumbnail(pixels, side(w), side(h));
        &small
    } else {
        pixels
    };
    egui::ColorImage::from_rgba_unmultiplied(
        [shown.width() as usize, shown.height() as usize],
        shown.as_raw(),
    )
}

/// Watermark one file and return where it was written and the PSNR against the original.
fn embed_file(
    input: &Path,
    key: &WatermarkKey,
    payload: &[u8; PAYLOAD_BYTES],
    options: &EmbedOptions,
    format: FormatChoice,
    quality: u8,
    out_dir: Option<&Path>,
) -> Result<(PathBuf, f32), String> {
    let mut loaded = io::load(input).map_err(|e| e.to_string())?;
    let report = embed(&mut loaded.pixels, key, payload, options);

    let is_jpeg = input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["jpg", "jpeg", "jfif"].contains(&e.to_ascii_lowercase().as_str()));
    let format = match format {
        FormatChoice::Auto if is_jpeg => OutputFormat::Jpeg(quality),
        FormatChoice::Jpeg => OutputFormat::Jpeg(quality),
        FormatChoice::Auto | FormatChoice::Png => OutputFormat::Png,
    };
    let output = output_path(input, out_dir, format.extension());
    io::save(
        &output,
        &loaded.pixels,
        format,
        loaded.icc_profile.as_deref(),
    )
    .map_err(|e| e.to_string())?;
    Ok((output, report.psnr))
}

/// `<stem>_wm.<ext>` in `out_dir` (or next to the input), numbered if that name is taken,
/// so neither the source nor an earlier output is ever overwritten.
fn output_path(input: &Path, out_dir: Option<&Path>, extension: &str) -> PathBuf {
    let dir = out_dir
        .or(input.parent())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let stem = input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut candidate = dir.join(format!("{stem}_wm.{extension}"));
    let mut n = 2;
    while candidate.exists() {
        candidate = dir.join(format!("{stem}_wm{n}.{extension}"));
        n += 1;
    }
    candidate
}
