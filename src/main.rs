use anyhow::{Context, anyhow};
use dicom::core::smallvec::SmallVec;
use dicom::pixeldata::PixelDecoder;
use dicom::{
    self,
    core::{DataElement, PrimitiveValue, VR},
    dictionary_std::tags,
    object::{FileDicomObject, InMemDicomObject},
};
use eframe::{
    egui,
    egui::{ColorImage, Pos2, Rect, Sense, Stroke, Vec2},
};
use image::{DynamicImage, ImageBuffer, Luma};
use std::path::PathBuf;

type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(Debug)]
enum DCMRedactErrors {
    ValueError(String),
}

fn write_dynamic_image_to_dicom(
    file_obj: &mut dicom::object::FileDicomObject<InMemDicomObject>,
    img: &Gray16Image,
    save_path: &PathBuf,
) {
    let raw_u16 = SmallVec::from_vec(img.to_vec());

    file_obj.put(DataElement::new(
        tags::BITS_ALLOCATED,
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    file_obj.put(DataElement::new(
        tags::BITS_STORED,
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    file_obj.put(DataElement::new(
        tags::HIGH_BIT,
        VR::US,
        PrimitiveValue::from(15u16),
    ));
    file_obj.put(DataElement::new(
        tags::PIXEL_REPRESENTATION,
        VR::US,
        PrimitiveValue::from(0u16),
    )); // unsigned
    file_obj.put(DataElement::new(
        tags::PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    file_obj.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OW,
        PrimitiveValue::U16(raw_u16),
    ));

    let _ = file_obj.write_to_file(save_path);
}

/// Turn pixels in the given (x0..x1, y0..y1) rectangle to black (in-place).
fn blacken_rect(img: &mut Gray16Image, x0: u32, y0: u32, x1: u32, y1: u32) {
    let (w, h) = img.dimensions();
    let x0 = x0.min(w.saturating_sub(1));
    let y0 = y0.min(h.saturating_sub(1));
    let x1 = x1.min(w);
    let y1 = y1.min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, Luma([0u16]));
        }
    }
}

fn dynamic_to_color_image(img: &DynamicImage) -> ColorImage {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for p in rgba.pixels() {
        pixels.push(egui::Color32::from_rgba_unmultiplied(
            p[0], p[1], p[2], p[3],
        ));
    }
    ColorImage {
        size: [w as usize, h as usize],
        pixels,
    }
}

struct App {
    // Source image (mutable for edits)
    gray_img: Option<Gray16Image>,
    // GPU texture + CPU copy for drawing
    color_img: Option<ColorImage>,
    tex: Option<egui::TextureHandle>,

    // For drag-to-select
    drag_start_px: Option<[u32; 2]>,
    drag_start_screen: Option<Pos2>,
    drag_current_screen: Option<Pos2>,

    // Bookkeeping
    opened_path: Option<PathBuf>,
    fit_scale: f32, // UI zoom to fit (1.0 = native)
    is_dcm: bool,
    dcm: Option<FileDicomObject<InMemDicomObject>>,
    last_error: Option<String>,
    photometric_interpretation: Option<String>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            gray_img: None,
            color_img: None,
            tex: None,
            drag_start_px: None,
            drag_start_screen: None,
            drag_current_screen: None,
            opened_path: None,
            fit_scale: 1.0,
            is_dcm: false,
            dcm: None,
            last_error: None,
            photometric_interpretation: None,
        }
    }

    fn load_dcm(&mut self, path: &PathBuf) -> Result<DynamicImage, DCMRedactErrors> {
        // Try to open the DICOM file
        let file = dicom::object::open_file(&path)
            .map_err(|e| DCMRedactErrors::ValueError(format!("Failed to open DICOM file: {e}")))?;
        self.dcm = Some(file);
        self.is_dcm = true;

        if let Some(dcm) = self.dcm.as_ref() {
            // Check Bits Allocated
            let bits_allocated: u16 = dcm
                .element(tags::BITS_ALLOCATED)
                .map_err(|_| DCMRedactErrors::ValueError("Missing BITS_ALLOCATED tag".to_string()))?
                .to_int()
                .map_err(|_| {
                    DCMRedactErrors::ValueError("Invalid BITS_ALLOCATED value".to_string())
                })?;

            if bits_allocated != 16u16 && bits_allocated != 12u16 {
                return Err(DCMRedactErrors::ValueError(format!(
                    "Mismatched BITS_ALLOCATED, expected 16 got {bits_allocated}"
                )));
            }

            // Check Photometric Interpretation
            self.photometric_interpretation = Some(
                dcm.element(tags::PHOTOMETRIC_INTERPRETATION)
                    .map_err(|_| {
                        DCMRedactErrors::ValueError(
                            "Missing PHOTOMETRIC_INTERPRETATION tag".to_string(),
                        )
                    })?
                    .to_str()
                    .map_err(|_| {
                        DCMRedactErrors::ValueError(
                            "Invalid PHOTOMETRIC_INTERPRETATION value (not UTF-8)".to_string(),
                        )
                    })?
                    .into_owned(),
            );

            if let Some(v) = self.photometric_interpretation.as_ref() {
                if v != "MONOCHROME1" && v != "MONOCHROME2" {
                    return Err(DCMRedactErrors::ValueError(format!(
                        "Mismatched PHOTOMETRIC_INTERPRETATION, expected MONOCHROME1 or MONOCHROME2 got {v}"
                    )));
                }
            }
        }

        // Decode pixel data safely
        let dyn_img = self
            .dcm
            .as_mut()
            .ok_or_else(|| DCMRedactErrors::ValueError("Missing DICOM object".to_string()))?
            .decode_pixel_data()
            .map_err(|e| DCMRedactErrors::ValueError(format!("Failed to decode pixel data: {e}")))?
            .to_dynamic_image(0)
            .map_err(|e| {
                DCMRedactErrors::ValueError(format!("Failed to convert to DynamicImage: {e}"))
            })?;

        Ok(dyn_img)
    }

    fn load_image(&mut self, ctx: &egui::Context, path: PathBuf) -> anyhow::Result<()> {
        // Try to load without mutating state first
        let dyn_img = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("dcm") => {
                // Your load_dcm returns Result<DynamicImage, DCMRedactErrors>
                let img = self
                    .load_dcm(&path)
                    .map_err(|e| anyhow!("Invalid DICOM: {:?}", e))?;
                // load_dcm can set flags like is_dcm, but if you want to avoid partial state,
                // you can set them after success below.
                img
            }
            _ => {
                // Non-DICOM path
                let img = image::open(&path)
                    .with_context(|| format!("Failed to open image: {}", path.display()))?;
                img
            }
        };

        // If we got here, it's safe to update state
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("dcm"))
            .unwrap_or(false)
        {
            self.is_dcm = true;
        } else {
            self.is_dcm = false;
            self.dcm = None;
        }

        let gray_img = dyn_img.to_luma16();
        let color_img = dynamic_to_color_image(&dyn_img);

        self.gray_img = Some(gray_img);
        self.color_img = Some(color_img.clone());
        self.tex = Some(ctx.load_texture("image", color_img, egui::TextureOptions::LINEAR));
        self.opened_path = Some(path);
        self.fit_scale = 1.0;

        Ok(())
    }

    fn refresh_texture(&mut self, ctx: &egui::Context) {
        if let (Some(ci), Some(tex)) = (self.color_img.as_ref(), self.tex.as_mut()) {
            tex.set(ci.clone(), egui::TextureOptions::LINEAR);
        } else if let Some(ci) = self.color_img.clone() {
            self.tex = Some(ctx.load_texture("image", ci, egui::TextureOptions::LINEAR));
        }
    }

    fn apply_blacken(&mut self, rect_px: [u32; 4], ctx: &egui::Context) {
        if let (Some(img), Some(ci)) = (self.gray_img.as_mut(), self.color_img.as_mut()) {
            blacken_rect(img, rect_px[0], rect_px[1], rect_px[2], rect_px[3]);

            // Mirror to egui::ColorImage
            let (w, h) = img.dimensions();
            debug_assert_eq!(ci.size, [w as usize, h as usize]);
            for (i, p) in img.pixels().enumerate() {
                if p[0] == 0 {
                    ci.pixels[i] = egui::Color32::from_gray(0u8);
                }
            }
            self.refresh_texture(ctx);
        }
    }

    /// Map a screen point to image pixel coordinates, clamped, given the on-screen rect of the image.
    fn screen_to_pixel(&self, img_rect: Rect, p: Pos2) -> Option<[u32; 2]> {
        let (w, h) = match self.gray_img.as_ref() {
            Some(i) => i.dimensions(),
            None => return None,
        };
        if !img_rect.contains(p) {
            return None;
        }

        let uv = (
            (p.x - img_rect.left()) / img_rect.width(),
            (p.y - img_rect.top()) / img_rect.height(),
        );
        let x = (uv.0 * w as f32).floor().clamp(0.0, (w - 1) as f32) as u32;
        let y = (uv.1 * h as f32).floor().clamp(0.0, (h - 1) as f32) as u32;
        Some([x, y])
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open Image…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "tiff", "tif", "dcm"])
                        .pick_file()
                    {
                        if let Err(e) = self.load_image(ctx, path) {
                            self.last_error = Some(e.to_string());
                        }
                    }
                }
                if self.last_error.is_some() {
                    let mut dismiss = false;

                    egui::Window::new("Error")
                        .collapsible(false)
                        .resizable(false)
                        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                        .show(ctx, |ui| {
                            // Borrow immutably only inside this closure
                            if let Some(err) = &self.last_error {
                                ui.label(err);
                            }
                            // Record the click; do not mutate here
                            if ui.button("OK").clicked() {
                                dismiss = true;
                            }
                        });

                    // Now mutate after the borrow ends
                    if dismiss {
                        self.last_error = None;
                    }
                }
                if ui.button("Save As…").clicked() {
                    if let (Some(img), Some(path)) =
                        (self.gray_img.as_ref(), self.opened_path.clone())
                    {
                        let file_name = path.file_name().unwrap().to_owned().into_string().unwrap();
                        //let mut new_path = path.parent().unwrap().to_path_buf();

                        //new_path.push("redacted");
                        //new_path.push(file_name);
                        //
                        // let default = new_path.into_os_string().into_string().unwrap();

                        if let Some(out) =
                            rfd::FileDialog::new().set_file_name(file_name).save_file()
                        {
                            if self.is_dcm {
                                if let Some(dcm) = self.dcm.as_mut() {
                                    write_dynamic_image_to_dicom(dcm, img, &out);
                                }
                            } else {
                                let _ = img.save(out);
                            }
                        }
                    }
                }
                if ui.button("Reset").clicked() {
                    if let Err(e) = self.load_image(ctx, self.opened_path.as_ref().unwrap().clone())
                    {
                        eprintln!("Failed to open image: {e}");
                    }
                }
                ui.add(egui::Slider::new(&mut self.fit_scale, 0.1..=5.0).text("Zoom"));
                ui.label("Drag to draw a box; release to blacken.");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                if let (Some(tex), Some(ci)) = (self.tex.as_ref(), self.color_img.as_ref()) {
                    let img_size = Vec2::new(ci.size[0] as f32, ci.size[1] as f32) * self.fit_scale;
                    let response = ui.add(
                        egui::Image::from_texture((tex.id(), img_size))
                            .sense(Sense::click_and_drag()),
                    );
                    let img_rect = response.rect;

                    // --- Zoom handling (scroll wheel / pinch) ---
                    let zoom_speed: f32 = 1.0;

                    // Pinch gesture support (touchpads, trackpads)
                    let pinch_factor = ctx.input(|i| i.zoom_delta());
                    if response.hovered() && pinch_factor != 1.0 {
                        self.fit_scale = (self.fit_scale * pinch_factor).clamp(0.05, 20.0);
                        ctx.request_repaint();
                    }

                    // Mouse Wheel zoom
                    let wheel_zoom = ctx.input(|i| {
                        if response.hovered() {
                            // if i.modifiers.ctrl && response.hovered() {
                            let dy = i.smooth_scroll_delta.y;
                            if dy != 0.0 {
                                let factor = (dy / 240.0 * zoom_speed).exp();
                                Some(factor)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });

                    if let Some(factor) = wheel_zoom {
                        self.fit_scale = (self.fit_scale * factor).clamp(0.05, 20.0);
                        ctx.request_repaint();
                    }
                    // --- end zoom handling ---

                    // Handle mouse interactions over the image
                    if response.hovered() || response.dragged() || response.clicked() {
                        if response.drag_started() {
                            if let Some(px) = self
                                .screen_to_pixel(img_rect, response.interact_pointer_pos().unwrap())
                            {
                                self.drag_start_px = Some(px);
                                self.drag_start_screen = response.interact_pointer_pos();
                                self.drag_current_screen = self.drag_start_screen;
                            }
                        }
                        if response.dragged() {
                            self.drag_current_screen = response.interact_pointer_pos();
                        }
                        if response.drag_stopped() {
                            if let (Some(start_px), Some(curr_screen)) =
                                (self.drag_start_px.take(), self.drag_current_screen.take())
                            {
                                // Convert end screen pos to pixels
                                if let Some(end_px) = self.screen_to_pixel(img_rect, curr_screen) {
                                    let x0 = start_px[0].min(end_px[0]);
                                    let y0 = start_px[1].min(end_px[1]);
                                    let x1 = start_px[0].max(end_px[0]) + 1; // make end exclusive
                                    let y1 = start_px[1].max(end_px[1]) + 1;
                                    self.apply_blacken([x0, y0, x1, y1], ctx);
                                }
                            }
                            self.drag_start_screen = None;
                        }
                    }

                    // Draw temporary selection rectangle overlay
                    if let (Some(p0), Some(p1)) = (self.drag_start_screen, self.drag_current_screen)
                    {
                        let rect = Rect::from_two_pos(p0, p1);
                        let painter = ui.painter();
                        painter.rect_stroke(rect, 0.0, Stroke::new(2.0, egui::Color32::YELLOW));
                    }
                } else {
                    ui.label("Click “Open Image…” to begin.");
                }
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default(),
        ..Default::default()
    };
    eframe::run_native(
        "Box Blackout (drag to blacken)",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
