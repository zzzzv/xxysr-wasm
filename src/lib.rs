mod utils;
mod osu_func;

use wasm_bindgen::prelude::*;
use std::fmt;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// Some miscellaneous stuff:

#[derive(Debug, Clone)]
pub struct BeatMapInfo {
    pub title: String,
    pub title_unicode: Option<String>,
    pub artist: String,
    pub artist_unicode: Option<String>,
    pub creator: String,
    pub version: String,
    pub beatmap_id: u64,
    pub beatmap_set_id: i64,
    pub column_count: u8,
    pub min_bpm: f64,
    pub max_bpm: Option<f64>,
    pub length: u32,
    pub sr: Option<f64>,
    pub note_count: u32,
    pub ln_count: u32,
    pub bg_name: Option<String>, // Not used in formatted display
}

impl fmt::Display for BeatMapInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title_unicode_str = self.title_unicode.as_ref().map_or("".into(), |v| v.clone());
        let artist_unicode_str = self.artist_unicode.as_ref().map_or("".into(), |v| v.clone());
        let title_str = match title_unicode_str.is_empty() {
            true => self.title.clone(),
            false => format!("{} ({})", self.title, title_unicode_str)
        };
        let artist_str = match artist_unicode_str.is_empty() {
            true => self.artist.clone(),
            false => format!("{} ({})", self.artist, artist_unicode_str)
        };
        let bpm_str = match self.max_bpm {
            Some(val) => format!("{}-{}", self.min_bpm, val),
            None => format!("{}", self.min_bpm)
        };
        let length_str = format!("{}:{:02}.{:03}", self.length / 60000, (self.length % 60000) / 1000, self.length % 1000);
        
        let sr_str = self.sr.map_or("N/A".into(), |v| format!("{:.4}", v));
        let ln_ratio = self.ln_count as f64 /  (self.ln_count + self.note_count) as f64;
        
        write!(
            f,
            "Title: {}\nArtist: {}\nCreator: {}\nVersion: {}\nBeatmapID: {}\nBeatmapSetID: {}\nColumns: {}\nBPM: {}\nLength: {}\nSR: {}\nLN_Ratio: {:.3}",
            title_str, artist_str, self.creator, self.version, self.beatmap_id, self.beatmap_set_id, self.column_count, bpm_str, length_str, sr_str, ln_ratio
        )
    }
}

#[wasm_bindgen]
pub fn calc_sr(beatmap_text: &str, speed: f64) -> Result<f64, JsValue> {
    utils::set_panic_hook();
    match osu_func::calculate_from_text(beatmap_text, speed) {
        Ok(sr) => Ok(sr),
        Err(e) => Err(JsValue::from_str(&format!("Error: {}", e))),
    }
}