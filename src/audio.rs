//! Streaming encoders for the engine's s16le 24 kHz mono PCM: wav (open-ended RIFF), raw pcm, mp3 (LAME, 64 kbps
//! CBR), Ogg/Opus (libopus VOIP, 20 ms packets, one Ogg page per push) and FLAC (one frame per push). Every `push`
//! returns the bytes that are ready now, so the first audio frame leaves the server as soon as it is generated.

use crate::engine::SR;
use anyhow::{anyhow, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Wav,
    Pcm,
    Mp3,
    Opus,
    Flac,
}

pub const FORMATS: [&str; 5] = ["wav", "pcm", "mp3", "opus", "flac"];

impl Format {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "wav" => Self::Wav,
            "pcm" => Self::Pcm,
            "mp3" => Self::Mp3,
            "opus" | "ogg" => Self::Opus,
            "flac" => Self::Flac,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Pcm => "pcm",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Flac => "flac",
        }
    }
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg; codecs=opus",
            Self::Flac => "audio/flac",
        }
    }
}

pub trait Encoder: Send {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>>;
    fn finish(&mut self) -> Result<Vec<u8>>;
}

pub fn encoder(f: Format) -> Result<Box<dyn Encoder>> {
    Ok(match f {
        Format::Wav => Box::new(Wav { head: true }),
        Format::Pcm => Box::new(Wav { head: false }),
        Format::Mp3 => Box::new(Mp3::new()?),
        Format::Opus => Box::new(Opus::new()?),
        Format::Flac => Box::new(Flac::new()?),
    })
}

fn le_bytes(p: &[i16], out: &mut Vec<u8>) {
    out.reserve(p.len() * 2);
    for s in p {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------------------------- wav / pcm
/// RIFF/WAVE header with 0xFFFFFFFF sizes (a stream of unknown length), s16le mono 24 kHz.
pub fn wav_header() -> Vec<u8> {
    let sr = SR as u32;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&sr.to_le_bytes());
    h.extend_from_slice(&(sr * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h
}

struct Wav {
    head: bool,
}

impl Encoder for Wav {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        let mut out = if std::mem::take(&mut self.head) { wav_header() } else { Vec::new() };
        le_bytes(pcm, &mut out);
        Ok(out)
    }
    fn finish(&mut self) -> Result<Vec<u8>> {
        Ok(if std::mem::take(&mut self.head) { wav_header() } else { Vec::new() })
    }
}

// ---------------------------------------------------------------------------------------------- mp3
struct Mp3(mp3lame_encoder::Encoder);

impl Mp3 {
    fn new() -> Result<Self> {
        use mp3lame_encoder::{Bitrate, Builder, Mode, Quality};
        let mut b = Builder::new().ok_or_else(|| anyhow!("lame init"))?;
        b.set_num_channels(1).map_err(|e| anyhow!("lame: {e:?}"))?;
        b.set_sample_rate(SR as u32).map_err(|e| anyhow!("lame: {e:?}"))?;
        b.set_brate(Bitrate::Kbps64).map_err(|e| anyhow!("lame: {e:?}"))?;
        b.set_mode(Mode::Mono).map_err(|e| anyhow!("lame: {e:?}"))?;
        b.set_quality(Quality::Good).map_err(|e| anyhow!("lame: {e:?}"))?;
        b.set_to_write_vbr_tag(false).map_err(|e| anyhow!("lame: {e:?}"))?;
        Ok(Self(b.build().map_err(|e| anyhow!("lame: {e:?}"))?))
    }
}

impl Encoder for Mp3 {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));
        self.0.encode_to_vec(mp3lame_encoder::MonoPcm(pcm), &mut out).map_err(|e| anyhow!("lame: {e:?}"))?;
        Ok(out)
    }
    fn finish(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(7200);
        self.0.flush_to_vec::<mp3lame_encoder::FlushNoGap>(&mut out).map_err(|e| anyhow!("lame: {e:?}"))?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------- ogg/opus
const OPUS_FRAME: usize = SR / 50; // 20 ms
const SERIAL: u32 = 0x6d617961; // "maya"

struct Opus {
    enc: opus::Encoder,
    ogg: ogg::PacketWriter<'static, Vec<u8>>,
    buf: Vec<i16>,
    granule: u64, // 48 kHz units, includes pre-skip
    head: bool,
    pre_skip: u16,
    packet: Vec<u8>,
}

impl Opus {
    fn new() -> Result<Self> {
        let mut enc = opus::Encoder::new(SR as u32, opus::Channels::Mono, opus::Application::Voip)?;
        enc.set_bitrate(opus::Bitrate::Bits(32_000))?;
        let pre_skip = (enc.get_lookahead()? as u64 * 48_000 / SR as u64) as u16;
        Ok(Self {
            enc,
            ogg: ogg::PacketWriter::new(Vec::new()),
            buf: Vec::new(),
            granule: pre_skip as u64,
            head: true,
            pre_skip,
            packet: vec![0; 4000],
        })
    }

    fn headers(&mut self) -> Result<()> {
        use ogg::PacketWriteEndInfo::EndPage;
        let mut h = b"OpusHead".to_vec();
        h.push(1);
        h.push(1);
        h.extend_from_slice(&self.pre_skip.to_le_bytes());
        h.extend_from_slice(&(SR as u32).to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes());
        h.push(0);
        self.ogg.write_packet(h, SERIAL, EndPage, 0)?;
        let vendor = b"maya-rs";
        let mut t = b"OpusTags".to_vec();
        t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        t.extend_from_slice(vendor);
        t.extend_from_slice(&0u32.to_le_bytes());
        self.ogg.write_packet(t, SERIAL, EndPage, 0)?;
        Ok(())
    }

    /// Encode every whole 20 ms frame in `buf`; the last packet ends the page (or the stream when `last`).
    fn drain(&mut self, last: bool) -> Result<Vec<u8>> {
        use ogg::PacketWriteEndInfo::*;
        if std::mem::take(&mut self.head) {
            self.headers()?;
        }
        if last && self.buf.len() % OPUS_FRAME != 0 {
            let pad = OPUS_FRAME - self.buf.len() % OPUS_FRAME;
            self.buf.extend(std::iter::repeat(0).take(pad));
        }
        let n = self.buf.len() / OPUS_FRAME;
        if last && n == 0 {
            // close the stream with an empty-audio packet so the final page carries the EOS flag
            self.buf.resize(OPUS_FRAME, 0);
            return self.drain(true);
        }
        for i in 0..n {
            let len = self.enc.encode(&self.buf[i * OPUS_FRAME..(i + 1) * OPUS_FRAME], &mut self.packet)?;
            self.granule += (OPUS_FRAME * 48_000 / SR) as u64;
            let end = if i + 1 < n { NormalPacket } else if last { EndStream } else { EndPage };
            self.ogg.write_packet(self.packet[..len].to_vec(), SERIAL, end, self.granule)?;
        }
        self.buf.drain(..n * OPUS_FRAME);
        Ok(std::mem::take(self.ogg.inner_mut()))
    }
}

impl Encoder for Opus {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        self.buf.extend_from_slice(pcm);
        self.drain(false)
    }
    fn finish(&mut self) -> Result<Vec<u8>> {
        self.drain(true)
    }
}

// ---------------------------------------------------------------------------------------------- flac
const FLAC_BLOCK: usize = crate::engine::FRAME; // one 80 ms engine frame per FLAC frame

struct Flac {
    cfg: flacenc::error::Verified<flacenc::config::Encoder>,
    info: flacenc::component::StreamInfo,
    fb: flacenc::source::FrameBuf,
    buf: Vec<i32>,
    n: usize,
    head: bool,
}

impl Flac {
    fn new() -> Result<Self> {
        use flacenc::error::Verify;
        Ok(Self {
            cfg: flacenc::config::Encoder::default().into_verified().map_err(|e| anyhow!("flac config: {e:?}"))?,
            info: flacenc::component::StreamInfo::new(SR, 1, 16).map_err(|e| anyhow!("flac: {e:?}"))?,
            fb: flacenc::source::FrameBuf::with_size(1, FLAC_BLOCK).map_err(|e| anyhow!("flac: {e:?}"))?,
            buf: Vec::new(),
            n: 0,
            head: true,
        })
    }

    /// "fLaC" + a STREAMINFO for a stream of unknown length (0 total samples, no MD5), written by hand because the
    /// frame sizes are not known yet when the header has to leave.
    fn header() -> Vec<u8> {
        let mut h = b"fLaC".to_vec();
        h.extend_from_slice(&[0x80, 0, 0, 34]); // last metadata block, STREAMINFO, 34 bytes
        h.extend_from_slice(&16u16.to_be_bytes()); // min block size (the final block may be short)
        h.extend_from_slice(&(FLAC_BLOCK as u16).to_be_bytes());
        h.extend_from_slice(&[0; 6]); // min/max frame size unknown
        // 20 bits rate, 3 bits channels-1, 5 bits bps-1, 36 bits total samples (0 = unknown)
        let packed: u64 = ((SR as u64) << 44) | (0 << 41) | (15 << 36);
        h.extend_from_slice(&packed.to_be_bytes());
        h.extend_from_slice(&[0; 16]);
        h
    }

    fn drain(&mut self, last: bool) -> Result<Vec<u8>> {
        use flacenc::component::BitRepr;
        use flacenc::source::Fill;
        let mut out = if std::mem::take(&mut self.head) { Self::header() } else { Vec::new() };
        let mut off = 0;
        while self.buf.len() - off >= FLAC_BLOCK || (last && self.buf.len() > off) {
            let len = (self.buf.len() - off).min(FLAC_BLOCK);
            self.fb.resize(len);
            self.fb.fill_interleaved(&self.buf[off..off + len]).map_err(|e| anyhow!("flac: {e:?}"))?;
            let frame = flacenc::encode_fixed_size_frame(&self.cfg, &self.fb, self.n, &self.info)
                .map_err(|e| anyhow!("flac: {e:?}"))?;
            let mut sink = flacenc::bitsink::ByteSink::new();
            frame.write(&mut sink).map_err(|e| anyhow!("flac write: {e:?}"))?;
            out.extend_from_slice(sink.as_slice());
            self.n += 1;
            off += len;
        }
        self.buf.drain(..off);
        Ok(out)
    }
}

impl Encoder for Flac {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        self.buf.extend(pcm.iter().map(|&s| s as i32));
        self.drain(false)
    }
    fn finish(&mut self) -> Result<Vec<u8>> {
        if self.buf.len() % FLAC_BLOCK != 0 && self.buf.len() % FLAC_BLOCK < 16 {
            // FLAC's smallest block is 16 samples: pad the tail with silence
            let pad = 16 - self.buf.len() % FLAC_BLOCK;
            self.buf.extend(std::iter::repeat(0).take(pad));
        }
        self.drain(true)
    }
}
