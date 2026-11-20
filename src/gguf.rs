//! Minimal GGUF (v2/v3) reader over a read-only mmap.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path};

use crate::quant::GgmlType;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) => v as u64,
            Value::U64(v) => v,
            Value::I64(v) => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset into the file.
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("gguf: overflow")?;
        if end > self.buf.len() {
            bail!("gguf: truncated header");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => bail!("gguf: unknown value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the model file is treated as immutable for the process lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { buf: &mmap, pos: 0 };
        if c.take(4)? != b"GGUF" {
            bail!("not a GGUF file");
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            meta.insert(k, c.value(t)?);
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
            let ty = GgmlType::from_u32(c.u32()?)?;
            let off = c.u64()? as usize;
            infos.push((name, dims, ty, off));
        }
        let align = meta
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32) as usize;
        let data_start = c.pos.div_ceil(align) * align;
        let mut tensors = HashMap::new();
        for (name, dims, ty, off) in infos {
            let numel: u64 = dims.iter().product();
            let nbytes = ty.row_bytes(numel as usize);
            let offset = data_start + off;
            if offset + nbytes > mmap.len() {
                bail!("gguf: tensor {name} out of bounds");
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ty,
                    offset,
                    nbytes,
                },
            );
        }
        Ok(Self {
            mmap,
            meta,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("gguf: missing tensor {name}"))
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.nbytes]
    }

    pub fn meta_u64(&self, key: &str) -> Result<u64> {
        self.meta
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("gguf: missing/invalid {key}"))
    }

    pub fn meta_f64(&self, key: &str) -> Result<f64> {
        self.meta
            .get(key)
            .and_then(Value::as_f64)
            .with_context(|| format!("gguf: missing/invalid {key}"))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        self.meta
            .get(key)
            .and_then(Value::as_str)
            .with_context(|| format!("gguf: missing/invalid {key}"))
    }

    pub fn meta_array(&self, key: &str) -> Result<&[Value]> {
        self.meta
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("gguf: missing/invalid {key}"))
    }
}
