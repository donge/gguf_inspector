/*
 * GGUF Inspector - A fast, dependency-free GGUF file metadata inspector.
 *
 * Author: Gemini
 * Date: 2025-07-02
 *
 * This tool is designed to quickly parse and display metadata from GGUF files
 * without reading the large tensor data section, making it nearly instantaneous
 * even for very large model files.
 *
 * 这是一个快速、无依赖的 GGUF 文件元数据查看工具。
 * 它的设计目标是快速解析并展示 GGUF 文件的元数据，而无需读取庞大的张量数据部分，
 * 因此即使对于非常大的模型文件，也能几乎瞬时完成。
 */

use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::PathBuf;

// GGUF 相关的常量
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

// GGUF 值类型的枚举
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

// GGUF 文件头结构
#[derive(Debug, Serialize)]
struct GGUFHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

// 元数据键值对结构
#[derive(Debug, Serialize, Clone)]
struct MetadataKV {
    key: String,
    value: GGUFValue,
}

// 张量信息结构
#[derive(Debug, Serialize)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    shape: Vec<u64>,
    dtype: String,
    offset: u64,
    size_in_bytes: u64,
}

// 整个 GGUF 文件的元信息容器
#[derive(Debug, Serialize)]
struct GGUFMeta {
    header: GGUFHeader,
    metadata: Vec<MetadataKV>,
    tensors: Vec<TensorInfo>,
}

// 命令行参数定义
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = "一个快速检查 GGUF 文件元数据的工具。")]
struct Args {
    /// GGUF 文件的路径
    #[arg(required = true)]
    file_path: PathBuf,

    /// 只显示文件头信息
    #[arg(long)]
    header: bool,

    /// 只显示元数据
    #[arg(long)]
    metadata: bool,

    /// 只显示张量信息
    #[arg(long)]
    tensors: bool,

    /// 以 JSON 格式输出所有信息
    #[arg(long)]
    json: bool,

    /// [过滤] 只显示键名包含此字符串的元数据
    #[arg(long, value_name = "SUBSTRING")]
    filter_meta: Option<String>,

    /// [过滤] 只显示名称包含此字符串的张量
    #[arg(long, value_name = "SUBSTRING")]
    filter_tensor: Option<String>,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let file = File::open(&args.file_path)?;
    let mut reader = BufReader::new(file);

    let gguf_meta = parse_gguf(&mut reader)?;

    if args.json {
        // 以 JSON 格式输出
        let json_output = serde_json::to_string_pretty(&gguf_meta).unwrap();
        println!("{}", json_output);
    } else {
        // 以人类可读的格式输出
        display_pretty(&gguf_meta, &args);
    }

    Ok(())
}

// 主解析函数
fn parse_gguf<R: Read + Seek>(reader: &mut R) -> io::Result<GGUFMeta> {
    // 1. 解析文件头
    let magic = reader.read_u32::<LittleEndian>()?;

    if magic != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("不是有效的 GGUF 文件 (Magic number 不正确). Expected: {:X}, Got: {:X}", GGUF_MAGIC, magic),
        ));
    }
    let header = GGUFHeader {
        version: reader.read_u32::<LittleEndian>()?,
        tensor_count: reader.read_u64::<LittleEndian>()?,
        metadata_kv_count: reader.read_u64::<LittleEndian>()?,
    };

    // 2. 解析元数据
    let mut metadata = Vec::with_capacity(header.metadata_kv_count as usize);
    for _ in 0..header.metadata_kv_count {
        metadata.push(read_metadata_kv(reader)?);
    }

    // 3. 解析张量信息
    let mut tensors = Vec::with_capacity(header.tensor_count as usize);
    for _ in 0..header.tensor_count {
        tensors.push(read_tensor_info(reader)?);
    }
    
    // 4. 计算张量数据的起始偏移量和对齐填充
    // 这对于验证和完整性检查很有用
    let current_pos = reader.stream_position()?;
    let alignment = match metadata.iter().find(|kv| kv.key == "general.alignment") {
        Some(kv) => match kv.value {
            GGUFValue::U32(val) => val as u64,
            _ => 32,
        },
        None => 32,
    };
    let padding = (alignment - (current_pos % alignment)) % alignment;
    let _tensor_data_offset = current_pos + padding;


    Ok(GGUFMeta {
        header,
        metadata,
        tensors,
    })
}

// 读取 GGUF 字符串
fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buffer = vec![0; len];
    reader.read_exact(&mut buffer)?;
    Ok(String::from_utf8(buffer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?)
}

// 读取一个元数据键值对
fn read_metadata_kv<R: Read>(reader: &mut R) -> io::Result<MetadataKV> {
    let key = read_string(reader)?;
    let value_type = reader.read_u32::<LittleEndian>()?;
    let value = read_value(reader, value_type)?;
    Ok(MetadataKV { key, value })
}

// 根据类型读取一个值
fn read_value<R: Read>(reader: &mut R, value_type: u32) -> io::Result<GGUFValue> {
    match value_type {
        0 => Ok(GGUFValue::U8(reader.read_u8()?)),
        1 => Ok(GGUFValue::I8(reader.read_i8()?)),
        2 => Ok(GGUFValue::U16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GGUFValue::I16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GGUFValue::U32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GGUFValue::I32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GGUFValue::F32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GGUFValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GGUFValue::String(read_string(reader)?)),
        9 => { // Array
            let array_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut array = Vec::with_capacity(len);
            for _ in 0..len {
                array.push(read_value(reader, array_type)?);
            }
            Ok(GGUFValue::Array(array))
        }
        10 => Ok(GGUFValue::U64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GGUFValue::I64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GGUFValue::F64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("未知的 GGUF 值类型: {}", value_type),
        )),
    }
}

// 读取一个张量信息
fn read_tensor_info<R: Read>(reader: &mut R) -> io::Result<TensorInfo> {
    let name = read_string(reader)?;
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = vec![0; n_dims as usize];
    reader.read_u64_into::<LittleEndian>(&mut shape)?;
    let dtype_enum = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    let (dtype, size_in_bytes) = ggml_type_info(dtype_enum, &shape);

    Ok(TensorInfo {
        name,
        n_dims,
        shape,
        dtype,
        offset,
        size_in_bytes,
    })
}

// GGML 类型信息 (名称和大小)
fn ggml_type_info(dtype: u32, shape: &[u64]) -> (String, u64) {
    let type_name = match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        _ => "UNKNOWN",
    };

    // 简化的尺寸计算，对于复杂量化类型可能不完全精确，但可作为参考
    let num_elements = shape.iter().product::<u64>();
    let bits_per_element = match dtype {
        0 => 32,
        1 => 16,
        2 | 3 => 4, // 简化
        6 | 7 => 5, // 简化
        8 | 9 => 8,
        10 => 2, // 简化
        11 => 3, // 简化
        12 => 4, // 简化
        13 => 5, // 简化
        14 => 6, // 简化
        15 => 8,
        _ => 0,
    };
    let size_in_bytes = (num_elements * bits_per_element) / 8;

    (type_name.to_string(), size_in_bytes)
}

// 漂亮地打印信息
fn display_pretty(meta: &GGUFMeta, args: &Args) {
    let show_all = !args.header && !args.metadata && !args.tensors;

    if show_all || args.header {
        println!("--- 文件头 (Header) ---");
        println!("  GGUF 版本: v{}", meta.header.version);
        println!("  张量数量: {}", meta.header.tensor_count);
        println!("  元数据条目数: {}", meta.header.metadata_kv_count);
        println!();
    }

    if show_all || args.metadata {
        println!("--- 元数据 (Metadata) ---");
        let mut sorted_metadata = meta.metadata.clone();
        sorted_metadata.sort_by(|a, b| a.key.cmp(&b.key));

        for kv in sorted_metadata {
            if let Some(filter) = &args.filter_meta {
                if !kv.key.contains(filter) {
                    continue;
                }
            }
            // 为了可读性，对长数组进行截断
            let value_str = match &kv.value {
                GGUFValue::Array(arr) if arr.len() > 8 => {
                    format!("[Array of {} items, first 8: {:?}...]", arr.len(), &arr[..8])
                }
                _ => format!("{:?}", kv.value),
            };
            println!("  - {}: {}", kv.key, value_str);
        }
        println!();
    }

    if show_all || args.tensors {
        println!("--- 张量信息 (Tensors) ---");
        println!(
            "{:<50} | {:<8} | {:<20} | {:<15}",
            "名称", "类型", "形状", "尺寸 (约)"
        );
        println!("{:-<100}", "");
        for t in &meta.tensors {
            if let Some(filter) = &args.filter_tensor {
                if !t.name.contains(filter) {
                    continue;
                }
            }
            let shape_str = format!("{:?}", t.shape);
            let size_hr = format_bytes(t.size_in_bytes);
            println!(
                "{:<50} | {:<8} | {:<20} | {:<15}",
                t.name, t.dtype, shape_str, size_hr
            );
        }
    }
}

// 格式化字节大小为人类可读的字符串
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

