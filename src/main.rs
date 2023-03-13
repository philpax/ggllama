use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    ptr::NonNull,
    sync::Mutex,
};

use anyhow::Context;
use clap::Parser;
use ggml_sys::{
    ggml_add, ggml_blck_size, ggml_build_forward_expand, ggml_cgraph, ggml_context, ggml_cpy,
    ggml_diag_mask_inf, ggml_element_size, ggml_free, ggml_get_data, ggml_get_rows,
    ggml_graph_compute, ggml_init, ggml_init_params, ggml_mul, ggml_mul_mat, ggml_nbytes,
    ggml_nelements, ggml_new_f32, ggml_new_tensor_1d, ggml_new_tensor_2d, ggml_new_tensor_3d,
    ggml_norm, ggml_permute, ggml_repeat, ggml_reshape_3d, ggml_rope, ggml_scale, ggml_silu,
    ggml_soft_max, ggml_tensor, ggml_time_init, ggml_time_us, ggml_type_GGML_TYPE_F16,
    ggml_type_GGML_TYPE_F32, ggml_type_GGML_TYPE_I32, ggml_type_GGML_TYPE_Q4_0,
    ggml_type_GGML_TYPE_Q4_1, ggml_type_size, ggml_type_sizef, ggml_used_mem, ggml_view_1d,
};
use once_cell::sync::Lazy;
use rand::SeedableRng;
use utils::{llama_sample_top_p_top_k, llama_tokenize, GptParams, GptVocab, GptVocabId};

mod utils;

static LLAMA_N_PARTS: Lazy<HashMap<u32, u32>> =
    Lazy::new(|| HashMap::from_iter([(4096, 1), (5120, 2), (6656, 4), (8192, 8)]));

struct LlamaHParams {
    n_vocab: i32,
    // this is provided as user input?
    n_ctx: i32,
    n_embd: i32,
    n_mult: i32,
    n_head: i32,
    n_layer: i32,
    n_rot: i32,
    f16: i32,
}
impl Default for LlamaHParams {
    fn default() -> Self {
        Self {
            n_vocab: 32000,
            n_ctx: 512,
            n_embd: 4096,
            n_mult: 256,
            n_head: 32,
            n_layer: 32,
            n_rot: 64,
            f16: 1,
        }
    }
}

#[allow(dead_code)]
struct LlamaLayer {
    // normalization
    attention_norm: NonNull<ggml_tensor>,

    // attention
    wq: NonNull<ggml_tensor>,
    wk: NonNull<ggml_tensor>,
    wv: NonNull<ggml_tensor>,
    wo: NonNull<ggml_tensor>,

    // normalization
    ffn_norm: NonNull<ggml_tensor>,

    // ff
    w1: NonNull<ggml_tensor>,
    w2: NonNull<ggml_tensor>,
    w3: NonNull<ggml_tensor>,
}

#[allow(dead_code)]
struct LlamaModel {
    hparams: LlamaHParams,

    tok_embeddings: NonNull<ggml_tensor>,

    norm: NonNull<ggml_tensor>,
    output: NonNull<ggml_tensor>,

    layers: Vec<LlamaLayer>,

    memory_k: NonNull<ggml_tensor>,
    memory_v: NonNull<ggml_tensor>,

    ctx: NonNull<ggml_context>,
    tensors: HashMap<String, NonNull<ggml_tensor>>,
}
impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { ggml_free(self.ctx.as_ptr()) }
    }
}
impl LlamaModel {
    fn load(fname: &Path, vocab: &mut GptVocab, n_ctx: i32) -> anyhow::Result<LlamaModel> {
        const PRINT_LAYERS: bool = false;

        log::info!("loading model from {fname:?} - please wait ...");

        fn read_i32(f: &mut File) -> std::io::Result<Option<i32>> {
            let mut out = [0u8; 4];
            if f.read(&mut out)? == 0 {
                return Ok(None);
            };
            Ok(Some(i32::from_le_bytes(out)))
        }

        fn read_u32(f: &mut File) -> std::io::Result<Option<u32>> {
            let mut out = [0u8; 4];
            if f.read(&mut out)? == 0 {
                return Ok(None);
            };
            Ok(Some(u32::from_le_bytes(out)))
        }

        fn read_string_with_len(f: &mut File, len: usize) -> anyhow::Result<Option<String>> {
            let mut string_buf = vec![0u8; len];
            if f.read(&mut string_buf)? == 0 {
                return Ok(None);
            };

            Ok(Some(String::from_utf8(string_buf)?))
        }

        fn read_string(f: &mut File) -> anyhow::Result<Option<String>> {
            let len = usize::try_from(read_u32(f)?.context("eof while reading string")?)?;
            if len == 0 {
                return Ok(Some(String::new()));
            }
            read_string_with_len(f, len)
        }

        fn read_into_pointer(f: &mut File, ptr: *mut u8, len: u64) -> anyhow::Result<()> {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len.try_into()?) };
            let read_len = f.read(slice)?;
            assert_eq!(read_len, len as usize);
            Ok(())
        }

        let mut fin = std::fs::File::open(fname)?;

        {
            if read_u32(&mut fin)?.context("eof while reading magic")? != 0x67676d6c {
                anyhow::bail!("invalid model file {fname:?} (bad magic)");
            }
        }

        let n_ff: i32;
        let n_parts: u32;

        // load hparams
        let hparams = {
            let n_vocab = read_i32(&mut fin)?.context("eof reading n_vocab")?;
            let n_embd = read_i32(&mut fin)?.context("eof reading n_embd")?;
            let n_mult = read_i32(&mut fin)?.context("eof reading n_mult")?;
            let n_head = read_i32(&mut fin)?.context("eof reading n_head")?;
            let n_layer = read_i32(&mut fin)?.context("eof reading n_layer")?;
            let n_rot = read_i32(&mut fin)?.context("eof reading n_rot")?;
            let f16 = read_i32(&mut fin)?.context("eof reading f16")?;

            let hparams = LlamaHParams {
                n_vocab,
                n_ctx,
                n_embd,
                n_mult,
                n_head,
                n_layer,
                n_rot,
                f16,
            };

            n_ff = ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult)
                * hparams.n_mult;
            n_parts = LLAMA_N_PARTS
                .get(&(hparams.n_embd as u32))
                .copied()
                .context("invalid embed for n_parts")?;

            log::info!("n_vocab = {}", hparams.n_vocab);
            log::info!("n_ctx   = {}", hparams.n_ctx);
            log::info!("n_embd  = {}", hparams.n_embd);
            log::info!("n_mult  = {}", hparams.n_mult);
            log::info!("n_head  = {}", hparams.n_head);
            log::info!("n_layer = {}", hparams.n_layer);
            log::info!("n_rot   = {}", hparams.n_rot);
            log::info!("f16     = {}", hparams.f16);
            log::info!("n_ff    = {}", n_ff);
            log::info!("n_parts = {}", n_parts);

            hparams
        };

        // load vocab
        {
            let n_vocab = hparams.n_vocab;

            for i in 0..n_vocab {
                let word = read_string(&mut fin)?.context("eof while reading vocab")?;
                vocab.token_to_id.insert(word.clone(), i);
                vocab.id_to_token.insert(i, word.clone());
            }
        }

        let wtype = match hparams.f16 {
            0 => ggml_type_GGML_TYPE_F32,
            1 => ggml_type_GGML_TYPE_F16,
            2 => ggml_type_GGML_TYPE_Q4_0,
            3 => ggml_type_GGML_TYPE_Q4_1,
            _ => {
                anyhow::bail!(
                    "invalid model file {fname:?} (bad f16 value {})",
                    hparams.f16,
                );
            }
        };

        let mut ctx_size: usize = 0;

        {
            let n_embd = hparams.n_embd as f32;
            let n_layer = hparams.n_layer as f32;
            let n_ctx = hparams.n_ctx as f32;
            let n_vocab = hparams.n_vocab as f32;
            let n_ff = n_ff as f32;

            let wtype_sizef = unsafe { ggml_type_sizef(wtype) };
            let f32_sizef = unsafe { ggml_type_sizef(ggml_type_GGML_TYPE_F32) };

            {
                ctx_size += (n_embd * n_vocab * wtype_sizef) as usize; // tok_embeddings

                ctx_size += (n_embd * f32_sizef) as usize; // norm

                ctx_size += (n_embd * n_vocab * wtype_sizef) as usize; // output

                ctx_size += (n_layer * (n_embd * f32_sizef)) as usize; // attention_norm

                ctx_size += (n_layer * (n_embd * n_embd * wtype_sizef)) as usize; // wq
                ctx_size += (n_layer * (n_embd * n_embd * wtype_sizef)) as usize; // wk
                ctx_size += (n_layer * (n_embd * n_embd * wtype_sizef)) as usize; // wv
                ctx_size += (n_layer * (n_embd * n_embd * wtype_sizef)) as usize; // wo

                ctx_size += (n_layer * (n_embd * f32_sizef)) as usize; // ffn_norm

                ctx_size += (n_layer * (n_ff * n_embd * wtype_sizef)) as usize; // w1
                ctx_size += (n_layer * (n_ff * n_embd * wtype_sizef)) as usize; // w2
                ctx_size += (n_layer * (n_ff * n_embd * wtype_sizef)) as usize; // w3

                ctx_size += (n_ctx * n_layer * n_embd * f32_sizef) as usize; // memory_k
                ctx_size += (n_ctx * n_layer * n_embd * f32_sizef) as usize; // memory_v

                ctx_size += ((5 + 10 * hparams.n_layer) * 256) as usize; // object overhead
            }

            log::info!("ggml ctx size = {} MB", ctx_size as f32 / (1024.0 * 1024.0));
        }

        let ctx = NonNull::new(unsafe {
            ggml_init(ggml_init_params {
                mem_size: ctx_size,
                mem_buffer: std::ptr::null_mut(),
            })
        })
        .context("ggml_init() failed")?;

        let (layers, tok_embeddings, norm, output, tensors) = {
            let n_embd = hparams.n_embd;
            let n_layer = hparams.n_layer;
            // let n_ctx = hparams.n_ctx;
            let n_vocab = hparams.n_vocab;

            let mut layers = vec![];

            let tok_embeddings =
                NonNull::new(unsafe { ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_vocab) })
                    .context("failed to create tok_embeddings")?;

            let norm = NonNull::new(unsafe {
                ggml_new_tensor_1d(ctx.as_ptr(), ggml_type_GGML_TYPE_F32, n_embd)
            })
            .context("failed to create norm")?;
            let output =
                NonNull::new(unsafe { ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_vocab) })
                    .context("failed to create output")?;

            // map by name
            let mut tensors = HashMap::default();
            tensors.insert("tok_embeddings.weight".to_string(), tok_embeddings);

            tensors.insert("norm.weight".to_string(), norm);
            tensors.insert("output.weight".to_string(), output);

            for i in 0..n_layer {
                let attention_norm = NonNull::new(unsafe {
                    ggml_new_tensor_1d(ctx.as_ptr(), ggml_type_GGML_TYPE_F32, n_embd)
                })
                .context("failed to create attention_norm")?;

                let wq = NonNull::new(unsafe {
                    ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_embd)
                })
                .context("failed to create wq")?;
                let wk = NonNull::new(unsafe {
                    ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_embd)
                })
                .context("failed to create wk")?;
                let wv = NonNull::new(unsafe {
                    ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_embd)
                })
                .context("failed to create wv")?;
                let wo = NonNull::new(unsafe {
                    ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_embd)
                })
                .context("failed to create wo")?;

                let ffn_norm = NonNull::new(unsafe {
                    ggml_new_tensor_1d(ctx.as_ptr(), ggml_type_GGML_TYPE_F32, n_embd)
                })
                .context("failed to create ffn_norm")?;

                let w1 =
                    NonNull::new(unsafe { ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_ff) })
                        .context("failed to create w1")?;
                let w2 =
                    NonNull::new(unsafe { ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_ff, n_embd) })
                        .context("failed to create w2")?;
                let w3 =
                    NonNull::new(unsafe { ggml_new_tensor_2d(ctx.as_ptr(), wtype, n_embd, n_ff) })
                        .context("failed to create w3")?;

                // map by name
                tensors.insert(format!("layers.{i}.attention_norm.weight"), attention_norm);

                tensors.insert(format!("layers.{i}.attention.wq.weight"), wq);
                tensors.insert(format!("layers.{i}.attention.wk.weight"), wk);
                tensors.insert(format!("layers.{i}.attention.wv.weight"), wv);
                tensors.insert(format!("layers.{i}.attention.wo.weight"), wo);

                tensors.insert(format!("layers.{i}.ffn_norm.weight"), ffn_norm);

                tensors.insert(format!("layers.{i}.feed_forward.w1.weight"), w1);
                tensors.insert(format!("layers.{i}.feed_forward.w2.weight"), w2);
                tensors.insert(format!("layers.{i}.feed_forward.w3.weight"), w3);

                layers.push(LlamaLayer {
                    attention_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    ffn_norm,
                    w1,
                    w2,
                    w3,
                })
            }

            (layers, tok_embeddings, norm, output, tensors)
        };

        // key + value memory
        let (memory_k, memory_v) = {
            let n_embd = hparams.n_embd;
            let n_layer = hparams.n_layer;
            let n_ctx = hparams.n_ctx;

            let n_mem = n_layer * n_ctx;
            let n_elements = n_embd * n_mem;

            let memory_k = NonNull::new(unsafe {
                ggml_new_tensor_1d(ctx.as_ptr(), ggml_type_GGML_TYPE_F32, n_elements)
            })
            .context("failed to create memory_k")?;
            let memory_v = NonNull::new(unsafe {
                ggml_new_tensor_1d(ctx.as_ptr(), ggml_type_GGML_TYPE_F32, n_elements)
            })
            .context("failed to create memory_v")?;

            let memory_size =
                unsafe { ggml_nbytes(memory_k.as_ptr()) + ggml_nbytes(memory_v.as_ptr()) };

            log::info!(
                "memory_size = {} MB, n_mem = {}",
                (memory_size as f32) / 1024.0 / 1024.0,
                n_mem,
            );

            (memory_k, memory_v)
        };

        let file_offset = fin.stream_position()?;
        std::mem::drop(fin);

        for i in 0..n_parts {
            let part_id = i;

            let mut fname_part = fname.to_string_lossy().to_string();
            if i > 0 {
                fname_part += &format!(".{i}");
            }
            let fname_part = Path::new(&fname_part);

            log::info!(
                "loading model part {}/{} from {:?}",
                i + 1,
                n_parts,
                fname_part
            );

            let mut fin = std::fs::File::open(Path::new(fname_part))?;
            fin.seek(std::io::SeekFrom::Start(file_offset))?;

            // load weights
            {
                let mut n_tensors = 0;
                let mut total_size = 0;

                loop {
                    let (n_dims, length, ftype) = match (
                        read_i32(&mut fin)?,
                        read_i32(&mut fin)?,
                        read_i32(&mut fin)?,
                    ) {
                        (Some(n_dims), Some(length), Some(ftype)) => (n_dims, length, ftype),
                        _ => break,
                    };

                    let mut nelements = 1;
                    let mut ne = [1, 1];
                    for e in ne.iter_mut().take(usize::try_from(n_dims)?) {
                        *e = read_i32(&mut fin)?.context("eof while reading ne")?;
                        nelements *= *e;
                    }

                    let name = read_string_with_len(&mut fin, length.try_into()?)?
                        .context("eof while reading name with len")?;

                    if !tensors.contains_key(&name) {
                        anyhow::bail!("unknown tensor '{name}' in model_file");
                    }

                    // split_type = 0: split by columns
                    // split_type = 1: split by rows
                    let mut split_type = 0;

                    // split_type = 0:
                    // regex:
                    //   - tok_embeddings.*
                    //   - layers.*.attention.wo.weight
                    //   - layers.*.feed_forward.w2.weight

                    // split_type = 1:
                    // regex:
                    //   - output.*
                    //   - layers.*.attention.wq.weight
                    //   - layers.*.attention.wk.weight
                    //   - layers.*.attention.wv.weight
                    //   - layers.*.feed_forward.w1.weight
                    //   - layers.*.feed_forward.w3.weight

                    if name.contains("tok_embeddings") {
                        split_type = 0;
                    } else if name.contains("layers") {
                        if name.contains("attention.wo.weight")
                            || name.contains("feed_forward.w2.weight")
                        {
                            split_type = 0;
                        } else {
                            split_type = 1;
                        }
                    } else if name.contains("output") {
                        split_type = 1;
                    }

                    let mut tensor = *tensors.get(&name).unwrap();

                    if n_dims == 1 {
                        if unsafe { ggml_nelements(tensor.as_ptr()) } != nelements {
                            anyhow::bail!("tensor {} has wrong size in model file", name);
                        }
                    } else if unsafe { ggml_nelements(tensor.as_ptr()) } / (n_parts as i32)
                        != nelements
                    {
                        anyhow::bail!("tensor {} has wrong size in model file", name);
                    }

                    unsafe {
                        if n_dims == 1 {
                            if tensor.as_ref().ne[0] != ne[0] || tensor.as_ref().ne[1] != ne[1] {
                                anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                     name, tensor.as_ref().ne[0], tensor.as_ref().ne[1], ne[0], ne[1]);
                            }
                        } else if split_type == 0 {
                            if tensor.as_ref().ne[0] / (n_parts as i32) != ne[0]
                                || tensor.as_ref().ne[1] != ne[1]
                            {
                                anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                     name, tensor.as_ref().ne[0]/(n_parts as i32), tensor.as_ref().ne[1], ne[0], ne[1]);
                            }
                        } else if tensor.as_ref().ne[0] != ne[0]
                            || tensor.as_ref().ne[1] / (n_parts as i32) != ne[1]
                        {
                            anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                 name, tensor.as_ref().ne[0], tensor.as_ref().ne[1]/(n_parts as i32), ne[0], ne[1]);
                        }
                    }

                    if PRINT_LAYERS {
                        let ftype_str = ["f32", "f16", "q4_0", "q4_1"];
                        log::info!(
                            "{name} - [{}, {}], type = {}, split = {split_type}",
                            ne[0],
                            ne[1],
                            ftype_str[ftype as usize],
                        );
                    }

                    let bpe = match ftype {
                        0 => unsafe { ggml_type_size(ggml_type_GGML_TYPE_F32) },
                        1 => unsafe { ggml_type_size(ggml_type_GGML_TYPE_F16) },
                        2 => {
                            let bpe = unsafe { ggml_type_size(ggml_type_GGML_TYPE_Q4_0) };
                            assert_eq!(ne[0] % 64, 0);
                            bpe
                        }
                        3 => {
                            let bpe = unsafe { ggml_type_size(ggml_type_GGML_TYPE_Q4_1) };
                            assert_eq!(ne[0] % 64, 0);
                            bpe
                        }
                        _ => anyhow::bail!("unknown ftype {ftype} in model file"),
                    };

                    unsafe {
                        if n_dims == 1 || n_parts == 1 {
                            if (usize::try_from(nelements)? * bpe)
                                / usize::try_from(ggml_blck_size(tensor.as_ref().type_))?
                                != ggml_nbytes(tensor.as_ptr())
                            {
                                anyhow::bail!(
                                    "tensor '{}' has wrong size in model file: got {}, expected {}",
                                    name,
                                    ggml_nbytes(tensor.as_ptr()),
                                    nelements * i32::try_from(bpe)?
                                );
                            }

                            if part_id == 0 {
                                read_into_pointer(
                                    &mut fin,
                                    tensor.as_mut().data as *mut u8,
                                    ggml_nbytes(tensor.as_ptr()).try_into()?,
                                )?;
                            } else {
                                fin.seek(SeekFrom::Current(
                                    ggml_nbytes(tensor.as_ptr()).try_into()?,
                                ))?;
                            }

                            total_size += ggml_nbytes(tensor.as_ptr());
                        } else {
                            if (usize::try_from(nelements)? * bpe)
                                / usize::try_from(ggml_blck_size(tensor.as_ref().type_))?
                                != (ggml_nbytes(tensor.as_ptr()) / usize::try_from(n_parts)?)
                            {
                                anyhow::bail!(
                                    "tensor '{}' has wrong size in model file: got {}, expected {}",
                                    name,
                                    ggml_nbytes(tensor.as_ptr()) / usize::try_from(n_parts)?,
                                    nelements * i32::try_from(bpe)?
                                );
                            }

                            if split_type == 0 {
                                let np0 = ne[0];

                                let row_size = usize::try_from(
                                    tensor.as_ref().ne[0] / ggml_blck_size(tensor.as_ref().type_),
                                )? * ggml_type_size(tensor.as_ref().type_);
                                assert_eq!(row_size, tensor.as_ref().nb[1]);

                                for i1 in 0..ne[1] {
                                    let offset_row = usize::try_from(i1)? * row_size;
                                    let offset = offset_row
                                        + (usize::try_from(part_id)? * usize::try_from(np0)?)
                                            / usize::try_from(ggml_blck_size(
                                                tensor.as_ref().type_,
                                            ))?
                                            * ggml_type_size(tensor.as_ref().type_);
                                    read_into_pointer(
                                        &mut fin,
                                        (tensor.as_mut().data as *mut u8)
                                            .offset(offset.try_into()?),
                                        u64::try_from(row_size)? / u64::try_from(n_parts)?,
                                    )?;
                                }
                            } else {
                                let np1 = ne[1];

                                let row_size = usize::try_from(tensor.as_ref().ne[0])?
                                    / usize::try_from(ggml_blck_size(tensor.as_ref().type_))?
                                    * ggml_type_size(tensor.as_ref().type_);

                                for i1 in 0..ne[1] {
                                    let offset_row = usize::try_from(i1)?
                                        + usize::try_from(part_id)?
                                            * usize::try_from(np1)?
                                            * row_size;
                                    read_into_pointer(
                                        &mut fin,
                                        (tensor.as_mut().data as *mut u8)
                                            .offset(isize::try_from(offset_row)?),
                                        row_size.try_into()?,
                                    )?;
                                }
                            }
                        }

                        total_size += ggml_nbytes(tensor.as_ptr()) / usize::try_from(n_parts)?;
                    }

                    n_tensors += 1;
                    if n_tensors % 8 == 0 {
                        log::info!("loaded tensor {n_tensors}");
                    }
                }

                log::info!(" done");

                log::info!(
                    "model size = {} MB / num tensors = {}",
                    (total_size as f32) / 1024.0 / 1024.0,
                    n_tensors
                );
            }
        }

        Ok(LlamaModel {
            hparams,
            tok_embeddings,
            norm,
            output,
            layers,
            memory_k,
            memory_v,
            ctx,
            tensors,
        })
    }
}

const LLAMA_BUF_SIZE_DEFAULT: usize = 512 * 1024 * 1024;
static LLAMA_BUF: Lazy<Mutex<Box<[u8]>>> =
    Lazy::new(|| Mutex::new(vec![0u8; LLAMA_BUF_SIZE_DEFAULT].into_boxed_slice()));

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
fn llama_eval(
    model: &LlamaModel,
    n_threads: i32,
    n_past: i32,
    embd_inp: &[GptVocabId],
    embd_w: &mut Vec<f32>,
    mem_per_token: &mut usize,
) -> anyhow::Result<()> {
    let n = embd_inp.len();

    let hparams = &model.hparams;

    let n_embd = hparams.n_embd;
    let n_layer = hparams.n_layer;
    let n_ctx = hparams.n_ctx;
    let n_head = hparams.n_head;
    let n_vocab = hparams.n_vocab;
    let n_rot = hparams.n_embd / hparams.n_head;

    if *mem_per_token > 0 && *mem_per_token * n > LLAMA_BUF.lock().unwrap().len() {
        let buf_size_new = (1.1 * (*mem_per_token * n) as f32) as usize; // add 10% to account for ggml object overhead
                                                                         //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        *LLAMA_BUF.lock().unwrap() = vec![0u8; buf_size_new].into_boxed_slice();
    }

    let mut buf = LLAMA_BUF.lock().unwrap();

    let ctx0 = NonNull::new(unsafe {
        ggml_init(ggml_init_params {
            mem_size: buf.len(),
            mem_buffer: buf.as_mut_ptr() as *mut _,
        })
    })
    .context("failed to init ctx0")?;

    let mut gf: ggml_cgraph = unsafe { std::mem::zeroed() };
    gf.n_threads = n_threads;

    let mut embd = NonNull::new(unsafe {
        ggml_new_tensor_1d(ctx0.as_ptr(), ggml_type_GGML_TYPE_I32, n.try_into()?)
    })
    .context("failed to crate embd")?;

    unsafe { std::slice::from_raw_parts_mut(embd.as_mut().data as *mut GptVocabId, n) }
        .copy_from_slice(embd_inp);

    let mut inp_l =
        unsafe { ggml_get_rows(ctx0.as_ptr(), model.tok_embeddings.as_ptr(), embd.as_ptr()) };

    for il in 0..n_layer {
        let inp_sa = inp_l;
        let mut cur;

        // norm
        unsafe {
            cur = ggml_norm(ctx0.as_ptr(), inp_l);

            // cur = attention_norm*cur
            cur = ggml_mul(
                ctx0.as_ptr(),
                ggml_repeat(
                    ctx0.as_ptr(),
                    model.layers[il as usize].attention_norm.as_ptr(),
                    cur,
                ),
                cur,
            );
        }

        // self-attention
        unsafe {
            let q_cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].wq.as_ptr(), cur);
            let k_cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].wk.as_ptr(), cur);
            let v_cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].wv.as_ptr(), cur);

            // store key and value to memory
            if n >= 1 {
                let k = ggml_view_1d(
                    ctx0.as_ptr(),
                    model.memory_k.as_ptr(),
                    i32::try_from(n)? * n_embd,
                    (ggml_element_size(model.memory_k.as_ptr()) * usize::try_from(n_embd)?)
                        * usize::try_from(il * n_ctx + n_past)?,
                );
                let v = ggml_view_1d(
                    ctx0.as_ptr(),
                    model.memory_v.as_ptr(),
                    i32::try_from(n)? * n_embd,
                    (ggml_element_size(model.memory_v.as_ptr()) * usize::try_from(n_embd)?)
                        * usize::try_from(il * n_ctx + n_past)?,
                );

                ggml_build_forward_expand(&mut gf, ggml_cpy(ctx0.as_ptr(), k_cur, k));
                ggml_build_forward_expand(&mut gf, ggml_cpy(ctx0.as_ptr(), v_cur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            let q = ggml_permute(
                ctx0.as_ptr(),
                ggml_rope(
                    ctx0.as_ptr(),
                    ggml_cpy(
                        ctx0.as_ptr(),
                        q_cur,
                        ggml_new_tensor_3d(
                            ctx0.as_ptr(),
                            ggml_type_GGML_TYPE_F32,
                            n_embd / n_head,
                            n_head,
                            i32::try_from(n)?,
                        ),
                    ),
                    n_past,
                    n_rot,
                    0,
                ),
                0,
                2,
                1,
                3,
            );

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            let k = ggml_permute(
                ctx0.as_ptr(),
                ggml_rope(
                    ctx0.as_ptr(),
                    ggml_reshape_3d(
                        ctx0.as_ptr(),
                        ggml_view_1d(
                            ctx0.as_ptr(),
                            model.memory_k.as_ptr(),
                            (n_past + i32::try_from(n)?) * n_embd,
                            usize::try_from(il * n_ctx * n_embd)?
                                * ggml_element_size(model.memory_k.as_ptr()),
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + i32::try_from(n)?,
                    ),
                    n_past,
                    n_rot,
                    1,
                ),
                0,
                2,
                1,
                3,
            );

            // K * Q
            let kq = ggml_mul_mat(ctx0.as_ptr(), k, q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            let kq_scaled = ggml_scale(
                ctx0.as_ptr(),
                kq,
                ggml_new_f32(
                    ctx0.as_ptr(),
                    1.0 / ((n_embd as f32) / (n_head as f32)).sqrt(),
                ),
            );

            // KQ_masked = mask_past(KQ_scaled)
            let kq_masked = ggml_diag_mask_inf(ctx0.as_ptr(), kq_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            let kq_soft_max = ggml_soft_max(ctx0.as_ptr(), kq_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            let v_trans = ggml_permute(
                ctx0.as_ptr(),
                ggml_reshape_3d(
                    ctx0.as_ptr(),
                    ggml_view_1d(
                        ctx0.as_ptr(),
                        model.memory_v.as_ptr(),
                        (n_past + i32::try_from(n)?) * n_embd,
                        usize::try_from(il * n_ctx * n_embd)?
                            * ggml_element_size(model.memory_v.as_ptr()),
                    ),
                    n_embd / n_head,
                    n_head,
                    n_past + i32::try_from(n)?,
                ),
                1,
                2,
                0,
                3,
            );

            // KQV = transpose(V) * KQ_soft_max
            let kqv = ggml_mul_mat(ctx0.as_ptr(), v_trans, kq_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            let kqv_merged = ggml_permute(ctx0.as_ptr(), kqv, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(
                ctx0.as_ptr(),
                kqv_merged,
                ggml_new_tensor_2d(
                    ctx0.as_ptr(),
                    ggml_type_GGML_TYPE_F32,
                    n_embd,
                    i32::try_from(n)?,
                ),
            );

            // projection (no bias)
            cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].wo.as_ptr(), cur);
        }

        let inp_ff = unsafe { ggml_add(ctx0.as_ptr(), cur, inp_sa) };

        // feed-forward network
        unsafe {
            // norm
            {
                cur = ggml_norm(ctx0.as_ptr(), inp_ff);

                // cur = ffn_norm*cur
                cur = ggml_mul(
                    ctx0.as_ptr(),
                    ggml_repeat(
                        ctx0.as_ptr(),
                        model.layers[il as usize].ffn_norm.as_ptr(),
                        cur,
                    ),
                    cur,
                );
            }

            let tmp = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].w3.as_ptr(), cur);

            cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].w1.as_ptr(), cur);

            // SILU activation
            cur = ggml_silu(ctx0.as_ptr(), cur);

            cur = ggml_mul(ctx0.as_ptr(), cur, tmp);

            cur = ggml_mul_mat(ctx0.as_ptr(), model.layers[il as usize].w2.as_ptr(), cur);
        }

        cur = unsafe { ggml_add(ctx0.as_ptr(), cur, inp_ff) };

        // input for next layer
        inp_l = cur;
    }

    // norm
    unsafe {
        inp_l = ggml_norm(ctx0.as_ptr(), inp_l);

        // inpL = norm*inpL
        inp_l = ggml_mul(
            ctx0.as_ptr(),
            ggml_repeat(ctx0.as_ptr(), model.norm.as_ptr(), inp_l),
            inp_l,
        );
    }

    // lm_head
    unsafe {
        inp_l = ggml_mul_mat(ctx0.as_ptr(), model.output.as_ptr(), inp_l);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0.as_ptr(), inpL);

    // run the computation
    unsafe { ggml_build_forward_expand(&mut gf, inp_l) };
    unsafe { ggml_graph_compute(ctx0.as_ptr(), &mut gf) };

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab.try_into()?, Default::default());
    embd_w[0..n_vocab as usize].copy_from_slice(unsafe {
        std::slice::from_raw_parts(
            (ggml_get_data(inp_l) as *const f32)
                .offset(isize::try_from(usize::try_from(n_vocab)? * (n - 1))?),
            n_vocab as usize,
        )
    });

    if *mem_per_token == 0 {
        *mem_per_token = unsafe { ggml_used_mem(ctx0.as_ptr()) } / n;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    unsafe { ggml_free(ctx0.as_ptr()) };

    Ok(())
}

fn main() -> anyhow::Result<()> {
    simplelog::CombinedLogger::init(vec![simplelog::TermLogger::new(
        simplelog::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    )])?;

    unsafe { ggml_time_init() };

    let t_main_start_us = unsafe { ggml_time_us() };

    let mut params = GptParams::parse();
    params.seed = match params.seed {
        Some(seed) => Some(seed),
        _ => Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        ),
    };

    log::info!("seed: {}", params.seed.unwrap());

    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed.unwrap());

    let t_load_us;

    let mut vocab = GptVocab::default();

    // load the model
    let model = {
        let t_start_us = unsafe { ggml_time_us() };
        let model = LlamaModel::load(&params.model, &mut vocab, 512)?;
        t_load_us = unsafe { ggml_time_us() } - t_start_us;

        model
    };

    let mut n_past = 0;

    let mut t_sample_us = 0;
    let mut t_predict_us = 0;

    let mut logits: Vec<f32> = vec![];
    let embd_inp = llama_tokenize(&vocab, &params.prompt, true);

    params.n_predict = params
        .n_predict
        .min(model.hparams.n_ctx - i32::try_from(embd_inp.len())?);

    log::info!("prompt: '{}'", params.prompt);
    log::info!("number of tokens in prompt = {}", embd_inp.len());
    for embedding in &embd_inp {
        log::info!(
            "{} -> '{}'",
            embedding,
            vocab.id_to_token.get(embedding).expect("embedding missing")
        );
    }

    log::info!("sampling parameters: temp = {}, top_k = {}, top_p = {}, repeat_last_n = {}, repeat_penalty = {}", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);

    let mut embd: Vec<GptVocabId> = vec![];

    // determine the required inference memory per token:
    let mut mem_per_token: usize = 0;
    llama_eval(
        &model,
        params.n_threads().try_into()?,
        0,
        &[0, 1, 2, 3],
        &mut logits,
        &mut mem_per_token,
    )?;

    let last_n_size = usize::try_from(params.repeat_last_n)?;
    let mut last_n_tokens = vec![0; last_n_size];

    let mut remaining_tokens = params.n_predict;
    let mut input_consumed = 0;

    while remaining_tokens > 0 {
        if !embd.is_empty() {
            let t_start_us = unsafe { ggml_time_us() };

            llama_eval(
                &model,
                params.n_threads().try_into()?,
                n_past,
                &embd,
                &mut logits,
                &mut mem_per_token,
            )?;

            t_predict_us += unsafe { ggml_time_us() } - t_start_us;
        }

        n_past += i32::try_from(embd.len())?;
        embd.clear();

        if embd_inp.len() <= input_consumed {
            // out of input, sample next token
            let top_k = params.top_k;
            let top_p = params.top_p;
            let temp = params.temp;
            let repeat_penalty = params.repeat_penalty;

            let n_vocab = model.hparams.n_vocab;

            let id = {
                let t_start_sample_us = unsafe { ggml_time_us() };

                let id = llama_sample_top_p_top_k(
                    &vocab,
                    &logits[logits.len() - usize::try_from(n_vocab)?..],
                    &last_n_tokens,
                    repeat_penalty.try_into()?,
                    top_k,
                    top_p.try_into()?,
                    temp.try_into()?,
                    &mut rng,
                );

                last_n_tokens.remove(0);
                last_n_tokens.push(id);

                t_sample_us += unsafe { ggml_time_us() } - t_start_sample_us;

                id
            };

            embd.push(id);
            remaining_tokens -= 1;
        } else {
            while embd_inp.len() > input_consumed {
                embd.push(embd_inp[input_consumed]);
                last_n_tokens.remove(0);
                last_n_tokens.push(embd_inp[input_consumed]);
                input_consumed += 1;

                if embd.len() > usize::try_from(params.n_batch)? {
                    break;
                }
            }
        }

        for id in &embd {
            log::info!("{}", vocab.id_to_token.get(id).expect("no token"));
        }

        if embd.last() == Some(&2) {
            log::info!("[end of text]");
            break;
        }
    }

    // report timing
    {
        let t_main_end_us = unsafe { ggml_time_us() };

        log::info!("mem per token = {} bytes", mem_per_token);
        log::info!("    load time = {} ms", (t_load_us as f64) / 1000.0);
        log::info!("  sample time = {} ms", (t_sample_us as f64) / 1000.0);
        log::info!(
            " predict time = {} ms / {} ms per token",
            (t_predict_us as f64) / 1000.0,
            (t_predict_us as f64) / 1000.0 / (n_past as f64),
        );
        log::info!(
            "   total time = {} ms",
            (t_main_end_us - t_main_start_us) as f64 / 1000.0,
        );
    }

    Ok(())
}
