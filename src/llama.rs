use crate::ggml;
use anyhow::Context;
use once_cell::sync::Lazy;
use partial_sort::PartialSort;
use rand::prelude::Distribution;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Mutex,
};

static LLAMA_N_PARTS: Lazy<HashMap<u32, u32>> =
    Lazy::new(|| HashMap::from_iter([(4096, 1), (5120, 2), (6656, 4), (8192, 8)]));

pub type VocabularyId = i32;
pub type VocabularyToken = String;

pub struct Model<'a> {
    hparams: Hyperparameters,

    tok_embeddings: ggml::Tensor<'a>,

    norm: ggml::Tensor<'a>,
    output: ggml::Tensor<'a>,

    layers: Vec<Layer<'a>>,

    memory_k: ggml::Tensor<'a>,
    memory_v: ggml::Tensor<'a>,
}
impl Model<'_> {
    pub fn n_ctx(&self) -> i32 {
        self.hparams.n_ctx
    }
    pub fn n_vocab(&self) -> i32 {
        self.hparams.n_vocab
    }
}
impl Model<'_> {
    pub fn load(
        fname: &Path,
        n_ctx: i32,
        vocab: &mut Vocabulary,
    ) -> anyhow::Result<(ggml::Context, Preload)> {
        log::info!("loading model from {fname:?} - please wait ...");
        let mut fin = std::fs::File::open(fname)?;
        {
            if read_u32(&mut fin)?.context("eof while reading magic")? != 0x67676d6c {
                anyhow::bail!("invalid model file {fname:?} (bad magic)");
            }
        }
        let n_ff: i32;
        let n_parts: u32;
        let hparams = {
            let n_vocab = read_i32(&mut fin)?.context("eof reading n_vocab")?;
            let n_embd = read_i32(&mut fin)?.context("eof reading n_embd")?;
            let n_mult = read_i32(&mut fin)?.context("eof reading n_mult")?;
            let n_head = read_i32(&mut fin)?.context("eof reading n_head")?;
            let n_layer = read_i32(&mut fin)?.context("eof reading n_layer")?;
            let n_rot = read_i32(&mut fin)?.context("eof reading n_rot")?;
            let f16 = read_i32(&mut fin)?.context("eof reading f16")?;

            let hparams = Hyperparameters {
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
        {
            let n_vocab = hparams.n_vocab;

            for i in 0..n_vocab {
                let word = read_string(&mut fin)?.context("eof while reading vocab")?;
                vocab.token_to_id.insert(word.clone(), i);
                vocab.id_to_token.insert(i, word.clone());
            }
        }
        let wtype = match hparams.f16 {
            0 => ggml::Type::F32,
            1 => ggml::Type::F16,
            2 => ggml::Type::Q4_0,
            3 => ggml::Type::Q4_1,
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

            let wtype_sizef = wtype.sizef()?;
            let f32_sizef = ggml::Type::F32.sizef()?;

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
        let ctx = ggml::Context::new(ctx_size, None).context("failed to create ggml context")?;
        let file_offset = fin.stream_position()?;
        Ok((
            ctx,
            Preload {
                fname: fname.to_owned(),
                file_offset,
                n_ff,
                n_parts,
                hparams,
                wtype,
            },
        ))
    }

    // evaluate the transformer
    //
    //   - n_threads: number of threads to use
    //   - n_past:    the context size so far
    //   - embd_inp:  the embeddings of the tokens in the context
    //   - embd_w:    the predicted logits for the next token
    //
    pub fn evaluate(
        &self,
        n_threads: usize,
        n_past: i32,
        embd_inp: &[VocabularyId],
        embd_w: &mut Vec<f32>,
        mem_per_token: &mut usize,
    ) -> anyhow::Result<()> {
        let n = embd_inp.len();

        let hparams = &self.hparams;

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

        let ctx0 = ggml::Context::new(buf.len(), NonNull::new(buf.as_mut_ptr()))
            .context("failed to create ctx0")?;

        let mut gf = ggml::ComputationGraph::new(n_threads)?;

        let mut embd = ctx0.new_tensor_1d(ggml::Type::I32, n)?;
        embd.as_mut_slice().copy_from_slice(embd_inp);

        let mut inp_l = ctx0.get_rows(self.tok_embeddings, embd)?;

        for il in 0..n_layer {
            let inp_sa = inp_l;
            let mut cur;

            // norm
            {
                cur = ctx0.norm(inp_l)?;

                // cur = attention_norm*cur
                let a = ctx0.repeat(self.layers[il as usize].attention_norm, cur)?;
                cur = ctx0.mul(a, cur)?;
            }

            // self-attention
            {
                let q_cur = ctx0.mul_mat(self.layers[il as usize].wq, cur)?;
                let k_cur = ctx0.mul_mat(self.layers[il as usize].wk, cur)?;
                let v_cur = ctx0.mul_mat(self.layers[il as usize].wv, cur)?;

                // store key and value to memory
                if n >= 1 {
                    let k = ctx0.view_1d(
                        self.memory_k,
                        n * usize::try_from(n_embd)?,
                        (self.memory_k.element_size() * usize::try_from(n_embd)?)
                            * usize::try_from(il * n_ctx + n_past)?,
                    )?;
                    let v = ctx0.view_1d(
                        self.memory_v,
                        n * usize::try_from(n_embd)?,
                        (self.memory_v.element_size() * usize::try_from(n_embd)?)
                            * usize::try_from(il * n_ctx + n_past)?,
                    )?;

                    gf.build_forward_expand(ctx0.cpy(k_cur, k)?);
                    gf.build_forward_expand(ctx0.cpy(v_cur, v)?);
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let b = ctx0.new_tensor_3d(
                    ggml::Type::F32,
                    usize::try_from(n_embd / n_head)?,
                    usize::try_from(n_head)?,
                    n,
                )?;
                let a = ctx0.cpy(q_cur, b)?;
                let a = ctx0.rope(a, usize::try_from(n_past)?, usize::try_from(n_rot)?, 0)?;
                let q = ctx0.permute(a, 0, 2, 1, 3)?;

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let a = ctx0.view_1d(
                    self.memory_k,
                    (usize::try_from(n_past)? + n) * usize::try_from(n_embd)?,
                    usize::try_from(il * n_ctx * n_embd)? * self.memory_k.element_size(),
                )?;
                let a = ctx0.reshape_3d(
                    a,
                    usize::try_from(n_embd / n_head)?,
                    usize::try_from(n_head)?,
                    usize::try_from(n_past)? + n,
                )?;
                let a = ctx0.rope(a, usize::try_from(n_past)?, usize::try_from(n_rot)?, 1)?;
                let k = ctx0.permute(a, 0, 2, 1, 3)?;

                // K * Q
                let kq = ctx0.mul_mat(k, q)?;

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let b = ctx0.new_tensor_f32(1.0 / ((n_embd as f32) / (n_head as f32)).sqrt())?;
                let kq_scaled = ctx0.scale(kq, b)?;

                // KQ_masked = mask_past(KQ_scaled)
                let kq_masked = ctx0.diag_mask_inf(kq_scaled, n_past.try_into()?)?;

                // KQ = soft_max(KQ_masked)
                let kq_soft_max = ctx0.soft_max(kq_masked)?;

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                let a = ctx0.view_1d(
                    self.memory_v,
                    (usize::try_from(n_past)? + n) * usize::try_from(n_embd)?,
                    usize::try_from(il * n_ctx * n_embd)? * self.memory_v.element_size(),
                )?;
                let a = ctx0.reshape_3d(
                    a,
                    usize::try_from(n_embd / n_head)?,
                    usize::try_from(n_head)?,
                    usize::try_from(n_past)? + n,
                )?;
                let v_trans = ctx0.permute(a, 1, 2, 0, 3)?;

                // KQV = transpose(V) * KQ_soft_max
                let kqv = ctx0.mul_mat(v_trans, kq_soft_max)?;

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let kqv_merged = ctx0.permute(kqv, 0, 2, 1, 3)?;

                // cur = KQV_merged.contiguous().view(n_embd, N)
                let b = ctx0.new_tensor_2d(ggml::Type::F32, usize::try_from(n_embd)?, n)?;
                cur = ctx0.cpy(kqv_merged, b)?;

                // projection (no bias)
                cur = ctx0.mul_mat(self.layers[il as usize].wo, cur)?;
            }

            let inp_ff = ctx0.add(cur, inp_sa)?;

            // feed-forward network
            {
                // norm
                {
                    cur = ctx0.norm(inp_ff)?;

                    // cur = ffn_norm*cur
                    let a = ctx0.repeat(self.layers[il as usize].ffn_norm, cur)?;
                    cur = ctx0.mul(a, cur)?;
                }

                let tmp = ctx0.mul_mat(self.layers[il as usize].w3, cur)?;

                cur = ctx0.mul_mat(self.layers[il as usize].w1, cur)?;

                // SILU activation
                cur = ctx0.silu(cur)?;

                cur = ctx0.mul(cur, tmp)?;

                cur = ctx0.mul_mat(self.layers[il as usize].w2, cur)?;
            }

            cur = ctx0.add(cur, inp_ff)?;

            // input for next layer
            inp_l = cur;
        }

        // norm
        {
            inp_l = ctx0.norm(inp_l)?;

            // inpL = norm*inpL
            let a = ctx0.repeat(self.norm, inp_l)?;
            inp_l = ctx0.mul(a, inp_l)?;
        }

        // lm_head
        {
            inp_l = ctx0.mul_mat(self.output, inp_l)?;
        }

        // logits -> probs
        //inpL = ggml_soft_max(ctx0.as_ptr(), inpL);

        // run the computation
        {
            gf.build_forward_expand(inp_l);
        };
        {
            ctx0.compute(&mut gf);
        };

        // return result for just the last token
        embd_w.resize(n_vocab.try_into()?, Default::default());
        embd_w[0..n_vocab as usize].copy_from_slice({
            let inp_l_slice = inp_l.as_mut_slice();
            let base = usize::try_from(n_vocab)? * (n - 1);
            &inp_l_slice[base..base + usize::try_from(n_vocab)?]
        });

        if *mem_per_token == 0 {
            *mem_per_token = ctx0.used_memory() / n;
        }

        Ok(())
    }
}

pub struct Preload {
    fname: PathBuf,
    file_offset: u64,
    n_ff: i32,
    n_parts: u32,
    hparams: Hyperparameters,
    wtype: ggml::Type,
}
impl Preload {
    pub fn finish(self, ctx: &mut ggml::Context) -> anyhow::Result<Model<'_>> {
        const PRINT_LAYERS: bool = false;

        let Self {
            fname,
            file_offset,
            n_ff,
            n_parts,
            hparams,
            wtype,
        } = self;

        let (layers, tok_embeddings, norm, output, tensors) = {
            let n_embd = usize::try_from(hparams.n_embd)?;
            let n_layer = usize::try_from(hparams.n_layer)?;
            // let n_ctx = hparams.n_ctx;
            let n_vocab = usize::try_from(hparams.n_vocab)?;
            let n_ff = usize::try_from(n_ff)?;

            let mut layers = vec![];

            let tok_embeddings = ctx.new_tensor_2d(wtype, n_embd, n_vocab)?;
            let norm = ctx.new_tensor_1d(ggml::Type::F32, n_embd)?;
            let output = ctx.new_tensor_2d(wtype, n_embd, n_vocab)?;
            // map by name
            let mut tensors: HashMap<String, ggml::Tensor> = HashMap::default();
            tensors.insert("tok_embeddings.weight".to_string(), tok_embeddings);

            tensors.insert("norm.weight".to_string(), norm);
            tensors.insert("output.weight".to_string(), output);

            for i in 0..n_layer {
                let attention_norm = ctx.new_tensor_1d(ggml::Type::F32, n_embd)?;

                let wq = ctx.new_tensor_2d(wtype, n_embd, n_embd)?;
                let wk = ctx.new_tensor_2d(wtype, n_embd, n_embd)?;
                let wv = ctx.new_tensor_2d(wtype, n_embd, n_embd)?;
                let wo = ctx.new_tensor_2d(wtype, n_embd, n_embd)?;

                let ffn_norm = ctx.new_tensor_1d(ggml::Type::F32, n_embd)?;

                let w1 = ctx.new_tensor_2d(wtype, n_embd, n_ff)?;
                let w2 = ctx.new_tensor_2d(wtype, n_ff, n_embd)?;
                let w3 = ctx.new_tensor_2d(wtype, n_embd, n_ff)?;

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

                layers.push(Layer {
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
            let n_elements = usize::try_from(n_embd * n_mem)?;

            let memory_k = ctx.new_tensor_1d(ggml::Type::F32, n_elements)?;
            let memory_v = ctx.new_tensor_1d(ggml::Type::F32, n_elements)?;

            let memory_size = memory_k.n_bytes() + memory_v.n_bytes();

            log::info!(
                "memory_size = {} MB, n_mem = {}",
                (memory_size as f32) / 1024.0 / 1024.0,
                n_mem,
            );

            (memory_k, memory_v)
        };

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
                        if tensor.n_elements()? != usize::try_from(nelements)? {
                            anyhow::bail!("tensor {} has wrong size in model file", name);
                        }
                    } else if tensor.n_elements()? / usize::try_from(n_parts)?
                        != usize::try_from(nelements)?
                    {
                        anyhow::bail!("tensor {} has wrong size in model file", name);
                    }

                    {
                        if n_dims == 1 {
                            if tensor.ne()[0] != ne[0] || tensor.ne()[1] != ne[1] {
                                anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                     name, tensor.ne()[0], tensor.ne()[1], ne[0], ne[1]);
                            }
                        } else if split_type == 0 {
                            if tensor.ne()[0] / (n_parts as i32) != ne[0] || tensor.ne()[1] != ne[1]
                            {
                                anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                     name, tensor.ne()[0]/(n_parts as i32), tensor.ne()[1], ne[0], ne[1]);
                            }
                        } else if tensor.ne()[0] != ne[0]
                            || tensor.ne()[1] / (n_parts as i32) != ne[1]
                        {
                            anyhow::bail!("tensor {} has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                                 name, tensor.ne()[0], tensor.ne()[1]/(n_parts as i32), ne[0], ne[1]);
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
                        0 => ggml::Type::F32.size(),
                        1 => ggml::Type::F16.size(),
                        2 => {
                            let bpe = ggml::Type::Q4_0.size();
                            assert_eq!(ne[0] % 64, 0);
                            bpe
                        }
                        3 => {
                            let bpe = ggml::Type::Q4_1.size();
                            assert_eq!(ne[0] % 64, 0);
                            bpe
                        }
                        _ => anyhow::bail!("unknown ftype {ftype} in model file"),
                    };

                    {
                        if n_dims == 1 || n_parts == 1 {
                            if (usize::try_from(nelements)? * bpe) / tensor.type_().blck_size()?
                                != tensor.n_bytes()
                            {
                                anyhow::bail!(
                                    "tensor '{}' has wrong size in model file: got {}, expected {}",
                                    name,
                                    tensor.n_bytes(),
                                    nelements * i32::try_from(bpe)?
                                );
                            }

                            if part_id == 0 {
                                read_into_slice(&mut fin, tensor.as_mut_slice_u8())?;
                            } else {
                                fin.seek(SeekFrom::Current(tensor.n_bytes().try_into()?))?;
                            }

                            total_size += tensor.n_bytes();
                        } else {
                            if (usize::try_from(nelements)? * bpe) / tensor.type_().blck_size()?
                                != (tensor.n_bytes() / usize::try_from(n_parts)?)
                            {
                                anyhow::bail!(
                                    "tensor '{}' has wrong size in model file: got {}, expected {}",
                                    name,
                                    tensor.n_bytes() / usize::try_from(n_parts)?,
                                    nelements * i32::try_from(bpe)?
                                );
                            }

                            if split_type == 0 {
                                let np0 = ne[0];

                                let row_size = usize::try_from(tensor.ne()[0])?
                                    / tensor.type_().blck_size()?
                                    * tensor.type_().size();
                                assert_eq!(row_size, tensor.nb()[1]);

                                for i1 in 0..ne[1] {
                                    let offset_row = usize::try_from(i1)? * row_size;
                                    let offset = offset_row
                                        + (usize::try_from(part_id)? * usize::try_from(np0)?)
                                            / tensor.type_().blck_size()?
                                            * tensor.type_().size();

                                    let slice = tensor.as_mut_slice_u8();
                                    read_into_slice(
                                        &mut fin,
                                        &mut slice[offset
                                            ..offset + (row_size / usize::try_from(n_parts)?)],
                                    )?;
                                }
                            } else {
                                let np1 = ne[1];

                                let row_size = usize::try_from(tensor.ne()[0])?
                                    / tensor.type_().blck_size()?
                                    * tensor.type_().size();

                                for i1 in 0..ne[1] {
                                    let offset_row = usize::try_from(i1)?
                                        + usize::try_from(part_id)?
                                            * usize::try_from(np1)?
                                            * row_size;

                                    let slice = tensor.as_mut_slice_u8();
                                    read_into_slice(
                                        &mut fin,
                                        &mut slice[offset_row..offset_row + row_size],
                                    )?;
                                }
                            }
                        }

                        total_size += tensor.n_bytes() / usize::try_from(n_parts)?;
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

        Ok(Model {
            hparams,
            tok_embeddings,
            norm,
            output,
            layers,
            memory_k,
            memory_v,
        })
    }
}

const LLAMA_BUF_SIZE_DEFAULT: usize = 512 * 1024 * 1024;
static LLAMA_BUF: Lazy<Mutex<Box<[u8]>>> =
    Lazy::new(|| Mutex::new(vec![0u8; LLAMA_BUF_SIZE_DEFAULT].into_boxed_slice()));

#[derive(Default)]
pub struct Vocabulary {
    pub token_to_id: HashMap<VocabularyToken, VocabularyId>,
    pub id_to_token: HashMap<VocabularyId, VocabularyToken>,
}
impl Vocabulary {
    // TODO: this is probably wrong, but I cannot figure out how this tokenizer works ..
    // ref: https://github.com/google/sentencepiece
    pub fn tokenize(&self, text: &str, bos: bool) -> Vec<VocabularyId> {
        let mut res: Vec<VocabularyId> = vec![];

        if bos {
            res.push(1); // TODO: replace with vocab.bos
        }

        //find the longest token that matches the text
        let mut pos = 0;
        loop {
            let mut l = 0;
            let mut t = 0;
            for (k, v) in &self.id_to_token {
                if v.len() < l {
                    continue;
                }
                if v.len() > text.len() - pos {
                    continue;
                }
                if &text[pos..pos + v.len()] == v {
                    l = v.len();
                    t = *k;
                }
            }

            if l == 0 {
                break;
            }

            res.push(t);
            pos += l;
        }

        res
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_top_p_top_k(
        &self,
        logits: &[f32],
        last_n_tokens: &[VocabularyId],
        repeat_penalty: f64,
        top_k: usize,
        top_p: f64,
        temperature: f64,
        rng: &mut impl rand::Rng,
    ) -> VocabularyId {
        let n_logits = self.id_to_token.len();
        assert_eq!(logits.len(), n_logits);

        let mut logits_id: Vec<(f64, VocabularyId)> = vec![];
        logits_id.reserve(n_logits);

        {
            let scale: f64 = 1.0 / temperature;
            for (i, logit) in logits.iter().copied().enumerate() {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                if last_n_tokens.contains(&i32::try_from(i).unwrap()) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logit < 0.0 {
                        logits_id.push((
                            f64::from(logit) * scale * repeat_penalty,
                            i.try_into().unwrap(),
                        ));
                    } else {
                        logits_id.push((
                            f64::from(logit) * scale / repeat_penalty,
                            i.try_into().unwrap(),
                        ));
                    }
                } else {
                    logits_id.push((f64::from(logit) * scale, i.try_into().unwrap()));
                }
            }
        }

        sample_top_k(&mut logits_id, top_k);

        let mut maxl: f64 = -f64::INFINITY;
        for (k, _) in &logits_id {
            maxl = maxl.max(*k);
        }

        // compute probs for the top K tokens
        let mut probs: Vec<f64> = vec![];
        probs.reserve(logits_id.len());

        let mut sum: f64 = 0.0;
        for (k, _) in &logits_id {
            let p: f64 = (*k - maxl).exp();
            probs.push(p);
            sum += p;
        }

        // normalize the probs
        for p in &mut probs {
            *p /= sum;
        }

        if top_p < 1.0 {
            let mut cumsum: f64 = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    probs.resize(i + 1, Default::default());
                    logits_id.resize(i + 1, Default::default());
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for prob in &mut probs {
                *prob *= cumsum;
            }
        }

        let idx = rand::distributions::WeightedIndex::new(&probs)
            .unwrap()
            .sample(rng);

        logits_id[idx].1
    }
}

struct Layer<'a> {
    // normalization
    attention_norm: ggml::Tensor<'a>,

    // attention
    wq: ggml::Tensor<'a>,
    wk: ggml::Tensor<'a>,
    wv: ggml::Tensor<'a>,
    wo: ggml::Tensor<'a>,

    // normalization
    ffn_norm: ggml::Tensor<'a>,

    // ff
    w1: ggml::Tensor<'a>,
    w2: ggml::Tensor<'a>,
    w3: ggml::Tensor<'a>,
}

struct Hyperparameters {
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
impl Default for Hyperparameters {
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

fn sample_top_k(logits_id: &mut Vec<(f64, VocabularyId)>, top_k: usize) {
    // find the top K tokens
    logits_id.partial_sort(top_k, |a, b| a.0.total_cmp(&b.0));
    logits_id.resize(top_k, Default::default());
}

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

fn read_into_slice(f: &mut File, slice: &mut [u8]) -> anyhow::Result<()> {
    let read_len = f.read(slice)?;
    assert_eq!(read_len, slice.len());
    Ok(())
}
