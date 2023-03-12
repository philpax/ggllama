use std::{collections::HashMap, path::PathBuf};

use clap::Parser;
use partial_sort::PartialSort;
use rand::prelude::Distribution;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct GptParams {
    /// RNG seed
    #[arg(short, long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub n_threads: Option<usize>,
    /// new tokens to predict
    #[arg(long, default_value_t = 128)]
    pub n_predict: i32,
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: i32,

    /// sampling parameters
    #[arg(long, default_value_t = 40)]
    pub top_k: i32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0.80)]
    pub temp: f32,
    #[arg(long, default_value_t = 1.30)]
    pub repeat_penalty: f32,

    /// batch size for prompt processing
    #[arg(long, default_value_t = 8)]
    pub n_batch: i32,

    /// model path
    #[arg(short, long, default_value = "models/7B/ggml-model-q4_0.bin")]
    pub model: PathBuf,
    #[arg()]
    pub prompt: String,
}
impl GptParams {
    pub fn n_threads(&self) -> usize {
        self.n_threads
            .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get().min(4))
    }
}

//
// Vocab utils
//

pub type GptVocabId = i32;
pub type GptVocabToken = String;

#[derive(Default)]
pub struct GptVocab {
    pub token_to_id: HashMap<GptVocabToken, GptVocabId>,
    pub id_to_token: HashMap<GptVocabId, GptVocabToken>,
}

// TODO: this is probably wrong, but I cannot figure out how this tokenizer works ..
// ref: https://github.com/google/sentencepiece
pub fn llama_tokenize(vocab: &GptVocab, text: &str, bos: bool) -> Vec<GptVocabId> {
    let mut res: Vec<GptVocabId> = vec![];

    if bos {
        res.push(1); // TODO: replace with vocab.bos
    }

    //find the longest token that matches the text
    let mut pos = 0;
    loop {
        let mut l = 0;
        let mut t = 0;
        for (k, v) in &vocab.id_to_token {
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

// load the tokens from encoder.json
#[allow(dead_code)]
pub fn gpt_vocab_init(fname: &str, vocab: &mut GptVocab) -> bool {
    log::info!("loading vocab from '{}'", fname);

    vocab.token_to_id = serde_json::from_str(&std::fs::read_to_string(fname).unwrap()).unwrap();
    vocab.id_to_token = vocab
        .token_to_id
        .iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    log::info!("vocab size = {}\n", vocab.token_to_id.len());

    true
}

fn sample_top_k(logits_id: &mut Vec<(f64, GptVocabId)>, top_k: i32) {
    // find the top K tokens
    logits_id.partial_sort(top_k.try_into().unwrap(), |a, b| a.0.total_cmp(&b.0));
    logits_id.resize(top_k.try_into().unwrap(), Default::default());
}

pub fn llama_sample_top_p_top_k(
    vocab: &GptVocab,
    logits: &[f32],
    last_n_tokens: &mut Vec<GptVocabId>,
    repeat_penalty: f64,
    top_k: i32,
    top_p: f64,
    temp: f64,
    rng: &mut impl rand::Rng,
) -> GptVocabId {
    let n_logits = vocab.id_to_token.len();
    assert_eq!(logits.len(), n_logits);

    let mut logits_id: Vec<(f64, GptVocabId)> = vec![];
    logits_id.reserve(n_logits);

    {
        let scale: f64 = 1.0 / temp;
        for i in 0..n_logits {
            // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if last_n_tokens.contains(&i32::try_from(i).unwrap()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if logits[i] < 0.0 {
                    logits_id.push((
                        f64::from(logits[i]) * scale * repeat_penalty,
                        i.try_into().unwrap(),
                    ));
                } else {
                    logits_id.push((
                        f64::from(logits[i]) * scale / repeat_penalty,
                        i.try_into().unwrap(),
                    ));
                }
            } else {
                logits_id.push((f64::from(logits[i]) * scale, i.try_into().unwrap()));
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
        for i in 0..probs.len() {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    let idx = rand::distributions::WeightedIndex::new(&probs)
        .unwrap()
        .sample(rng);

    logits_id[idx].1
}

//
// Quantization
//
/*
size_t ggml_quantize_q4_0(float * src, void * dst, int n, int k, int qk, int64_t * hist);,
size_t ggml_quantize_q4_1(float * src, void * dst, int n, int k, int qk, int64_t * hist);
*/
