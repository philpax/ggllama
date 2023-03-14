use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

mod ggml;
mod llama;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Params {
    /// RNG seed
    #[arg(short, long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub n_threads: Option<usize>,
    /// new tokens to predict
    #[arg(long, default_value_t = 128)]
    pub n_predict: usize,
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// sampling parameters
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f64,
    #[arg(long, default_value_t = 0.80)]
    pub temperature: f64,
    #[arg(long, default_value_t = 1.30)]
    pub repeat_penalty: f64,

    /// batch size for prompt processing
    #[arg(long, default_value_t = 8)]
    pub n_batch: usize,

    /// model path
    #[arg(short, long, default_value = "models/7B/ggml-model-q4_0.bin")]
    pub model: PathBuf,
    #[arg()]
    pub prompt: String,
}
impl Params {
    pub fn n_threads(&self) -> usize {
        self.n_threads
            .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get().min(4))
    }
}

fn main() -> anyhow::Result<()> {
    simplelog::CombinedLogger::init(vec![simplelog::TermLogger::new(
        simplelog::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    )])?;

    ggml::time::init();

    let t_main_start_us = ggml::time::us();

    let mut params = Params::parse();
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

    // load the model
    let mut vocabulary = llama::Vocabulary::default();
    let (mut ctx, load_state) = llama::Model::load(&params.model, 512, &mut vocabulary)?;

    let t_load_us;
    let model = {
        let t_start_us = ggml::time::us();
        let model = load_state.finish(&mut ctx)?;
        t_load_us = ggml::time::us() - t_start_us;
        model
    };

    let mut n_past = 0;

    let mut t_sample_us = 0;
    let mut t_predict_us = 0;

    let embd_inp = vocabulary.tokenize(&params.prompt, true);

    params.n_predict = params.n_predict.min(model.n_ctx() - embd_inp.len());

    log::info!("prompt: '{}'", params.prompt);
    log::info!("number of tokens in prompt = {}", embd_inp.len());
    for embedding in &embd_inp {
        log::info!(
            "{} -> '{}'",
            embedding,
            vocabulary
                .id_to_token
                .get(embedding)
                .expect("embedding missing")
        );
    }

    log::info!("sampling parameters: temp = {}, top_k = {}, top_p = {}, repeat_last_n = {}, repeat_penalty = {}", params.temperature, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);

    let mut logits: Vec<f32> = vec![];
    let mut embd: Vec<llama::VocabularyId> = vec![];

    // determine the required inference memory per token:
    let mut mem_per_token: usize = 0;
    model.evaluate(
        params.n_threads(),
        0,
        &[0, 1, 2, 3],
        &mut logits,
        &mut mem_per_token,
    )?;

    let last_n_size = params.repeat_last_n;
    let mut last_n_tokens = vec![0; last_n_size];

    let mut remaining_tokens = params.n_predict;
    let mut input_consumed = 0;

    while remaining_tokens > 0 {
        if !embd.is_empty() {
            let t_start_us = ggml::time::us();

            model.evaluate(
                params.n_threads(),
                n_past,
                &embd,
                &mut logits,
                &mut mem_per_token,
            )?;

            t_predict_us += ggml::time::us() - t_start_us;
        }

        n_past += embd.len();
        embd.clear();

        if embd_inp.len() <= input_consumed {
            // out of input, sample next token
            let n_vocab = model.n_vocab();

            let id = {
                let t_start_sample_us = ggml::time::us();

                let id = vocabulary.sample_top_p_top_k(
                    &logits[logits.len() - usize::try_from(n_vocab)?..],
                    &last_n_tokens,
                    params.repeat_penalty,
                    params.top_k,
                    params.top_p,
                    params.temperature,
                    &mut rng,
                );

                last_n_tokens.remove(0);
                last_n_tokens.push(id);

                t_sample_us += ggml::time::us() - t_start_sample_us;

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

                if embd.len() > params.n_batch {
                    break;
                }
            }
        }

        for id in &embd {
            log::info!("{}", vocabulary.id_to_token.get(id).expect("no token"));
        }

        if embd.last() == Some(&2) {
            log::info!("[end of text]");
            break;
        }
    }

    // report timing
    {
        let t_main_end_us = ggml::time::us();

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
