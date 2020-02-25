use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::cmp::min;
use std::fs::File;

mod scraped_data;
use scraped_data::*;
mod features;
use features::{context_features, PickleableTokenMap, TokenMap, VEC_FEATURES_SIZE};

extern crate rayon;
#[pymodule]
fn dataloader(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "features_to_total_distances_tensors")]
    fn parallel_features_to_total_distances_tensors(
        py: Python,
        args: DataloaderArgs,
        filename: String,
    ) -> PyResult<(
        TokenMap,
        LongTensor2D,
        FloatTensor2D,
        FloatTensor2D,
        Vec<i64>,
        i64,
    )> {
        py.allow_threads(move || features_to_total_distances_tensors(args, filename, None))
    }
    #[pyfn(m, "features_to_total_distances_tensors_with_map")]
    fn parallel_features_to_total_distances_tensors_with_map(
        py: Python,
        args: DataloaderArgs,
        filename: String,
        map: TokenMap,
    ) -> PyResult<(
        TokenMap,
        LongTensor2D,
        FloatTensor2D,
        FloatTensor2D,
        Vec<i64>,
        i64,
    )> {
        py.allow_threads(move || features_to_total_distances_tensors(args, filename, Some(map)))
    }
    m.add_wrapped(wrap_pyfunction!(features_vocab_sizes))?;
    m.add_wrapped(wrap_pyfunction!(tmap_from_picklable))?;
    m.add_wrapped(wrap_pyfunction!(tmap_to_picklable))?;
    m.add_wrapped(wrap_pyfunction!(sample_context_features))?;
    m.add_class::<TokenMap>()?;
    m.add_class::<DataloaderArgs>()?;
    Ok(())
}
#[pyfunction]
fn features_to_total_distances_tensors(
    args: DataloaderArgs,
    filename: String,
    map: Option<TokenMap>,
) -> PyResult<(
    TokenMap,
    LongTensor2D,
    FloatTensor2D,
    FloatTensor2D,
    Vec<i64>,
    i64,
)> {
    match File::open(filename) {
        Result::Ok(file) => {
            let scraped = scraped_from_file(file).collect();
            let distanced = tactic_distances(scraped);
            let (tactics, distances): (Vec<ScrapedTactic>, Vec<usize>) =
                distanced.into_iter().unzip();
            let outputs = normalize_distances(args.max_distance, distances)
                .into_iter()
                .map(|distance| vec![distance])
                .collect();
            let tmap = match map {
                Some(m) => m,
                None => TokenMap::initialize(&tactics, args.num_keywords),
            };
            let (word_features, float_features) = context_features(args, &tmap, tactics);
            let word_features_sizes = tmap.word_features_sizes();

            Ok((
                tmap,
                word_features,
                float_features,
                outputs,
                word_features_sizes,
                VEC_FEATURES_SIZE,
            ))
        }
        Result::Err(_err) => Err(PyErr::new::<exceptions::TypeError, _>(
            "Failed to open file",
        )),
    }
}
#[pyfunction]
fn sample_context_features(
    args: DataloaderArgs,
    tmap: &TokenMap,
    relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    hypotheses: Vec<String>,
    goal: String,
) -> (LongTensor1D, FloatTensor1D) {
    features::sample_context_features(args, tmap, relevant_lemmas, prev_tactics, hypotheses, goal)
}

#[pyfunction]
fn features_vocab_sizes(tmap: TokenMap) -> (Vec<i64>, i64) {
    (tmap.word_features_sizes(), VEC_FEATURES_SIZE)
}

#[pyfunction]
fn tmap_to_picklable(tmap: TokenMap) -> PickleableTokenMap {
    tmap.to_dicts()
}

#[pyfunction]
fn tmap_from_picklable(picklable: PickleableTokenMap) -> TokenMap {
    TokenMap::from_dicts(picklable)
}

fn normalize_distances(max_distance: usize, distances: Vec<usize>) -> Vec<f64> {
    distances
        .into_iter()
        .map(|x| (min(x, max_distance) as f64) / (max_distance as f64))
        .collect()
}

fn tactic_distances(scraped_data: Vec<ScrapedData>) -> Vec<(ScrapedTactic, usize)> {
    let mut in_proof = false;
    let mut interaction_buffer = Vec::new();
    let mut blocks = Vec::new();

    for interaction in scraped_data {
        match interaction {
            ScrapedData::Tactic(tac) => {
                if !in_proof {
                    interaction_buffer.clear();
                    in_proof = true;
                }
                interaction_buffer.push(tac)
            }
            ScrapedData::Vernac(_cmd) => {
                if in_proof {
                    blocks.push(interaction_buffer.clone());
                    in_proof = false;
                }
            }
        }
    }

    let mut result = Vec::new();

    for block in blocks {
        let block_len = block.len();
        let mut distanced_block: Vec<(ScrapedTactic, usize)> = block
            .into_iter()
            .enumerate()
            .map(|(idx, val)| (val, block_len - idx))
            .collect();
        result.append(&mut distanced_block);
    }
    return result;
}
