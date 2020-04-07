#![feature(vec_remove_item)]
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::fs::File;

mod scraped_data;
use scraped_data::*;
mod features;
use features::{context_features, PickleableTokenMap, TokenMap, VEC_FEATURES_SIZE};
mod models;
use models::features_polyarg_predictor::*;
use models::goal_enc_evaluator::*;
use models::evaluator_common::*;
mod context_filter;
mod context_filter_ast;
mod paren_util;
mod tokenizer;

#[macro_use]
extern crate lazy_static;

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
    #[pyfn(m, "features_polyarg_tensors")]
    fn parallel_features_polyarg_tensors_py(
        py: Python,
        args: DataloaderArgs,
        filename: String,
    ) -> PyResult<(
        PickleableFPAMetadata,
        (
            LongUnpaddedTensor3D,
            FloatUnpaddedTensor3D,
            LongTensor1D,
            LongTensor2D,
            LongTensor2D,
            FloatTensor2D,
            LongTensor1D,
            LongTensor1D,
        ),
        (Vec<i64>, i64),
    )> {
        py.allow_threads(move || features_polyarg_tensors(args, filename, None))
    }
    #[pyfn(m, "features_polyarg_tensors_with_meta")]
    fn parallel_features_polyarg_tensors_with_meta(
        py: Python,
        args: DataloaderArgs,
        filename: String,
        meta: PickleableFPAMetadata,
    ) -> PyResult<(
        PickleableFPAMetadata,
        (
            LongUnpaddedTensor3D,
            FloatUnpaddedTensor3D,
            LongTensor1D,
            LongTensor2D,
            LongTensor2D,
            FloatTensor2D,
            LongTensor1D,
            LongTensor1D,
        ),
        (Vec<i64>, i64),
    )> {
        py.allow_threads(move || features_polyarg_tensors(args, filename, Some(meta)))
    }
    #[pyfn(m, "sample_fpa")]
    fn sample_fpa_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        relevant_lemmas: Vec<String>,
        prev_tactics: Vec<String>,
        hypotheses: Vec<String>,
        goal: String,
    ) -> (
        LongUnpaddedTensor3D,
        FloatUnpaddedTensor3D,
        LongTensor1D,
        LongTensor2D,
        LongTensor2D,
        FloatTensor2D,
    ) {
        sample_fpa(
            args,
            metadata,
            relevant_lemmas,
            prev_tactics,
            hypotheses,
            goal,
        )
    }
    #[pyfn(m, "decode_fpa_result")]
    fn decode_fpa_result_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        hyps: Vec<String>,
        goal: &str,
        tac_idx: i64,
        arg_idx: i64,
    ) -> String {
        decode_fpa_result(args, metadata, hyps, goal, tac_idx, arg_idx)
    }
    #[pyfn(m, "get_num_tokens")]
    fn get_num_tokens(_py: Python, metadata: PickleableFPAMetadata) -> i64 {
        let (_indexer, tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
        tokenizer.num_tokens()
    }
    #[pyfn(m, "get_num_indices")]
    fn get_num_indices(_py: Python, metadata: PickleableFPAMetadata) -> i64 {
        let (indexer, _tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
        indexer.num_indices()
    }
    #[pyfn(m, "get_word_feature_vocab_sizes")]
    fn get_word_feature_vocab_sizes(_py: Python, metadata: PickleableFPAMetadata) -> Vec<i64> {
        let (_indexer, _tokenizer, ftmap) = fpa_metadata_from_pickleable(metadata);
        ftmap.word_features_sizes()
    }
    #[pyfn(m, "get_vec_features_size")]
    fn get_vec_features_size(_py: Python, _metadata: PickleableFPAMetadata) -> i64 {
        VEC_FEATURES_SIZE
    }

    #[pyfn(m, "goals_to_total_distances_tensors")]
    fn _goals_to_total_distances_tensors(
        py: Python,
        args: DataloaderArgs,
        filename: String,
    ) -> PyResult<(GoalEncMetadata, LongTensor2D, FloatTensor1D)> {
        py.allow_threads(move || {
            Ok(goals_to_total_distances_tensors(args, filename, None)
               .map_err(|err| PyErr::new::<exceptions::IOError, _>(err))?)
        })
    }
    #[pyfn(m, "goals_to_total_distances_tensors_with_meta")]
    fn _goals_to_total_distances_tensors_with_meta(
        py: Python,
        args: DataloaderArgs,
        filename: String,
        metadata: &GoalEncMetadata,
    ) -> PyResult<(LongTensor2D, FloatTensor1D)> {
        py.allow_threads(move || {
            let (_, goals, outputs) =
                goals_to_total_distances_tensors(args, filename, Some(metadata))
                .map_err(|err| PyErr::new::<exceptions::IOError, _>(err))?;
            Ok((goals, outputs))
        })
    }
    #[pyfn(m, "goal_enc_get_num_tokens")]
    fn _goal_enc_get_num_tokens(_py: Python, metadata: &GoalEncMetadata) -> i64 {
        goal_enc_get_num_tokens(metadata)
    }

    m.add_wrapped(wrap_pyfunction!(features_vocab_sizes))?;
    m.add_wrapped(wrap_pyfunction!(tmap_from_picklable))?;
    m.add_wrapped(wrap_pyfunction!(tmap_to_picklable))?;
    m.add_wrapped(wrap_pyfunction!(sample_context_features))?;
    m.add_class::<TokenMap>()?;
    m.add_class::<DataloaderArgs>()?;
    m.add_class::<GoalEncMetadata>()?;
    Ok(())
}
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
