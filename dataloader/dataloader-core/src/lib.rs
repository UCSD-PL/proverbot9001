/* *********************************************************************** */
//
//    This file is part of Proverbot9001.
//
//    Proverbot9001 is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    Proverbot9001 is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
//
//    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
//
/* *********************************************************************** */

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::fs::File;

mod context_filter;
mod context_filter_ast;
mod features;
mod models;
mod paren_util;
mod scraped_data;
mod tokenizer;
use context_filter::*;
use features::*;
use models::features_dnn_evaluator::*;
use models::features_polyarg_predictor::*;
use models::goal_enc_evaluator::*;
use paren_util::parse_sexp_one_level;
use scraped_data::*;
use tokenizer::get_words;

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
            BoolTensor2D,
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
            BoolTensor2D,
            LongTensor2D,
            FloatTensor2D,
            LongTensor1D,
            LongTensor1D,
        ),
        (Vec<i64>, i64),
    )> {
        py.allow_threads(move || features_polyarg_tensors(args, filename, Some(meta)))
    }
    #[pyfn(m, "sample_fpa_batch")]
    fn sample_fpa_batch_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        context_batch: Vec<TacticContext>,
    ) -> (
        LongUnpaddedTensor3D,
        FloatUnpaddedTensor3D,
        LongTensor1D,
        LongTensor2D,
        BoolTensor2D,
        LongTensor2D,
        FloatTensor2D,
    ) {
        sample_fpa_batch(args, metadata, context_batch)
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
        BoolTensor2D,
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
    #[pyfn(m, "tokenize")]
    fn tokenize_fpa_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        term: String) -> LongTensor1D {
        tokenize_fpa(args, metadata, term)
    }
    #[pyfn(m, "get_premise_features")]
    pub fn get_premise_features_py(
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        goal: String,
        premise: String) -> FloatTensor1D {
        get_premise_features(args, metadata, goal, premise)
    }
    #[pyfn(m, "get_premise_features_size")]
    pub fn get_premise_features_size_py(
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata) -> i64 {
        get_premise_features_size(args, metadata)
    }
    #[pyfn(m, "decode_fpa_stem")]
    fn decode_fpa_stem_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        tac_idx: i64,
    ) -> String {
        decode_fpa_stem(&args, metadata, tac_idx)
    }
    #[pyfn(m, "encode_fpa_stem")]
    fn encode_fpa_stem_py(
        _py: Python,
        args: DataloaderArgs,
        metadata: PickleableFPAMetadata,
        tac_stem: String,
    ) -> i64 {
        encode_fpa_stem(&args, metadata, tac_stem)
    }
    #[pyfn(m, "decode_fpa_arg")]
    fn decode_fpa_arg_py(
        _py: Python,
        args: DataloaderArgs,
        _metadata: PickleableFPAMetadata,
        hyps: Vec<String>,
        goal: &str,
        arg_idx: i64,
    ) -> String {
        decode_fpa_arg(&args, hyps, goal, arg_idx)
    }
    #[pyfn(m, "encode_fpa_arg")]
    fn encode_fpa_arg_py(
        _py: Python,
        args: DataloaderArgs,
        _metadata: PickleableFPAMetadata,
        hyps: Vec<String>,
        goal: &str,
        arg: &str,
    ) -> i64 {
        encode_fpa_arg_unbounded(&args, hyps, goal, arg)
    }
    #[pyfn(m, "get_num_tokens")]
    fn get_num_tokens(_py: Python, metadata: PickleableFPAMetadata) -> i64 {
        let (_indexer, tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
        tokenizer.num_tokens()
    }
    #[pyfn(m, "fpa_get_num_possible_args")]
    fn get_num_possible_args(_py: Python, args: DataloaderArgs) -> i64 {
        fpa_get_num_possible_args(&args)
    }
    #[pyfn(m, "get_num_indices")]
    fn get_num_indices(_py: Python, metadata: PickleableFPAMetadata) -> (PickleableFPAMetadata, i64) {
        let (mut indexer, tokenizer, ftmap) = fpa_metadata_from_pickleable(metadata);
        indexer.freeze();
        let num_indices = indexer.num_indices();
        (fpa_metadata_to_pickleable((indexer, tokenizer, ftmap)), num_indices)
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
    #[pyfn(m, "get_fpa_words")]
    fn get_fpa_words(_py: Python, s: String) -> Vec<String> {
        get_words(&s).into_iter().map(|s| s.to_string()).collect()
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
    #[pyfn(m, "goal_enc_tokenize_goal")]
    fn _goal_enc_tokenize_goal(
        _py: Python,
        args: DataloaderArgs,
        metadata: &GoalEncMetadata,
        s: String,
    ) -> Vec<i64> {
        tokenize_goal(args, metadata, s)
    }
    #[pyfn(m, "scraped_tactics_from_file")]
    fn _scraped_tactics_from_file(
        _py: Python,
        filename: String,
        num_tactics: Option<usize>,
    ) -> PyResult<Vec<ScrapedTactic>> {
        let iter = scraped_from_file(
            File::open(filename)
                .map_err(|_err| PyErr::new::<exceptions::IOError, _>("Failed to open file"))?,
        )
        .flat_map(|datum| match datum {
            ScrapedData::Vernac(_) => None,
            ScrapedData::Tactic(t) => Some(t),
        });
        match num_tactics {
            Some(num) => Ok(iter.take(num).collect()),
            None => Ok(iter.collect()),
        }
    }

    #[pyfn(m, "tactic_transitions_from_file")]
    fn _tactic_transitions_from_file(
        _py: Python,
        args: &DataloaderArgs,
        filename: String,
        num_tactics: usize,
    ) -> PyResult<Vec<ScrapedTransition>> {
        let filter = parse_filter(&args.context_filter);
        let raw_iter = scraped_from_file(
            File::open(filename)
                .map_err(|_err| PyErr::new::<exceptions::IOError, _>("Failed to open file"))?,
        );
        let transition_iter = scraped_transition_iter(raw_iter);
        let filtered_iter = transition_iter
            .filter(|transition| apply_filter(args, &filter, &transition.scraped_before()));
        Ok(filtered_iter.take(num_tactics).collect::<Vec<_>>())
    }

    #[pyfunction]
    pub fn sample_context_features(
        args: &DataloaderArgs,
        metadata: PickleableFPAMetadata,
        relevant_lemmas: Vec<String>,
        prev_tactics: Vec<String>,
        hypotheses: Vec<String>,
        goal: String,
    ) -> (LongTensor1D, FloatTensor1D) {
        crate::features::sample_context_features(
            args,
            &TokenMap::from_dicts(metadata.2),
            &relevant_lemmas,
            &prev_tactics,
            &hypotheses,
            &goal,
        )
    }

    #[pyfunction]
    pub fn features_vocab_sizes(tmap: TokenMap) -> (Vec<i64>, i64) {
        (tmap.word_features_sizes(), VEC_FEATURES_SIZE)
    }

    #[pyfunction]
    pub fn tmap_to_picklable(tmap: TokenMap) -> PickleableTokenMap {
        tmap.to_dicts()
    }

    #[pyfunction]
    pub fn tmap_from_picklable(picklable: PickleableTokenMap) -> TokenMap {
        TokenMap::from_dicts(picklable)
    }
    #[pyfunction]
    pub fn rust_parse_sexp_one_level<'a>(sexpstr: &'a str) -> Vec<&'a str> {
        parse_sexp_one_level(sexpstr)
    }
    m.add_wrapped(wrap_pyfunction!(features_vocab_sizes))?;
    m.add_wrapped(wrap_pyfunction!(tmap_from_picklable))?;
    m.add_wrapped(wrap_pyfunction!(tmap_to_picklable))?;
    m.add_wrapped(wrap_pyfunction!(sample_context_features))?;
    m.add_wrapped(wrap_pyfunction!(rust_parse_sexp_one_level))?;
    m.add_class::<TokenMap>()?;
    m.add_class::<DataloaderArgs>()?;
    m.add_class::<GoalEncMetadata>()?;
    m.add_class::<ScrapedTactic>()?;
    m.add_class::<ProofContext>()?;
    m.add_class::<ScrapedTransition>()?;
    m.add_class::<Obligation>()?;
    m.add_class::<TacticContext>()?;
    Ok(())
}
