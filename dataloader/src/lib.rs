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

mod scraped_data;
mod context_filter;
mod context_filter_ast;
mod paren_util;
mod tokenizer;
mod features;
mod models;
use scraped_data::*;
use features::*;
use models::features_polyarg_predictor::*;
use models::goal_enc_evaluator::*;
use models::features_dnn_evaluator::*;

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
    #[pyfn(m, "goal_enc_tokenize_goal")]
    fn _goal_enc_tokenize_goal(_py: Python, args: DataloaderArgs,
                               metadata: &GoalEncMetadata,
                               s: String) -> Vec<i64> {
        tokenize_goal(args, metadata, s)
    }

    #[pyfunction]
    pub fn sample_context_features(
        args: &DataloaderArgs,
        tmap: &TokenMap,
        relevant_lemmas: Vec<String>,
        prev_tactics: Vec<String>,
        hypotheses: Vec<String>,
        goal: String,
    ) -> (LongTensor1D, FloatTensor1D) {
        crate::features::sample_context_features(args, tmap, &relevant_lemmas, &prev_tactics, &hypotheses, &goal)
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
    m.add_wrapped(wrap_pyfunction!(features_vocab_sizes))?;
    m.add_wrapped(wrap_pyfunction!(tmap_from_picklable))?;
    m.add_wrapped(wrap_pyfunction!(tmap_to_picklable))?;
    m.add_wrapped(wrap_pyfunction!(sample_context_features))?;
    m.add_class::<TokenMap>()?;
    m.add_class::<DataloaderArgs>()?;
    m.add_class::<GoalEncMetadata>()?;
    Ok(())
}
