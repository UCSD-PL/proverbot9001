use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use std::fs::File;
use rayon::prelude::*;
use bincode::{deserialize, serialize};

use crate::context_filter::filter_data_by_key;
use crate::scraped_data::*;
use crate::tokenizer::{Tokenizer, normalize_sentence_length};
use crate::models::evaluator_common::*;

#[pyclass(module="dataloader")]
pub struct GoalEncMetadata {
    tokenizer: Tokenizer,
}

#[pymethods]
impl GoalEncMetadata {
    #[new]
    fn new(obj: &PyRawObject, keywords_filepath: String) {
        obj.init({GoalEncMetadata {tokenizer:
            Tokenizer::new(true, 2, &keywords_filepath)}})
    }
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(&self.tokenizer).unwrap()).to_object(py))
    }
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        self.tokenizer = deserialize(state.extract::<&PyBytes>(py)?.as_bytes()).unwrap();
        Ok(())
    }
}

pub fn goals_to_total_distances_tensors(
    args: DataloaderArgs,
    filename: String,
    metadata: Option<&GoalEncMetadata>,
) -> Result<(GoalEncMetadata, LongTensor2D, FloatTensor1D), String> {
    let raw_data =
        scraped_from_file(File::open(filename).map_err(|_err| "Failed to open file")?)
        .collect();
    let distanced = tactic_distances(raw_data);
    let filtered_data = filter_data_by_key(&args, &args.context_filter,
                                           distanced,
                                           |distanced| &(*distanced).0);
    let (tactics, distances) : (Vec<ScrapedTactic>, Vec<usize>) =
        filtered_data.into_iter().unzip();
    let tokenizer = match metadata {
        Some(meta) => meta.tokenizer.clone(),
        None => {
            let use_unknowns = true;
            let num_reserved_tokens = 2;
            Tokenizer::new(use_unknowns, num_reserved_tokens, &args.keywords_file)
        }
    };
    let tokenized_goals = tactics
        .par_iter()
        .map(|tac| {
            normalize_sentence_length(tokenizer.tokenize(&tac.prev_goal),
                                      args.max_length, 1)
        }).collect();

    Ok((GoalEncMetadata{tokenizer},
        tokenized_goals,
        normalize_distances(args.max_distance, distances)))
}

pub fn goal_enc_get_num_tokens(metadata: &GoalEncMetadata) -> i64 {
    metadata.tokenizer.num_tokens()
}
