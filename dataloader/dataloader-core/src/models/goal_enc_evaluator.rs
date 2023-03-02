use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::ToPyObject;
use rayon::prelude::*;
use std::fs::File;

use crate::context_filter::filter_data_by_key;
use crate::models::evaluator_common::*;
use crate::scraped_data::*;
use crate::tokenizer::{normalize_sentence_length, Tokenizer};

#[pyclass(module = "dataloader")]
pub struct GoalEncMetadata {
    tokenizer: Option<Tokenizer>,
}

#[pymethods]
impl GoalEncMetadata {
    #[new]
    fn new() -> Self {
        GoalEncMetadata { tokenizer: None }
    }
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(
            py,
            &serialize(self.tokenizer.as_ref().expect("No tokenizer")).unwrap(),
        )
        .to_object(py))
    }
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        self.tokenizer = Some(deserialize(state.extract::<&PyBytes>(py)?.as_bytes()).unwrap());
        Ok(())
    }
}

pub fn goals_to_total_distances_tensors_rs(
    args: DataloaderArgs,
    filename: String,
    metadata: Option<&GoalEncMetadata>,
) -> Result<(GoalEncMetadata, LongTensor2D, FloatTensor1D), String> {
    let raw_data =
        scraped_from_file(File::open(filename).map_err(|_err| "Failed to open file")?).collect();
    let distanced = tactic_distances(raw_data);
    let filtered_data = filter_data_by_key(args.max_length, &args.context_filter, distanced, |distanced| {
        &(*distanced).0
    });
    let (tactics, distances): (Vec<ScrapedTactic>, Vec<usize>) = filtered_data.into_iter().unzip();
    let tokenizer = match metadata {
        Some(meta) => meta.tokenizer.as_ref().expect("No tokenizer").clone(),
        None => {
            let use_unknowns = true;
            let num_reserved_tokens = 2;
            Tokenizer::new(use_unknowns, num_reserved_tokens, &args.keywords_file, &args.paths_file)
        }
    };
    let tokenized_goals: Vec<Vec<i64>> = tactics
        .par_iter()
        .map(|tac| truncate_to_length(tokenizer.tokenize(&tac.context.focused_goal()), args.max_length))  
        .collect();
    Ok((
        GoalEncMetadata {
            tokenizer: Some(tokenizer),
        },
        tokenized_goals,
        normalize_distances(args.max_distance, distances),
    ))
}

pub fn goal_enc_get_num_tokens_rs(metadata: &GoalEncMetadata) -> i64 {
    metadata
        .tokenizer
        .as_ref()
        .expect("No tokenizer")
        .num_tokens()
}

pub fn goal_enc_get_num_paths_tokens_rs(metadata: &GoalEncMetadata) -> i64 {
    metadata
        .tokenizer
        .as_ref()
        .expect("No tokenizer")
        .num_paths_tokens()
}

pub fn tokenize_goal(args: DataloaderArgs, metadata: &GoalEncMetadata, goal: String) -> Vec<i64> {
    normalize_sentence_length(
        metadata
            .tokenizer
            .as_ref()
            .expect("No tokenizer")
            .tokenize(&&goal),
        args.max_length,
        1,
    )
}

fn truncate_to_length(mut sentence: Vec<i64>, max_length: usize) -> Vec<i64> {
    sentence.truncate(max_length);
    sentence
}
