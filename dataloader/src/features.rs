use std::collections::{BinaryHeap, HashMap};
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;

use edit_distance::edit_distance;
use crate::scraped_data::*;
use rayon::prelude::*;

pub const VEC_FEATURES_SIZE: i64 = 1;

pub fn context_features(tmap: &TokenMap, data: Vec<ScrapedTactic>) -> (LongTensor2D, FloatTensor2D) {
    let (best_hyps, best_hyp_scores): (Vec<&str>, Vec<f64>) =
        data.par_iter().map(|scraped| best_scored_hyp(&scraped.prev_hyps, &scraped.prev_goal)).unzip();

    let word_features = data
        .iter()
        .zip(best_hyps)
        .map(|(scraped, best_hyp): (&ScrapedTactic, &str)| {
            vec![prev_tactic_feature(tmap, &scraped.prev_tactics),
                 goal_head_feature(tmap, &scraped.prev_goal),
                 hyp_head_feature(tmap, best_hyp)]
        })
        .collect();

    let vec_features = best_hyp_scores
        .into_iter()
        .map(|score| vec![score])
        .collect();

    (word_features, vec_features)
}

pub fn sample_context_features(tmap: &TokenMap,
                               _relevant_lemmas: Vec<String>,
                               prev_tactics: Vec<String>,
                               hypotheses: Vec<String>,
                               goal: String) -> (LongTensor1D, FloatTensor1D) {
    let (best_hyp, best_score) = best_scored_hyp(&hypotheses, &goal);
    let word_features = vec![prev_tactic_feature(tmap, &prev_tactics),
                             goal_head_feature(tmap, &goal),
                             hyp_head_feature(tmap, best_hyp)];
    let vec_features = vec![best_score];
    (word_features, vec_features)
}

// index of the previous tactic, or zero if it's not
// in the index table, or there is no previous tactic.
fn prev_tactic_feature(tmap: &TokenMap, prev_tactics : &Vec<String>) -> i64 {
    match prev_tactics
        .last()
        .map(|tac| get_stem(tac).expect("Couldn't get the first word of the tactic"))
        .and_then(|first_token| tmap.tactic_to_index.get(first_token))
    {
        Some(idx) => (idx + 1) as i64,
        None => 0,
    }
}

fn goal_head_feature(tmap: &TokenMap, goal : &str) -> i64 {
    match goal
        .split_whitespace()
        .next()
        .and_then(|first_token| tmap.goal_token_to_index.get(first_token))
    {
        Some(idx) => (idx + 1) as i64,
        None => 0,
    }
}

fn hyp_head_feature(tmap: &TokenMap, best_hyp : &str) -> i64 {
    match best_hyp
        .split_whitespace()
        .next()
        .and_then(|first_token| tmap.hyp_token_to_index.get(first_token))
    {
        Some(idx) => (idx + 1) as i64,
        None => 0,
    }
}

fn get_stem<'a>(full_tactic: &'a str) -> Option<&'a str> {
    full_tactic
        .split(|c| c == '.' || char::is_whitespace(c))
        .next()
}

#[pyclass(dict, module="dataloader")]
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenMap {
    tactic_to_index: HashMap<String, usize>,
    goal_token_to_index: HashMap<String, usize>,
    hyp_token_to_index: HashMap<String, usize>,
}

pub type PickleableTokenMap = (HashMap<String, usize>,
                               HashMap<String, usize>,
                               HashMap<String, usize>);

impl<'source> pyo3::FromPyObject<'source> for TokenMap{
    fn extract(ob: &'source pyo3::types::PyAny) -> pyo3::PyResult<TokenMap> {
        let cls: &TokenMap = pyo3::PyTryFrom::try_from(ob)?;
        Ok(cls.clone())
    }
}

fn flip_vec<T>(vec: Vec<T>) -> HashMap<T, usize>
where
    T: std::hash::Hash + std::cmp::Eq,
{
    let mut result = HashMap::new();
    for (idx, val) in vec.into_iter().enumerate() {
        result.insert(val, idx);
    }
    result
}

impl TokenMap {
    pub fn initialize(init_data: &Vec<ScrapedTactic>, count: usize) -> TokenMap {
        let index_to_tactic = index_common(
            init_data
                .iter()
                .flat_map(|scraped| get_stem(&scraped.tactic)),
            count,
        );
        let index_to_hyp_token = index_common(
            init_data.iter().flat_map(|scraped| {
                scraped
                    .prev_hyps
                    .iter()
                    .map(|hyp| hyp.split_whitespace().next().unwrap())
            }),
            count,
        );
        let index_to_goal_token = index_common(
            init_data
                .iter()
                .flat_map(|scraped| scraped.prev_goal.split_whitespace().next()),
            count,
        );

        TokenMap {
            tactic_to_index: flip_vec(index_to_tactic),
            goal_token_to_index: flip_vec(index_to_goal_token),
            hyp_token_to_index: flip_vec(index_to_hyp_token),
        }
    }
    pub fn word_features_sizes(&self) -> Vec<i64> {
        // Add one to each of these to account for the UNKNOWN token
        vec![(self.tactic_to_index.len() + 1) as i64,
             (self.goal_token_to_index.len() + 1) as i64,
             (self.hyp_token_to_index.len() + 1) as i64]
    }

    pub fn to_dicts(&self) -> PickleableTokenMap {
        (self.tactic_to_index.clone(), self.goal_token_to_index.clone(), self.hyp_token_to_index.clone())
    }

    pub fn from_dicts(dicts : PickleableTokenMap) -> TokenMap {
        TokenMap{tactic_to_index: dicts.0,
                 goal_token_to_index: dicts.1,
                 hyp_token_to_index: dicts.2}
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ScoredString<'a> {
    score: i64,
    contents: &'a str,
}

fn index_common<'a>(items: impl Iterator<Item = &'a str>, n: usize) -> Vec<String> {
    let mut counts = HashMap::new();
    for item in items {
        *counts.entry(item).or_insert(0) += 1;
    }
    let mut heap: BinaryHeap<ScoredString> = counts
        .iter()
        .map(|(s, c)| ScoredString {
            score: *c,
            contents: s,
        })
        .collect();
    let mut result = Vec::new();
    for _ in 0..n {
        match heap.pop() {
            Some(v) => result.push(v.contents.to_owned()),
            None => break,
        }
    }
    result
}

fn best_scored_hyp<'a>(hyps: &'a Vec<String>, goal: &String) -> (&'a str, f64) {
    let mut best_hyp = "";
    let mut best_score = std::usize::MAX;
    for hyp in hyps.iter() {
        let score = edit_distance(&hyp, &goal);
        if score < best_score {
            best_score = score;
            best_hyp = &hyp;
        }
    }
    (best_hyp, (best_score as f64) / (std::usize::MAX as f64))
}
