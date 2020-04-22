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

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};

use crate::scraped_data::*;
use crate::tokenizer::get_words;
use edit_distance::edit_distance;
use rayon::prelude::*;

pub const VEC_FEATURES_SIZE: i64 = 2;

pub fn context_features(
    args: DataloaderArgs,
    tmap: &TokenMap,
    data: Vec<ScrapedTactic>,
) -> (LongTensor2D, FloatTensor2D) {
    let (best_hyps, best_hyp_scores): (Vec<&str>, Vec<f64>) = data
        .par_iter()
        .map(|scraped| {
            best_scored_hyp(
                args.max_string_distance,
                args.max_length,
                &scraped.prev_hyps,
                &scraped.prev_goal,
            )
        })
        .unzip();

    let word_features = data
        .iter()
        .zip(best_hyps)
        .map(|(scraped, best_hyp): (&ScrapedTactic, &str)| {
            vec![
                prev_tactic_feature(tmap, &scraped.prev_tactics),
                goal_head_feature(tmap, &scraped.prev_goal),
                hyp_head_feature(tmap, best_hyp),
            ]
        })
        .collect();

    let vec_features = best_hyp_scores
        .into_iter()
        .zip(data.into_iter())
        .map(|(score, datum)|
             vec![
                 (std::cmp::min(get_words(&datum.prev_goal).len(), 100) as f64) / 100.0,
                 (std::cmp::min(datum.prev_hyps.len(), 20) as f64) / 20.0
             ])
        .collect();

    (word_features, vec_features)
}

pub fn sample_context_features(
    args: DataloaderArgs,
    tmap: &TokenMap,
    _relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    hypotheses: Vec<String>,
    goal: String,
) -> (LongTensor1D, FloatTensor1D) {
    let (best_hyp, best_score) =
        best_scored_hyp(args.max_distance, args.max_length, &hypotheses, &goal);
    let word_features = vec![
        prev_tactic_feature(tmap, &prev_tactics),
        goal_head_feature(tmap, &goal),
        hyp_head_feature(tmap, best_hyp),
    ];
    let vec_features = vec![
        (std::cmp::min(get_words(&goal).len(), 100) as f64) / 100.0,
        (std::cmp::min(hypotheses.len(), 20) as f64) / 20.0
    ];
    (word_features, vec_features)
}

// index of the previous tactic, or zero if it's not
// in the index table, or there is no previous tactic.
pub fn prev_tactic_feature(tmap: &TokenMap, prev_tactics: &Vec<String>) -> i64 {
    match prev_tactics
        .last()
        .and_then(|tac| get_stem(tac))
        .and_then(|first_token| tmap.tactic_to_index.get(&first_token))
    {
        Some(idx) => (idx + 1) as i64,
        None => 0,
    }
}

pub fn goal_head_feature(tmap: &TokenMap, goal: &str) -> i64 {
    match goal.split_whitespace().next() {
        None => 0,
        Some(first_token) => match tmap.goal_token_to_index.get(first_token) {
            None => 1,
            Some(idx) => (idx + 2) as i64,
        },
    }
}

pub fn hyp_head_feature(tmap: &TokenMap, best_hyp: &str) -> i64 {
    match best_hyp
        .split_whitespace()
        .next()
        .and_then(|first_token| tmap.hyp_token_to_index.get(first_token))
    {
        Some(idx) => (idx + 1) as i64,
        None => 0,
    }
}

#[pyclass(dict, module = "dataloader")]
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenMap {
    tactic_to_index: HashMap<String, usize>,
    goal_token_to_index: HashMap<String, usize>,
    hyp_token_to_index: HashMap<String, usize>,
}

pub type PickleableTokenMap = (
    HashMap<String, usize>,
    HashMap<String, usize>,
    HashMap<String, usize>,
);

impl<'source> pyo3::FromPyObject<'source> for TokenMap {
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
                    .map(|hyp| hyp.split_whitespace().next().unwrap().to_string())
            }),
            count,
        );
        let index_to_goal_token = index_common(
            init_data
                .iter()
                .flat_map(|scraped| scraped.prev_goal.split_whitespace().next())
                .map(|s| s.to_string()),
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
        vec![
            (self.tactic_to_index.len() + 1) as i64,
            (self.goal_token_to_index.len() + 2) as i64,
            (self.hyp_token_to_index.len() + 1) as i64,
        ]
    }

    pub fn to_dicts(&self) -> PickleableTokenMap {
        (
            self.tactic_to_index.clone(),
            self.goal_token_to_index.clone(),
            self.hyp_token_to_index.clone(),
        )
    }

    pub fn from_dicts(dicts: PickleableTokenMap) -> TokenMap {
        TokenMap {
            tactic_to_index: dicts.0,
            goal_token_to_index: dicts.1,
            hyp_token_to_index: dicts.2,
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ScoredString<'a> {
    score: i64,
    contents: &'a str,
}

fn index_common<'a>(items: impl Iterator<Item = String>, n: usize) -> Vec<String> {
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

pub fn score_hyps<'a>(
    max_distance: usize,
    max_length: usize,
    hyps: &Vec<String>,
    goal: &String,
) -> Vec<f64> {
    assert!(max_distance != 0);
    let truncated_goal = get_words(goal).into_iter().take(max_length).collect::<Vec<_>>().join(" ");
    hyps.into_iter()
        .map(|hyp| {
            let trunc_hyp_after_colon =
                get_words(hyp.splitn(2, ":").collect::<Vec<&str>>()[1])
                    .into_iter()
                    .take(max_length).collect::<Vec<_>>()
                    .join(" ");
            let score = edit_distance(&trunc_hyp_after_colon, &truncated_goal);
            (score as f64) / (max_distance as f64)
        })
        .collect()
}

fn best_scored_hyp<'a>(
    max_distance: usize,
    max_length: usize,
    hyps: &'a Vec<String>,
    goal: &String,
) -> (&'a str, f64) {
    let mut best_hyp = "";
    let mut best_score = max_distance;
    let mut truncated_goal = goal.clone();
    truncated_goal.truncate(max_length);
    for hyp in hyps.iter() {
        let mut hyp_after_colon = hyp.split(":").collect::<Vec<&str>>()[1].to_string();
        hyp_after_colon.truncate(max_length);
        let score = edit_distance(&hyp_after_colon, &truncated_goal);
        if score < best_score {
            best_score = score;
            best_hyp = &hyp;
        }
    }
    (best_hyp, (best_score as f64) / (max_distance as f64))
}
