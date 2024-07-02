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
use serde_json;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use indicatif::{ProgressBar, ProgressIterator, ParallelProgressIterator, ProgressStyle, ProgressFinish};

use crate::scraped_data::*;
// use crate::tokenizer::get_symbols;
use rayon::prelude::*;

use gestalt_ratio::gestalt_ratio;

pub const VEC_FEATURES_SIZE: i64 = 1;

pub fn context_features(
    _args: &DataloaderArgs,
    tmap: &TokenMap,
    data: &Vec<ScrapedTactic>,
) -> (LongTensor2D, FloatTensor2D) {
    let my_bar_style = ProgressStyle::with_template("{msg}: {wide_bar} [{elapsed}/{eta}]").unwrap();
    let length: u64 = data.len().try_into().unwrap();
    let (best_hyps, best_hyp_scores): (Vec<&str>, Vec<f64>) = data
        .par_iter()
        .map(|scraped| {
            best_scored_hyp(
                &scraped.context.focused_hyps(),
                &scraped.context.focused_goal(),
            )
        })
        .progress_with(ProgressBar::new(length).with_message("Finding best scoring hypotheses and their scores")
                                               .with_style(my_bar_style.clone())
                                               .with_finish(ProgressFinish::AndLeave))
        .unzip();

    let word_features = data
        .iter()
        .zip(best_hyps)
        .map(|(scraped, best_hyp): (&ScrapedTactic, &str)| {
            vec![
                prev_tactic_feature(tmap, &scraped.prev_tactics),
                goal_head_feature(tmap, &scraped.context.focused_goal()),
                hyp_head_feature(tmap, best_hyp),
            ]
        })
        .progress_with(ProgressBar::new(length).with_message("Gathering word features")
                                               .with_style(my_bar_style.clone())
                                               .with_finish(ProgressFinish::AndLeave))
        .collect();

    let vec_features = best_hyp_scores
        .into_iter()
        .zip(data.iter())
        .map(|(score, _datum)| {
            vec![
                score,
                // (std::cmp::min(get_symbols(&datum.context.focused_goal()).len(), 100) as f64) / 100.0,
                // (std::cmp::min(datum.context.focused_hyps().len(), 20) as f64) / 20.0
            ]
        })
        .progress_with(ProgressBar::new(length).with_message("Gathering vec features")
                                               .with_style(my_bar_style.clone())
                                               .with_finish(ProgressFinish::AndLeave))
        .collect();

    (word_features, vec_features)
}

pub fn sample_context_features_rs(
    _args: &DataloaderArgs,
    tmap: &TokenMap,
    _relevant_lemmas: &Vec<String>,
    prev_tactics: &Vec<String>,
    hypotheses: &Vec<String>,
    goal: &String,
) -> (LongTensor1D, FloatTensor1D) {
    let (best_hyp, best_score) = best_scored_hyp(
        &hypotheses,
        &goal,
    );
    let word_features = vec![
        prev_tactic_feature(tmap, &prev_tactics),
        goal_head_feature(tmap, &goal),
        hyp_head_feature(tmap, best_hyp),
    ];
    let vec_features = vec![
        best_score, // (std::cmp::min(get_symbols(&goal).len(), 100) as f64) / 100.0,
                   // (std::cmp::min(hypotheses.len(), 20) as f64) / 20.0
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
                    .context
                    .focused_hyps()
                    .iter()
                    .map(|hyp| hyp.split_whitespace().next().unwrap().to_string())
            }),
            count,
        );
        let index_to_goal_token = index_common(
            init_data
                .iter()
                .flat_map(|scraped| scraped.context.focused_goal().split_whitespace().next())
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
    pub fn prev_tactic_vocab_size(&self) -> i64 {
        return (self.tactic_to_index.len() + 1) as i64
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

    pub fn save_to_text(&self, filename: &str) {
        let mut index_to_tactic = vec![""; self.tactic_to_index.len()];
        for (tactic, index) in self.tactic_to_index.iter() {
            assert!(
                index < &self.tactic_to_index.len(),
                "index is {}, but there are only {} tactics",
                index,
                self.tactic_to_index.len()
            );
            index_to_tactic[*index] = tactic;
        }
        let mut index_to_goal_token = vec![""; self.goal_token_to_index.len()];
        for (goal_token, index) in self.goal_token_to_index.iter() {
            assert!(
                index < &self.goal_token_to_index.len(),
                "index is {}, but there are only {} goal tokens",
                index,
                self.goal_token_to_index.len()
            );
            index_to_goal_token[*index] = goal_token;
        }
        let mut index_to_hyp_token = vec![""; self.hyp_token_to_index.len()];
        for (hyp_token, index) in self.hyp_token_to_index.iter() {
            assert!(
                index < &self.hyp_token_to_index.len(),
                "index is {}, but there are only {} hyp tokens",
                index,
                self.hyp_token_to_index.len()
            );
            index_to_hyp_token[*index] = hyp_token;
        }

        let mut data = HashMap::new();
        data.insert("tactics", index_to_tactic);
        data.insert("goal_tokens", index_to_goal_token);
        data.insert("hyp_tokens", index_to_hyp_token);

        let file = File::create(filename).unwrap();
        serde_json::to_writer(file, &data).unwrap();
    }

    pub fn load_from_text(filename: &str) -> TokenMap {
        let file = File::open(filename)
            .expect(&format!("Couldn't find features file at \"{}\"", filename));
        let json_data = serde_json::from_reader(file).expect("Couldn't parse json data");
        let (goal_tokens, tactics, hyp_tokens) = match json_data {
            serde_json::Value::Object(vals) => {
                match (vals["goal_tokens"].clone(),
                       vals["tactics"].clone(),
                       vals["hyp_tokens"].clone()) {
                    (
                        serde_json::Value::Array(gts),
                        serde_json::Value::Array(ts),
                        serde_json::Value::Array(hts),
                    ) => (
                        gts.iter().map(|gt| match gt {
                            serde_json::Value::String(s) => s.clone(),
                            _ => panic!("Invalid data"),
                        })
                        .collect::<Vec<_>>(),
                        ts.iter().map(|t| match t {
                            serde_json::Value::String(s) => s.clone(),
                            _ => panic!("Invalid data"),
                        })
                        .collect::<Vec<_>>(),
                        hts.iter().map(|ht| match ht {
                            serde_json::Value::String(s) => s.clone(),
                            _ => panic!("Invalid data"),
                        })
                        .collect::<Vec<_>>(),
                    ),
                    _ => panic!("Invalid data"),
                }
            }
            _ => panic!("Json data is not an object!"),
        };
        TokenMap {
            tactic_to_index: flip_vec(tactics),
            goal_token_to_index: flip_vec(goal_tokens),
            hyp_token_to_index: flip_vec(hyp_tokens),
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
    hyps: &Vec<String>,
    goal: &String,
) -> Vec<f64> {
    hyps.into_iter()
        .map(|hyp| {
            gestalt_ratio(goal, &get_hyp_type(hyp).chars().take(128).collect::<String>())
        })
        .collect()
}

fn best_scored_hyp<'a>(
    hyps: &'a Vec<String>,
    goal: &String,
) -> (&'a str, f64) {
    let mut best_hyp = "";
    let mut best_score = 1.0;
    for hyp in hyps.iter() {
        let score = gestalt_ratio(goal, get_hyp_type(hyp));
        if score < best_score {
            best_score = score;
            best_hyp = &hyp;
        }
    }
    (best_hyp, best_score)
}
