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

use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use std::iter;

use crate::paren_util::*;
use regex::Regex;

pub type FloatUnpaddedTensor3D = Vec<Vec<Vec<f64>>>;
pub type LongUnpaddedTensor3D = Vec<Vec<Vec<i64>>>;

pub type LongTensor2D = Vec<Vec<i64>>;
// pub type LongUnpaddedTensor2D = Vec<Vec<i64>>;
pub type FloatTensor2D = Vec<Vec<f64>>;
// pub type FloatUnpaddedTensor2D = Vec<Vec<f64>>;

pub type LongTensor1D = Vec<i64>;
pub type FloatTensor1D = Vec<f64>;

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScrapedTactic {
    #[pyo3(get, set)]
    pub relevant_lemmas: Vec<String>,
    #[pyo3(get, set)]
    pub prev_tactics: Vec<String>,
    #[pyo3(get, set)]
    pub context: ProofContext,
    #[pyo3(get, set)]
    pub tactic: String,
}
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TacticContext {
    #[pyo3(get, set)]
    pub relevant_lemmas: Vec<String>,
    #[pyo3(get, set)]
    pub prev_tactics: Vec<String>,
    #[pyo3(get, set)]
    pub obligation: Obligation,
}

impl<'source> FromPyObject<'source> for TacticContext {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obj: PyObject = ob.to_object(py);
        let relevant_lemmas: Vec<String> = obj.getattr(py, "relevant_lemmas")?.extract(py)?;
        let prev_tactics: Vec<String> = obj.getattr(py, "prev_tactics")?.extract(py)?;
        let obligation: Obligation = obj.getattr(py, "obligation")?.extract(py)?;
        Ok(TacticContext {
            relevant_lemmas,
            prev_tactics,
            obligation,
        })
    }
}
#[pymethods]
impl TacticContext {
    #[new]
    fn new(obj: &PyRawObject) {
        let d: TacticContext = Default::default();
        obj.init({ d });
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Obligation {
    #[pyo3(get, set)]
    pub hypotheses: Vec<String>,
    #[pyo3(get, set)]
    pub goal: String,
}
#[pymethods]
impl Obligation {
    #[new]
    fn new(obj: &PyRawObject) {
        let d: Obligation = Default::default();
        obj.init({ d });
    }
}

impl<'source> FromPyObject<'source> for Obligation {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obj: PyObject = ob.to_object(py);
        let hypotheses: Vec<String> = obj.getattr(py, "hypotheses")?.extract(py)?;
        let goal: String = obj.getattr(py, "goal")?.extract(py)?;
        Ok(Obligation { hypotheses, goal })
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofContext {
    #[pyo3(get, set)]
    pub fg_goals: Vec<Obligation>,
    #[pyo3(get, set)]
    pub bg_goals: Vec<Obligation>,
    #[pyo3(get, set)]
    pub shelved_goals: Vec<Obligation>,
    #[pyo3(get, set)]
    pub given_up_goals: Vec<Obligation>,
}

impl ProofContext {
    pub fn empty() -> Self {
        ProofContext {
            fg_goals: vec![],
            bg_goals: vec![],
            shelved_goals: vec![],
            given_up_goals: vec![],
        }
    }
    pub fn focused_goal(&self) -> &String {
        lazy_static! {
            static ref EMPTY_STR: String = "".to_string();
        }
        match self.fg_goals.first() {
            Some(obl) => &obl.goal,
            None => &EMPTY_STR,
        }
    }
    pub fn focused_hyps(&self) -> &Vec<String> {
        lazy_static! {
            static ref EMPTY_VEC: Vec<String> = vec![];
        }
        match self.fg_goals.first() {
            Some(obl) => &obl.hypotheses,
            None => &EMPTY_VEC,
        }
    }
}

impl<'source> FromPyObject<'source> for ProofContext {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obj: PyObject = ob.to_object(py);
        let fg_goals: Vec<Obligation> = obj.getattr(py, "fg_goals")?.extract(py)?;
        let bg_goals: Vec<Obligation> = obj.getattr(py, "bg_goals")?.extract(py)?;
        let shelved_goals: Vec<Obligation> = obj.getattr(py, "shelved_goals")?.extract(py)?;
        let given_up_goals: Vec<Obligation> = obj.getattr(py, "given_up_goals")?.extract(py)?;
        Ok(ProofContext {
            fg_goals,
            bg_goals,
            shelved_goals,
            given_up_goals,
        })
    }
}
#[pyclass]
#[derive(Clone)]
pub struct ScrapedTransition {
    #[pyo3(get, set)]
    pub relevant_lemmas: Vec<String>,
    #[pyo3(get, set)]
    pub prev_tactics: Vec<String>,
    #[pyo3(get, set)]
    pub before: ProofContext,
    #[pyo3(get, set)]
    pub after: ProofContext,
    #[pyo3(get, set)]
    pub tactic: String,
}

pub struct VernacCommand {
    pub command: String,
}

pub enum ScrapedData {
    Vernac(VernacCommand),
    Tactic(ScrapedTactic),
}

pub fn scraped_from_file(file: File) -> impl iter::Iterator<Item = ScrapedData> {
    BufReader::new(file).lines().map(|line: Result<String>| {
        let actual_line = line.expect("Couldn't read line");
        if actual_line.starts_with("\"") {
            ScrapedData::Vernac(VernacCommand {
                command: serde_json::from_str(&actual_line).expect("Couldn't parse string"),
            })
        } else {
            ScrapedData::Tactic(serde_json::from_str(&actual_line).expect("Couldn't parse line"))
        }
    })
}

pub fn kill_comments(source: &str) -> String {
    let mut result = String::new();
    let mut depth = 0;
    let mut cur_pos = 0;
    macro_rules! lookup {
        ($pat:expr) => {
            source[cur_pos..]
                .find($pat)
                .map(|pos| pos + cur_pos)
                .unwrap_or(source.len())
        };
    };
    while cur_pos < source.len() {
        let next_open = lookup!("(*");
        let next_close = lookup!("*)");
        if depth == 0 {
            assert!(
                next_open <= next_close,
                "Unbalanced comment delimiters! Too many closes"
            );
            result.push_str(&source[cur_pos..next_open]);
            cur_pos = next_open + 2;
            depth += 1;
        } else if next_open < next_close {
            depth += 1;
            cur_pos = next_open + 2;
        } else {
            assert!(
                next_close < next_open,
                "Unbalanced comment delimiters! Not enough closes"
            );
            assert!(depth > 0);
            depth -= 1;
            cur_pos = next_close + 2;
        }
    }
    result
}

pub fn split_tactic(full_tactic: &str) -> Option<(String, String)> {
    let no_comments_tac = kill_comments(full_tactic);
    let prepped_tac = no_comments_tac.trim();
    lazy_static! {
        static ref GOAL_SELECTOR: Regex = Regex::new(r"^\s*[-+*{}]+\s*$").unwrap();
    }
    if GOAL_SELECTOR.is_match(prepped_tac)
        || split_to_next_pat_outside_parens(&prepped_tac, ";").is_some()
    {
        return None;
    }
    for prefix in &["try", "now", "repeat", "decide"] {
        if prepped_tac.starts_with(prefix) {
            return split_tactic(&prepped_tac[prefix.len()..]).map(|(rest_stem, rest_rest)| {
                let mut new_stem = prefix.to_string();
                new_stem.push_str(" ");
                new_stem.push_str(&rest_stem);
                (new_stem, rest_rest)
            });
        }
    }
    for special_stem in &["rewrite <-", "rewrite !", "intros until", "simpl in"] {
        if prepped_tac.starts_with(special_stem) {
            return Some((
                special_stem.to_string(),
                prepped_tac[special_stem.len()..].to_string(),
            ));
        }
    }
    prepped_tac
        .find(|c| c == '.' || char::is_whitespace(c))
        .map(|idx| {
            (
                prepped_tac[..idx].to_string(),
                prepped_tac[idx..].to_string(),
            )
        })
}

pub fn get_stem(full_tactic: &str) -> Option<String> {
    split_tactic(full_tactic).map(|(stem, _args)| stem)
}

#[pyclass]
#[derive(Default, Clone)]
pub struct DataloaderArgs {
    #[pyo3(get, set)]
    pub max_distance: usize,
    #[pyo3(get, set)]
    pub max_string_distance: usize,
    #[pyo3(get, set)]
    pub max_length: usize,
    #[pyo3(get, set)]
    pub max_premises: usize,
    #[pyo3(get, set)]
    pub num_keywords: usize,
    #[pyo3(get, set)]
    pub num_relevance_samples: usize,
    #[pyo3(get, set)]
    pub keywords_file: String,
    #[pyo3(get, set)]
    pub context_filter: String,
    #[pyo3(get, set)]
    pub save_embedding: Option<String>,
    #[pyo3(get, set)]
    pub save_features_state: Option<String>,
}
#[pymethods]
impl DataloaderArgs {
    #[new]
    fn new(obj: &PyRawObject) {
        let d: DataloaderArgs = Default::default();
        obj.init({ d });
    }
}
impl<'source> pyo3::FromPyObject<'source> for DataloaderArgs {
    fn extract(ob: &'source pyo3::types::PyAny) -> pyo3::PyResult<DataloaderArgs> {
        let cls: &DataloaderArgs = pyo3::PyTryFrom::try_from(ob)?;
        Ok(cls.clone())
    }
}

pub struct NormalFloat(f64);
impl NormalFloat {
    pub fn new(v: f64) -> NormalFloat {
        assert_eq!(v, v);
        NormalFloat(v)
    }
}
impl PartialEq for NormalFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for NormalFloat {}
impl PartialOrd for NormalFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.0 > other.0 {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Less)
        }
    }
}
impl Ord for NormalFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
