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

use crate::tokenizer::get_symbols;
use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result, Write};
use std::iter;

use crate::paren_util::*;
use regex::Regex;

pub type FloatUnpaddedTensor3D = Vec<Vec<Vec<f64>>>;
pub type LongUnpaddedTensor3D = Vec<Vec<Vec<i64>>>;

pub type LongTensor2D = Vec<Vec<i64>>;
// pub type LongUnpaddedTensor2D = Vec<Vec<i64>>;
pub type FloatTensor2D = Vec<Vec<f64>>;
// pub type FloatUnpaddedTensor2D = Vec<Vec<f64>>;

pub type BoolTensor2D = Vec<Vec<bool>>;

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

impl ScrapedTactic {
    pub fn with_tactic(&self, tactic: String) -> Self {
        ScrapedTactic {
            relevant_lemmas: self.relevant_lemmas.clone(),
            prev_tactics: self.prev_tactics.clone(),
            context: self.context.clone(),
            tactic: tactic,
        }
    }
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

#[pymethods]
impl TacticContext {
    #[new]
    fn new(
        relevant_lemmas: Vec<String>,
        prev_tactics: Vec<String>,
        obligation: Obligation,
    ) -> Self {
        TacticContext {
            relevant_lemmas,
            prev_tactics,
            obligation,
        }
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
    fn new(hypotheses: Vec<String>, goal: String) -> Self {
        Obligation { hypotheses, goal }
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

impl ScrapedTransition {
    pub fn scraped_before(&self) -> ScrapedTactic {
        ScrapedTactic {
            relevant_lemmas: self.relevant_lemmas.clone(),
            prev_tactics: self.prev_tactics.clone(),
            context: self.before.clone(),
            tactic: self.tactic.clone(),
        }
    }
}

#[derive(Clone)]
pub struct VernacCommand {
    pub command: String,
}

#[derive(Clone)]
pub enum ScrapedData {
    Vernac(VernacCommand),
    Tactic(ScrapedTactic),
}

pub struct AdjacentPairs<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    pk: iter::Peekable<I>,
}
impl<I, T> AdjacentPairs<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    fn new(it: I) -> Self {
        AdjacentPairs { pk: it.peekable() }
    }
}
impl<I: Iterator<Item = T>, T: Clone> Iterator for AdjacentPairs<I, T> {
    type Item = (I::Item, Option<I::Item>);
    fn next(&mut self) -> Option<(I::Item, Option<I::Item>)> {
        match self.pk.next() {
            Some(v) => Some((v, self.pk.peek().cloned())),
            None => None,
        }
    }
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

pub fn scraped_to_file(mut file: File, scraped: impl iter::Iterator<Item = ScrapedData>) {
    for point in scraped {
        match point {
            ScrapedData::Vernac(VernacCommand { command: cmd }) => {
                writeln!(&mut file, "\"{}\"", cmd).unwrap()
            }
            ScrapedData::Tactic(tac) => {
                serde_json::to_writer(
                    &mut file,
                    &serde_json::json!({
                    "prev_tactics": tac.prev_tactics,
                    "prev_hyps": tac.context.focused_hyps(),
                    "prev_goal": tac.context.focused_goal(),
                    "relevant_lemmas": tac.relevant_lemmas,
                    "tactic": tac.tactic}),
                )
                .unwrap();
                writeln!(&mut file, "").unwrap();
            }
        }
    }
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
    }
    while cur_pos < source.len() {
        let next_open = lookup!("(*");
        let next_close = lookup!("*)");
        if depth == 0 {
            // assert!(
            //     next_open <= next_close,
            //     "Unbalanced comment delimiters! Too many closes"
            // );
            if next_open <= next_close {
                result.push_str(&source[cur_pos..next_open]);
                cur_pos = next_open + 2;
                depth += 1;
            } else {
                cur_pos = next_close + 2;
            }
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
    if GOAL_SELECTOR.is_match(prepped_tac) || prepped_tac.contains(';') {
        return None;
    }
    for prefix in &["try", "now", "repeat", "decide"] {
        if prepped_tac.starts_with(prefix) {
            return split_tactic(&prepped_tac[prefix.len()..].trim()).map(
                |(rest_stem, rest_rest)| {
                    let mut new_stem = prefix.to_string();
                    new_stem.push_str(" ");
                    new_stem.push_str(&rest_stem);
                    (new_stem, rest_rest)
                },
            );
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
        .find(|c| !(char::is_alphabetic(c) || c == '\'' || c == '_' || c == '('))
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

pub fn preprocess_datum(datum: ScrapedTactic) -> ScrapedTactic {
    let tacstr = kill_comments(&datum.tactic);
    let mut newtac = tacstr.trim().to_string();
    // Truncate semicolons
    if newtac.chars().next() == Some('(') && newtac.chars().last() == Some(')') {
        newtac = newtac[1..newtac.len() - 1].to_string();
    }
    if let Some((before_semi, _)) = split_to_next_pat_outside_parens(&newtac, ";") {
        newtac = before_semi.trim().to_string();
        newtac.push('.');
    }

    // Substitutions
    lazy_static! {
        static ref SUBS_MAP: HashMap<&'static str, &'static str> = {
            let mut res = HashMap::new();
            res.insert("auto", "eauto.");
            res.insert("intros until", "intros.");
            res.insert("intro", "intros.");
            res.insert("constructor", "econstructor.");
            res
        };
    }
    if let Some(stem) = get_stem(&newtac) {
        let borrowed_stem: &str = &stem;
        if let Some(subbed_stem) = SUBS_MAP.get(borrowed_stem) {
            newtac = subbed_stem.to_string();
        }
    }

    // Numeric args
    if let Some((stem, argstr)) = split_tactic(&newtac) {
        if stem == "induction" || stem == "destruct" {
            // println!("Preprocessing {}", datum.tactic);
            let argstr = if argstr.chars().last() == Some('.') {
                argstr.chars().take(argstr.len() - 1).collect()
            } else {
                argstr
            };
            let argstr_tokens: Vec<_> = argstr.split_whitespace().collect();
            if argstr_tokens.len() == 1 {
                let new_argstr = argstr_tokens
                    .into_iter()
                    .map(|token| match token.parse::<i64>() {
                        Ok(var_idx) => {
                            match get_binder_var(datum.context.focused_goal(), var_idx) {
                                Some(var) => var,
                                None => token,
                            }
                        }
                        Err(_) => token,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                newtac = vec![stem, new_argstr].join(" ");
                newtac.push('.');
            }
        }
    }

    datum.with_tactic(newtac)
}

pub fn tactic_takes_hyp_args(tactic_stem: &str) -> bool {
    let tactic_stem = tactic_stem.trim();
    if tactic_stem.starts_with("now") || tactic_stem.starts_with("try") {
        tactic_takes_hyp_args(&tactic_stem[4..])
    } else if tactic_stem.starts_with("repeat") {
        tactic_takes_hyp_args(&tactic_stem[7..])
    } else {
        tactic_stem == "apply"
            || tactic_stem == "eapply"
            || tactic_stem == "eexploit"
            || tactic_stem == "exploit"
            || tactic_stem == "erewrite"
            || tactic_stem == "rewrite"
            || tactic_stem == "erewrite !"
            || tactic_stem == "rewrite !"
            || tactic_stem == "erewrite <-"
            || tactic_stem == "rewrite <-"
            || tactic_stem == "destruct"
            || tactic_stem == "elim"
            || tactic_stem == "eelim"
            || tactic_stem == "inversion"
            || tactic_stem == "monadInv"
            || tactic_stem == "pattern"
            || tactic_stem == "revert"
            || tactic_stem == "exact"
            || tactic_stem == "eexact"
            || tactic_stem == "simpl in"
            || tactic_stem == "fold"
            || tactic_stem == "generalize"
            || tactic_stem == "exists"
            || tactic_stem == "case"
            || tactic_stem == "inv"
            || tactic_stem == "subst"
            || tactic_stem == "specialize"
    }
}

pub fn indexed_premises<'a>(premises: impl Iterator<Item = &'a str>) -> Vec<(usize, String)> {
    let mut result = Vec::new();
    for (idx, premise) in premises.enumerate() {
        let before_colon = premise
            .split(":")
            .next()
            .expect("Hyp string doesn't have a colon in it")
            .trim();
        let vars = before_colon
            .split(",")
            .map(|varname| (idx, varname.trim().to_string()));
        result.extend(vars);
    }
    result
}

/// A function for doing some quick & dirty parsing of forall
/// binders. Ported from the python implementation of this same
/// function in serapi_instance.py.
fn get_binder_var(goal: &str, binder_idx: i64) -> Option<&str> {
    let mut paren_depth = 0;
    let mut binders_passed = 0;
    let mut skip = false;
    lazy_static! {
        static ref FORALL: Regex = Regex::new(r"forall\s+").unwrap();
    }
    let forall_match = match FORALL.find(goal) {
        Some(m) => m,
        None => return None,
    };
    let rest_goal = &goal[forall_match.end()..];
    for w in get_symbols(rest_goal) {
        if w == "(" {
            paren_depth += 1;
        } else if w == ")" {
            paren_depth -= 1;
            if paren_depth < 2 {
                skip = false;
            }
        } else if paren_depth < 2 && !skip {
            if w == ":" {
                skip = true;
            } else {
                binders_passed += 1;
                if binders_passed == binder_idx {
                    return Some(w);
                }
            }
        }
    }
    panic!("Not enough binders!")
}

pub fn get_hyp_type(hyp: &str) -> &str {
    lazy_static! {
        static ref TYPECOLON: Regex = Regex::new(r":[^=]").unwrap();
    }
    TYPECOLON.splitn(hyp, 2).last().unwrap_or("")
}

pub fn symbol_matches(full_symbol: &str, shorthand_symbol: &str) -> bool {
    if full_symbol == shorthand_symbol {
        true
    } else {
        full_symbol.split(".").last().unwrap() == shorthand_symbol
    }
}

#[pyclass]
#[derive(Default, Clone)]
pub struct DataloaderArgs {
    #[pyo3(get, set)]
    pub max_tuples: Option<usize>,
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
    #[pyo3(get, set)]
    pub load_embedding: Option<String>,
    #[pyo3(get, set)]
    pub load_features_state: Option<String>,
}
#[pymethods]
impl DataloaderArgs {
    #[new]
    fn new() -> Self {
        let d: DataloaderArgs = Default::default();
        d
    }
}
