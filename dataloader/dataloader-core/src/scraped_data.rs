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

use crate::tokenizer::{get_symbols, get_words};
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
    pub fn with_focused_obl(&self, fg_obl: Obligation) -> Self {
	let mut fg_goals = vec![fg_obl];
	fg_goals.extend(self.context.fg_goals.iter().skip(1).cloned());
	ScrapedTactic {
	    relevant_lemmas: self.relevant_lemmas.clone(),
	    prev_tactics: self.prev_tactics.clone(),
	    context: ProofContext {
		fg_goals,
		bg_goals: self.context.bg_goals.clone(),
		shelved_goals: self.context.shelved_goals.clone(),
		given_up_goals: self.context.given_up_goals.clone(),
	    },
	    tactic: self.tactic.clone()
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
            ScrapedData::Tactic(serde_json::from_str(&actual_line)
                                .expect(&format!("Couldn't parse line {}", actual_line)))
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

pub fn preprocess_datum(datum: ScrapedTactic) -> Vec<ScrapedTactic> {
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
	    // Figure out the numeric argument
            let argstr = if argstr.chars().last() == Some('.') {
                argstr.chars().take(argstr.len() - 1).collect()
            } else {
                argstr
            };
            let argstr_tokens: Vec<_> = argstr.split_whitespace().collect();
	    if argstr_tokens.len() != 1 {
		return vec![datum.with_tactic(newtac)];
	    }
	    let arg_num = match argstr_tokens[0].parse::<i64>() {
		Ok(var_idx) => var_idx,
		Err(_) => { return vec![datum.with_tactic(newtac)]; }
	    };

	    // Set up a vector to hold the interactions that this unfolds to.
	    let mut new_scrapeds = Vec::new();

	    let mut hyps_list = datum.context.focused_hyps().clone();
	    let mut curgoal = remove_paths_from_goal(&datum.context.focused_goal()).clone();
	    let mut induction_target_var = None;

	    for _ in 0..arg_num {
		// Get interactions from introing the named variables first
	        loop {
	            match get_named_intro_result(&curgoal) {
                        Some((new_hyp, new_goal)) => {
		            let new_hyp_var = new_hyp.split(":").next().unwrap().trim();
                            new_scrapeds.push(datum.with_focused_obl(Obligation{
                                hypotheses:hyps_list.clone(),
                                goal: curgoal})
                                              .with_tactic(
                                                  format!("intro {}.", new_hyp_var)));
                            hyps_list.push(new_hyp);
                            curgoal = new_goal;
                        },
                        None => break,
                    }
                }

                // Get interactions introing unnamed variable
                let fresh_hyp_name = {
                    let existing_hyp_names: Vec<_> = hyps_list.iter().map(
                        |hyp| hyp.split(":").next().unwrap().trim()).collect();
                    let alts = ["H", "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"];
                    let mut optional_result = None;
                    for alt in alts.into_iter() {
                        if !existing_hyp_names.contains(&alt) {
                            optional_result = Some(alt);
                            break;
                        }
                    }
                    optional_result.expect(
                        &format!("Couldn't find fresh hyp name from {:?}, \
                                  existing hyps were {:?}",
                                 alts, hyps_list))
                };
                new_scrapeds.push(datum.with_focused_obl(Obligation{
                    hypotheses: hyps_list.clone(),
                    goal: curgoal.clone()}).with_tactic(format!("intro {}.", fresh_hyp_name)));
                let unnamed_intro_result = get_unnamed_intro_result(
                    &curgoal, fresh_hyp_name);
                let (new_hyp, new_goal) = unnamed_intro_result.expect(
                    "Couldn't intro unnamed hypothesis");
                hyps_list.push(new_hyp);
                curgoal = new_goal;
                induction_target_var = Some(fresh_hyp_name);
            }

            new_scrapeds.push(datum.with_focused_obl(Obligation{
                hypotheses: hyps_list.clone(),
                goal: curgoal}).with_tactic(
                format!("induction {}.", induction_target_var.unwrap())));

            return new_scrapeds;
        }
    }

    vec![datum.with_tactic(newtac)]
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

/// A function for doing some quick & dirty parsing of forall binders,
/// to statically produce the context after an "intro" call, but fail
/// if that binder was unnamed. Ported from similar logic in
/// coq_serapy.
fn get_named_intro_result(goal: &str) -> Option<(String, String)> {
    match get_intro_result(goal) {
        Some((result_hyp, result_goal)) => {
            let hyp_parts: Vec<_> = result_hyp.split(":").collect();
            if hyp_parts[0].trim() == "_" {
                return None;
            }
            Some((result_hyp, result_goal))
        }
        None => None
    }
}
/// A function for doing some quick & dirty parsing of forall binders,
/// to statically produce the context after an "intro" call, but fail
/// if that binder was named. Ported from similar logic in coq_serapy.
fn get_unnamed_intro_result(goal: &str, fresh_hyp_name: &str) -> Option<(String, String)> {
    match get_intro_result(goal) {
        Some((result_hyp, result_goal)) => {
            let hyp_parts: Vec<_> = result_hyp.split(":").collect();
            if hyp_parts[0].trim() != "_" {
                return None;
            }
            Some((format!("{} :{}", fresh_hyp_name, hyp_parts[1]), result_goal))
        }
        None => None
    }
}

/// A function for doing some quick & dirty parsing of forall binders,
/// to statically produce the context after an "intro" call. Ported
/// from similar logic in coq_serapy.
fn get_intro_result(goal: &str) -> Option<(String, String)> {
    let mut paren_depth = 0;
    let mut got_binder = false;
    let goal_symbols = get_words(goal);
    if goal_symbols[0] != "forall" {
        return None
    }
    let mut new_hyp_symbols = vec![];
    let mut new_goal_symbols: Vec<&str> = vec![];
    for w in goal_symbols[1..].iter() {
        if got_binder {
            // This handles the case where there's only one
            // parenthesized binder left, and we don't want to leave a
            // "forall ," there, but our logic so far has dictated
            // that after popping a parenthesized argument we should
            // leave the rest of it's forall.
	    if *w == "," && new_goal_symbols.len() == 1 {
		new_goal_symbols = vec![];
	    } else {
		new_goal_symbols.push(w);
	    }
	} else if *w == "(" {
            paren_depth += 1;
	} else if *w == ")" {
	    paren_depth -= 1;
	    if paren_depth == 0 {
		got_binder = true;
		new_goal_symbols.push(&goal_symbols[0]);
                // If we're dealing with multiple variables in a single type binding,
                // like `forall ( a a' : expr) ...`, then we need to leave one and
                // take the other.
                if new_hyp_symbols[1] != ":" {
                    new_goal_symbols.push("(");
                    new_goal_symbols.extend(new_hyp_symbols.iter().skip(1));
                    new_goal_symbols.push(")");
                    while new_hyp_symbols[1] != ":" {
                        new_hyp_symbols.remove(1);
                    }
                }
	    }
	} else if *w == "," && paren_depth == 0 {
	    got_binder = true;
	} else {
	    new_hyp_symbols.push(*w)
	}
    }
    assert_eq!(new_hyp_symbols[1], ":", "Can't figure out how to parse goal {}", goal);
    Some((new_hyp_symbols.join(" "), new_goal_symbols.join(" ")))
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
    pub paths_file: String,
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

fn remove_paths_from_goal(goal: &str) -> String {
    let mut updated_goal: String = "".to_string();
    let words: Vec<&str> = goal.split(" ").collect();
    for word in words{
        let split: Vec<&str> = word.split("|-path-|").collect();
        updated_goal = updated_goal + " " + split[0];
        }
    updated_goal = updated_goal.trim().to_string();
    updated_goal
    }
