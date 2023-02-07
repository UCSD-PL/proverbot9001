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


extern crate regex;
use rayon::prelude::*;
use regex::Regex;

use crate::scraped_data::*;

use crate::context_filter_ast::ContextFilterAST;

use crate::tokenizer::{get_words, remove_paths_from_goal};
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(context_filter_parser);

pub fn filter_data_by_key<A: Send>(
    max_term_length: usize,
    filter_spec: &str,
    data: Vec<A>,
    key: fn(&A) -> &ScrapedTactic,
) -> Vec<A> {
    let parsed_filter = context_filter_parser::ToplevelFilterParser::new()
        .parse(filter_spec)
        .expect(&format!("Invalid context filter: {}", filter_spec));
    data.into_par_iter()
        .filter(|datum| apply_filter(max_term_length, &parsed_filter, key(datum)))
        .collect()
}

pub fn _filter_data(
    max_term_length: usize,
    filter_spec: &str,
    data: Vec<ScrapedTactic>,
) -> Vec<ScrapedTactic> {
    let parsed_filter = context_filter_parser::ToplevelFilterParser::new()
        .parse(filter_spec)
        .expect(&format!("Invalid context filter: {}", filter_spec));
    data.into_par_iter()
        .filter(|datum| apply_filter(max_term_length, &parsed_filter, datum))
        .collect()
}

pub fn parse_filter(filter_spec: &str) -> ContextFilterAST {
    context_filter_parser::ToplevelFilterParser::new()
        .parse(filter_spec)
        .expect(&format!("Invalid context filter: {}", filter_spec))
}

pub fn apply_filter(
    max_term_length: usize,
    parsed_filter: &ContextFilterAST,
    scraped: &ScrapedTactic,
) -> bool {
    match parsed_filter {
        ContextFilterAST::And(subfilters) => subfilters
            .iter()
            .all(|subfilter| apply_filter(max_term_length, subfilter, scraped)),
        ContextFilterAST::Or(subfilters) => subfilters
            .iter()
            .any(|subfilter| apply_filter(max_term_length, subfilter, scraped)),
        ContextFilterAST::None => false,
        ContextFilterAST::All => true,
        ContextFilterAST::GoalArgs => {
            let updated_goal = &remove_paths_from_goal(scraped.context.focused_goal());
            let goal_symbols: Vec<&str> = get_words(updated_goal)
                .into_iter()
                .take(max_term_length)
                .collect();
            let (tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let mut trimmed_args = tactic_argstr.trim();
            if trimmed_args.chars().last() == Some('.') {
                trimmed_args = &trimmed_args[..trimmed_args.len() - 1]
            }
            let arg_tokens: Vec<&str> = trimmed_args.split_whitespace().collect();
            // While the arguments to an intro(s) might *look* like
            // goal arguments, they are actually fresh variables
            if (tactic_stem == "intros" || tactic_stem == "intro") && arg_tokens.len() > 0 {
                return false;
            }
            arg_tokens.into_iter().all(|arg_token| {
                goal_symbols
                    .iter()
                    .find(|symbol| symbol_matches(*symbol, arg_token))
                    .is_some()
            })
        }
        ContextFilterAST::HypArgs => {
            let hyp_names: Vec<String> =
                indexed_premises(scraped.context.focused_hyps().iter().map(|s| s.as_ref()))
                    .into_iter()
                    .map(|(_idx, hyp_name)| hyp_name)
                    .collect();
            let (tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let mut trimmed_args = tactic_argstr.trim();
            if trimmed_args.chars().last() == Some('.') {
                trimmed_args = &trimmed_args[..trimmed_args.len() - 1]
            }
            let arg_tokens: Vec<&str> = trimmed_args.split_whitespace().collect();
            // While the arguments to an intro(s) might *look* like
            // hyp arguments, they are actually fresh variables
            if (tactic_stem == "intros" || tactic_stem == "intro") && arg_tokens.len() > 0 {
                return false;
            }
            let result = (tactic_takes_hyp_args(&tactic_stem)
                && arg_tokens
                    .iter()
                    .cloned() //.into_iter()
                    .all(|arg_token| hyp_names.contains(&arg_token.to_string())))
                || arg_tokens.len() == 0;
            result
        }
        ContextFilterAST::RelevantLemmaArgs => {
            let lemma_names: Vec<_> =
                indexed_premises(scraped.relevant_lemmas.iter().map(|s| s.as_ref()))
                    .into_iter()
                    .map(|(_idx, hyp_name)| hyp_name)
                    .collect();
            let (tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let mut trimmed_args = tactic_argstr.trim();
            if trimmed_args.chars().last() == Some('.') {
                trimmed_args = &trimmed_args[..trimmed_args.len() - 1]
            }
            let arg_tokens: Vec<&str> = trimmed_args.split_whitespace().collect();
            let result = tactic_takes_hyp_args(&tactic_stem)
                && arg_tokens
                    .iter()
                    .all(|arg_token| lemma_names.contains(&arg_token.to_string()))
                || arg_tokens.len() == 0;
            // assert!(!(scraped.tactic.trim() == "induction e." && result));
            // assert!(result, "{}", &scraped.tactic);
            result
        }
        ContextFilterAST::NumericArgs => {
            let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let trimmed_args = tactic_argstr.trim();
            trimmed_args[..trimmed_args.len() - 1]
                .split_whitespace()
                .all(|arg_token| arg_token.chars().all(char::is_numeric))
        }
        ContextFilterAST::NoSemis => !scraped.tactic.contains(";"),
        ContextFilterAST::Tactic(s) => {
            let (tactic_stem, _tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            tactic_stem == *s
        }
        ContextFilterAST::MaxArgs(num) => {
            let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let trimmed_args = tactic_argstr.trim();
            trimmed_args[..trimmed_args.len() - 1]
                .split_whitespace()
                .collect::<Vec<_>>()
                .len() as i64
                <= *num
        }
        ContextFilterAST::Default => {
            let tactic = kill_comments(&scraped.tactic);
            lazy_static! {
                static ref BACKGROUND_TAC: Regex = Regex::new(r"^\d+:.*").unwrap();
                static ref BULLET: Regex = Regex::new(r"^\s*[\{\}\+\-\*].*").unwrap();
            }
            let result = !tactic.contains(";")
                && !tactic.contains("Proof")
                && !tactic.contains("Opaque")
                && !tactic.contains("Qed")
                && !tactic.contains("Defined")
                && !tactic.contains("Unshelve")
                && !BACKGROUND_TAC.is_match(&tactic)
                && !BULLET.is_match(&tactic);
            result
        }
    }
}
