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

use crate::scraped_data::*;

use crate::context_filter_ast::ContextFilterAST;

extern crate regex;
use regex::Regex;
use rayon::prelude::*;

use lalrpop_util::lalrpop_mod;

#[allow(dead_code)]
lalrpop_mod!(context_filter_parser);

pub fn filter_data_by_key<A: Send>(
    args: &DataloaderArgs,
    filter_spec: &str,
    data: Vec<A>,
    key: fn(&A) -> &ScrapedTactic,
) -> Vec<A> {
    let parsed_filter = context_filter_parser::ToplevelFilterParser::new()
        .parse(filter_spec)
        .expect(&format!("Invalid context filter: {}", filter_spec));
    data.into_par_iter()
        .filter(|datum| apply_filter(args, &parsed_filter, key(datum)))
        .collect()
}

pub fn filter_data(
    args: &DataloaderArgs,
    filter_spec: &str,
    data: Vec<ScrapedTactic>,
) -> Vec<ScrapedTactic> {
    let parsed_filter = context_filter_parser::ToplevelFilterParser::new()
        .parse(filter_spec)
        .expect(&format!("Invalid context filter: {}", filter_spec));
    data.into_par_iter()
        .filter(|datum| apply_filter(args, &parsed_filter, datum))
        .collect()
}

fn apply_filter(
    args: &DataloaderArgs,
    parsed_filter: &ContextFilterAST,
    scraped: &ScrapedTactic,
) -> bool {
    match parsed_filter {
        ContextFilterAST::And(subfilters) => subfilters
            .iter()
            .all(|subfilter| apply_filter(args, subfilter, scraped)),
        ContextFilterAST::Or(subfilters) => subfilters
            .iter()
            .any(|subfilter| apply_filter(args, subfilter, scraped)),
        ContextFilterAST::None => false,
        ContextFilterAST::All => true,
        ContextFilterAST::GoalArgs => {
            let goal_symbols: Vec<&str> = scraped
                .prev_goal
                .split_whitespace()
                .take(args.max_length)
                .collect();
            let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let trimmed_args = tactic_argstr.trim();
            trimmed_args[..trimmed_args.len() - 1]
                .split_whitespace()
                .all(|arg_token| goal_symbols.contains(&arg_token))
        }
        ContextFilterAST::HypArgs => {
            let hyp_names: Vec<&str> = scraped
                .prev_hyps
                .iter()
                .map(|hyp| hyp.splitn(1, ":").next().unwrap().trim())
                .collect();
            let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let trimmed_args = tactic_argstr.trim();
            trimmed_args[..trimmed_args.len() - 1]
                .split_whitespace()
                .all(|arg_token| hyp_names.contains(&arg_token))
        }
        ContextFilterAST::RelevantLemmaArgs => {
            let lemma_names: Vec<_>= scraped
                .relevant_lemmas
                .iter()
                .map(|lemma| lemma.splitn(1, ":").next().unwrap().trim())
                .collect();
            let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
                None => return false,
                Some(x) => x,
            };
            let trimmed_args = tactic_argstr.trim();
            trimmed_args[..trimmed_args.len() - 1]
                .split_whitespace()
                .all(|arg_token| lemma_names.contains(&arg_token))
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
            lazy_static! {
                static ref BACKGROUND_TAC: Regex = Regex::new(r"\d+:.*").unwrap();
            }
            !scraped.tactic.contains(";") &&
                !scraped.tactic.contains("Proof") &&
                !scraped.tactic.contains("Opaque") &&
                !scraped.tactic.contains("Qed") &&
                !scraped.tactic.contains("Defined") &&
                !scraped.tactic.contains("Unshelve") &&
                !BACKGROUND_TAC.is_match(&scraped.tactic)
        }
    }
}
