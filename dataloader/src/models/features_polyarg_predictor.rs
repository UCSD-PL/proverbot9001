use pyo3::exceptions;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::features::PickleableTokenMap as PickleableFeaturesTokenMap;
use crate::features::TokenMap as FeaturesTokenMap;
use crate::features::*;
use crate::paren_util::split_to_next_matching_paren_or_space;
use crate::scraped_data::*;
use crate::tokenizer::{
    get_words, OpenIndexer, PickleableIndexer, PickleableTokenizer, Token, Tokenizer,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TacticArgument {
    HypVar(usize),
    GoalToken(usize),
    NoArg,
    Unrecognized,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FPAInput {
    hypothesis_types: Vec<Vec<Token>>,
    hypothesis_features: Vec<Vec<f64>>,
    goal: Vec<Token>,
    word_features: Vec<usize>,
    vec_features: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FPAOutput {
    tactic: usize,
    argument: TacticArgument,
}

pub type FPAMetadata = (OpenIndexer<String>, Tokenizer, FeaturesTokenMap);
pub type PickleableFPAMetadata = (
    PickleableIndexer<String>,
    PickleableTokenizer,
    PickleableFeaturesTokenMap,
);

pub fn fpa_metadata_to_pickleable(metadata: FPAMetadata) -> PickleableFPAMetadata {
    (
        metadata.0.to_pickleable(),
        metadata.1.to_pickleable(),
        metadata.2.to_dicts(),
    )
}

pub fn fpa_metadata_from_pickleable(pick: PickleableFPAMetadata) -> FPAMetadata {
    (
        OpenIndexer::from_pickleable(pick.0),
        Tokenizer::from_pickleable(pick.1),
        FeaturesTokenMap::from_dicts(pick.2),
    )
}

pub fn features_polyarg_tensors(
    args: DataloaderArgs,
    filename: String,
    metadata: Option<PickleableFPAMetadata>,
) -> PyResult<(
    PickleableFPAMetadata,
    (
        LongUnpaddedTensor3D,
        FloatUnpaddedTensor3D,
        LongTensor1D,
        LongTensor2D,
        LongTensor2D,
        FloatTensor2D,
        LongTensor1D,
        LongTensor1D,
    ),
    (Vec<i64>, i64),
)> {
    let mut raw_data: Vec<ScrapedTactic> = scraped_from_file(
        File::open(filename)
            .map_err(|_err| PyErr::new::<exceptions::TypeError, _>("Failed to open file"))?,
    )
    .flat_map(|data| match data {
        ScrapedData::Vernac(_) => None,
        ScrapedData::Tactic(t) => {
            if get_stem(&t.tactic).is_some() {
                Some(t)
            } else {
                None
            }
        }
    })
    .collect();
    let (mut indexer, rest_meta) = match metadata {
        Some((indexer, tokenizer, tmap)) => (
            OpenIndexer::from_pickleable(indexer),
            Some((tokenizer, tmap)),
        ),
        None => (OpenIndexer::new(), None),
    };
    let tactic_stem_indices: Vec<i64> = raw_data
        .iter()
        .map(|data| {
            indexer.lookup(
                get_stem(&data.tactic)
                    .expect(&format!("Couldn't get the stem for {}", data.tactic))
                    .to_string(),
            )
        })
        .collect();
    indexer.freeze();

    let (tokenizer, features_token_map) = match rest_meta {
        Some((ptok, ptmap)) => (
            Tokenizer::from_pickleable(ptok),
            FeaturesTokenMap::from_dicts(ptmap),
        ),
        None => {
            let use_unknowns = true;
            let num_reserved_tokens = 2;
            (
                Tokenizer::new(use_unknowns, num_reserved_tokens, &args.keywords_file),
                FeaturesTokenMap::initialize(&raw_data, args.num_keywords),
            )
        }
    };
    raw_data.sort_by_key(|pnt| -(pnt.prev_hyps.len() as i64));

    let hyp_scores: Vec<Vec<f64>> = raw_data
        .iter()
        .map(|scraped| {
            score_hyps(
                args.max_distance,
                args.max_length,
                &scraped.prev_hyps,
                &scraped.prev_goal,
            )
        })
        .collect();

    let num_hyps = raw_data
        .iter()
        .map(|tac| tac.prev_hyps.len() as i64)
        .collect();
    let (best_hyps, best_hyp_scores): (Vec<&str>, Vec<f64>) = raw_data
        .iter()
        .map(|tac| &tac.prev_hyps)
        .zip(hyp_scores.iter())
        .map(|(hyps, scores)| {
            hyps.iter()
                .map(|hyp| hyp.as_ref())
                .zip(scores.iter().map(|score| *score))
                .min_by_key(|(_hyp, score)| NormalFloat::new(*score))
                .unwrap_or(("", 1.0))
        })
        .unzip();
    let vec_features = best_hyp_scores
        .into_iter()
        .map(|score| vec![score])
        .collect();
    let word_features = raw_data
        .iter()
        .zip(best_hyps)
        .map(|(scraped, best_hyp)| {
            vec![
                prev_tactic_feature(&features_token_map, &scraped.prev_tactics),
                goal_head_feature(&features_token_map, &scraped.prev_goal),
                hyp_head_feature(&features_token_map, best_hyp),
            ]
        })
        .collect();
    let tokenized_goals = raw_data
        .iter()
        .map(|tac| {
            normalize_sentence_length(tokenizer.tokenize(&tac.prev_goal), args.max_length, 1)
        })
        .collect();
    let (arg_indices, selected_hyps): (Vec<i64>, Vec<Vec<&String>>) = raw_data
        .iter()
        .map(|scraped| {
            let (arg, selected) = get_argument(&args, scraped);
            (arg_to_index(&args, arg), selected)
        })
        .unzip();
    let tokenized_hyps = selected_hyps
        .iter()
        .map(|hyps| {
            hyps.iter()
                .map(|hyp| normalize_sentence_length(tokenizer.tokenize(hyp), args.max_length, 1))
                .collect()
        })
        .collect();
    let hyp_features = raw_data
        .iter()
        .zip(selected_hyps)
        .map(|(scraped, selected)| {
            score_hyps(
                args.max_distance,
                args.max_length,
                &selected.iter().map(|hyp| hyp.clone().clone()).collect(),
                &scraped.prev_goal,
            )
            .iter()
            .zip(selected)
            .map(|(score, hyp)| vec![*score, equality_hyp_feature(hyp, &scraped.prev_goal)])
            .collect()
        })
        .collect();
    let word_features_sizes = features_token_map.word_features_sizes();
    Ok((
        fpa_metadata_to_pickleable((indexer, tokenizer, features_token_map)),
        (
            tokenized_hyps,
            hyp_features,
            num_hyps,
            tokenized_goals,
            word_features,
            vec_features,
            tactic_stem_indices,
            arg_indices,
        ),
        (word_features_sizes, VEC_FEATURES_SIZE),
    ))
}

pub fn sample_fpa(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    hypotheses: Vec<String>,
    goal: String,
) -> (
    LongUnpaddedTensor3D,
    FloatUnpaddedTensor3D,
    LongTensor1D,
    LongTensor2D,
    LongTensor2D,
    FloatTensor2D,
) {
    let (_indexer, tokenizer, ftmap) = fpa_metadata_from_pickleable(metadata);
    let hyp_scores = score_hyps(args.max_distance, args.max_length, &hypotheses, &goal);
    let hyp_features = hypotheses
        .iter()
        .zip(hyp_scores.iter())
        .map(|(hyp, score)| vec![*score, equality_hyp_feature(hyp, &goal)])
        .collect();
    let num_hyps = hypotheses.len();
    let tokenized_goal = tokenizer.tokenize(&goal);
    let (best_hyp, best_score): (&str, f64) = hypotheses
        .iter()
        .zip(hyp_scores.iter())
        .map(|(h, s)| (h.as_str(), *s))
        .min_by_key(|(_hyp, score)| NormalFloat::new(*score))
        .unwrap_or(("", 1.0));
    let vec_features = vec![best_score];
    let word_features = vec![
        prev_tactic_feature(&ftmap, &prev_tactics),
        goal_head_feature(&ftmap, &goal),
        hyp_head_feature(&ftmap, best_hyp),
    ];
    let all_premises: Vec<String> = relevant_lemmas
        .into_iter()
        .chain(hypotheses.into_iter())
        .collect();
    let tokenized_premises = all_premises
        .into_iter()
        .map(|premise| tokenizer.tokenize(&premise))
        .collect();
    (
        vec![tokenized_premises],
        vec![hyp_features],
        vec![num_hyps as i64],
        vec![tokenized_goal],
        vec![word_features],
        vec![vec_features],
    )
}

pub fn decode_fpa_result(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    hyps: Vec<String>,
    goal: &str,
    tac_idx: i64,
    arg_idx: i64,
) -> String {
    let (indexer, _tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
    let stem = indexer.reverse_lookup(tac_idx);
    let argtype = if arg_idx == 0 {
        TacticArgument::NoArg
    } else if (arg_idx as usize) <= args.max_length {
        TacticArgument::GoalToken(arg_idx as usize - 1)
    } else {
        TacticArgument::HypVar((arg_idx as usize) - args.max_length)
    };
    match argtype {
        TacticArgument::NoArg => stem + ".",
        TacticArgument::Unrecognized => stem + ".",
        TacticArgument::GoalToken(tidx) => stem + get_words(goal)[tidx] + ".",
        TacticArgument::HypVar(hidx) => {
            stem + hyps[hidx].split(":").next().expect("No colon in hyp")
        }
    }
}

fn equality_hyp_feature(hyp: &str, goal: &str) -> f64 {
    lazy_static! {
        static ref EQ: Regex = Regex::new(r"eq\s+(.*)").expect("Can't build eq regex");
    }
    let equals_match = EQ.captures(hyp);
    if let Some(captures) = equals_match {
        let (left_side, right_side) = split_to_next_matching_paren_or_space(
            captures.get(1).expect("Can't get capture group").into(),
        );
        if goal.contains(left_side) {
            -1.0
        } else if goal.contains(right_side) {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    }
}

fn get_argument<'a>(
    args: &DataloaderArgs,
    scraped: &'a ScrapedTactic,
) -> (TacticArgument, Vec<&'a String>) {
    let all_hyps: Vec<&String> = scraped
        .prev_hyps
        .iter()
        .chain(scraped.relevant_lemmas.iter())
        .collect();
    macro_rules! rand_bounded_hyps {
        () => {
            if all_hyps.len() > args.max_premises {
                all_hyps
                    .choose_multiple(&mut thread_rng(), args.max_premises)
                    .map(|s| *s)
                    .collect()
            } else if all_hyps.len() == 0 {
                lazy_static! {
                    static ref COLONSTRING: String = ":".to_string();
                }
                vec![&COLONSTRING]
            } else {
                all_hyps
            }
        };
    }
    let (_tactic_stem, tactic_argstr) = match split_tactic(&scraped.tactic) {
        None => return (TacticArgument::Unrecognized, rand_bounded_hyps!()),
        Some(x) => x,
    };
    let argstr_tokens: Vec<&str> = tactic_argstr[..tactic_argstr.len() - 1]
        .split_whitespace()
        .collect();
    if argstr_tokens.len() == 0 {
        (TacticArgument::NoArg, rand_bounded_hyps!())
    } else if argstr_tokens.len() > 1 {
        (TacticArgument::Unrecognized, rand_bounded_hyps!())
    } else {
        let goal_symbols = get_words(&scraped.prev_goal);
        let arg_token = argstr_tokens[0];
        match goal_symbols
            .into_iter()
            .take(args.max_length)
            .enumerate()
            .find(|(_idx, symbol)| *symbol == arg_token)
        {
            Some((idx, _symbol)) => {
                return (TacticArgument::GoalToken(idx), rand_bounded_hyps!());
            }
            None => (),
        };
        let hyp_names_iter = all_hyps.iter().map(|hyp| {
            hyp.split(":")
                .next()
                .expect("Hyp string doesn't have a colon in it")
                .trim()
        });
        match hyp_names_iter
            .enumerate()
            .find(|(_idx, hname)| *hname == arg_token)
        {
            Some((idx, _hname)) => {
                if all_hyps.len() > args.max_premises {
                    let mut other_hyps = all_hyps.clone();
                    other_hyps.remove(idx);
                    let mut selected_hyps: Vec<&String> = other_hyps
                        .choose_multiple(&mut thread_rng(), args.max_premises - 1)
                        .map(|s| *s)
                        .collect();
                    let new_hyp_idx = thread_rng().gen_range(0, args.max_premises);
                    selected_hyps.insert(new_hyp_idx, all_hyps[idx]);
                    return (TacticArgument::HypVar(new_hyp_idx), selected_hyps);
                } else {
                    return (TacticArgument::HypVar(idx), all_hyps);
                }
            }
            None => (),
        };
        (TacticArgument::Unrecognized, rand_bounded_hyps!())
    }
}
fn arg_to_index(dargs: &DataloaderArgs, arg: TacticArgument) -> i64 {
    match arg {
        // For compatibility with the python version, we'll treat these as the same for now.
        TacticArgument::Unrecognized => 0,
        TacticArgument::NoArg => 0,
        TacticArgument::GoalToken(tidx) => (tidx + 1) as i64,
        TacticArgument::HypVar(hidx) => (hidx + dargs.max_length) as i64,
    }
}

fn normalize_sentence_length(mut tokenlist: Vec<i64>, length: usize, pad_value: i64) -> Vec<i64> {
    if tokenlist.len() > length {
        tokenlist.truncate(length);
    } else if tokenlist.len() < length {
        tokenlist.extend([pad_value].repeat(length - tokenlist.len()));
    }
    tokenlist
}
