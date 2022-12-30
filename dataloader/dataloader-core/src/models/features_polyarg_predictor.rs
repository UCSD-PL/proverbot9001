use indicatif::{ParallelProgressIterator, ProgressBar, ProgressFinish, ProgressStyle};
use itertools::multiunzip;
use pyo3::exceptions;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::context_filter::{apply_filter, parse_filter};
use crate::features::PickleableTokenMap as PickleableFeaturesTokenMap;
use crate::features::TokenMap as FeaturesTokenMap;
use crate::features::*;
use crate::paren_util::split_to_next_matching_paren_or_space;
use crate::scraped_data::*;
use crate::tokenizer::{
    get_words, normalize_sentence_length, IdentChunk, IdentChunkTokenizer, OpenIndexer,
    PickleableIndexer, PyIdentChunkTokenizer, Token,
};
use gestalt_ratio::gestalt_ratio;

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

pub type FPAMetadata = (OpenIndexer<String>, IdentChunkTokenizer, FeaturesTokenMap);
pub type PickleableFPAMetadata = (
    PickleableIndexer<String>,
    PyIdentChunkTokenizer,
    PickleableFeaturesTokenMap,
);

pub fn fpa_metadata_to_pickleable(metadata: FPAMetadata) -> PickleableFPAMetadata {
    (
        metadata.0.to_pickleable(),
        PyIdentChunkTokenizer::new(metadata.1),
        metadata.2.to_dicts(),
    )
}

pub fn fpa_metadata_from_pickleable(pick: PickleableFPAMetadata) -> FPAMetadata {
    (
        OpenIndexer::from_pickleable(pick.0),
        pick.1.inner.unwrap(),
        FeaturesTokenMap::from_dicts(pick.2),
    )
}

#[pyclass(module = "dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct FPAInputTensorSample {
    premise_keywords: Vec<Vec<i64>>,
    premise_subwords: Vec<Vec<Vec<i64>>>,
    premise_features: Vec<Vec<f64>>,
    num_premises: i64,

    goal_keywords: Vec<i64>,
    goal_subwords: Vec<Vec<i64>>,
    goal_mask: Vec<bool>,

    word_features: Vec<i64>,
    vec_features: Vec<f64>,
}
#[pyclass(module = "dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct FPAInputTensorDataset {
    #[pyo3(get, set)]
    premise_keywords: Vec<Vec<Vec<i64>>>,
    #[pyo3(get, set)]
    premise_subwords: Vec<Vec<Vec<Vec<i64>>>>,
    #[pyo3(get, set)]
    premise_features: Vec<Vec<Vec<f64>>>,
    #[pyo3(get, set)]
    num_premises: Vec<i64>,

    #[pyo3(get, set)]
    goal_keywords: Vec<Vec<i64>>,
    #[pyo3(get, set)]
    goal_subwords: Vec<Vec<Vec<i64>>>,
    #[pyo3(get, set)]
    goal_masks: Vec<Vec<bool>>,

    #[pyo3(get, set)]
    word_features: Vec<Vec<i64>>,
    #[pyo3(get, set)]
    vec_features: Vec<Vec<f64>>,
}

pub struct FPAOutputTensorSample {
    tactic_stem_idx: i64,
    tactic_arg_idx: i64,
}

impl FPAOutput {
    fn as_tensors(&self, dargs: &DataloaderArgs) -> FPAOutputTensorSample {
        FPAOutputTensorSample {
            tactic_stem_idx: self.tactic as i64,
            tactic_arg_idx: arg_to_index(dargs, self.argument.clone()),
        }
    }
}
#[pyclass(module = "dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct FPAOutputTensorDataset {
    #[pyo3(get, set)]
    tactic_stem_idxs: Vec<i64>,
    #[pyo3(get, set)]
    tactic_arg_idxs: Vec<i64>,
}

pub struct FPATensorSample {
    input: FPAInputTensorSample,
    output: FPAOutputTensorSample,
}

#[pyclass(module = "dataloader")]
#[derive(Serialize, Deserialize, Clone)]
pub struct FPATensorDataset {
    #[pyo3(get, set)]
    inputs: FPAInputTensorDataset,
    #[pyo3(get, set)]
    outputs: FPAOutputTensorDataset,
}

impl From<Vec<FPAInputTensorSample>> for FPAInputTensorDataset {
    fn from(input_samples: Vec<FPAInputTensorSample>) -> Self {
        let (
            prem_keys,
            prem_subs,
            prem_feats,
            num_prems,
            goal_keys,
            goal_subs,
            goal_masks,
            word_feats,
            vec_feats,
        ) = multiunzip(input_samples.into_iter().map(|input| {
            (
                input.premise_keywords,
                input.premise_subwords,
                input.premise_features,
                input.num_premises,
                input.goal_keywords,
                input.goal_subwords,
                input.goal_mask,
                input.word_features,
                input.vec_features,
            )
        }));
        FPAInputTensorDataset {
            premise_keywords: prem_keys,
            premise_subwords: prem_subs,
            premise_features: prem_feats,
            num_premises: num_prems,

            goal_keywords: goal_keys,
            goal_subwords: goal_subs,
            goal_masks,

            word_features: word_feats,
            vec_features: vec_feats,
        }
    }
}

impl From<Vec<FPAOutputTensorSample>> for FPAOutputTensorDataset {
    fn from(output_samples: Vec<FPAOutputTensorSample>) -> Self {
        let (tactic_stem_idxs, tactic_arg_idxs) = output_samples
            .into_iter()
            .map(|output| (output.tactic_stem_idx, output.tactic_arg_idx))
            .unzip();
        FPAOutputTensorDataset {
            tactic_stem_idxs,
            tactic_arg_idxs,
        }
    }
}

impl From<Vec<FPATensorSample>> for FPATensorDataset {
    fn from(samples: Vec<FPATensorSample>) -> Self {
        let (inputs, outputs): (Vec<_>, Vec<_>) = samples
            .into_iter()
            .map(|sample| (sample.input, sample.output))
            .unzip();
        FPATensorDataset {
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }
}

fn trim_premises<'a>(
    premises: &'a Vec<String>,
    tac_arg: TacticArgument,
) -> Vec<&'a String> {
    max_premises: usize,
    if premises.len() > max_premises {
        match tac_arg {
            TacticArgument::HypVar(hyp_idx) => {
                let mut other_prems = premises.clone();
                let arg_hyp = other_prems.remove(hyp_idx);
                let mut selected: Vec<&String> = other_prems
                    .choose_multiple(&mut thread_rng(), max_premises - 1)
                    .copied()
                    .collect();
                selected
                let new_arg_idx = thread_rng().gen_range(0, max_premises);
                selected.insert(new_arg_idx, arg_hyp);
            }
            _ => premises
                .choose_multiple(&mut thread_rng(), args.max_premises)
                .copied()
                .collect(),
        }
    } else if all_premises.len() == 0 {
        lazy_static! {
            static ref COLONSTRING: String = ":".to_string();
        }
        vec![&COLONSTRING]
    } else {
        all_premises.iter().collect()
    };
}

fn fpa_input_tensors(
    sample: TacticContext,
    tac_arg: TacticArgument,
    metadata: &FPAMetadata,
    args: &DataloaderArgs,
) -> FPAInputTensorSample {
    let (_, tokenizer, tokenmap) = metadata;
    let all_premises = sample
        .obligation
        .hypotheses
        .iter()
        .chain(sample.relevant_lemmas.iter())
        .collect::<Vec<_>>();
    let (word_features, vec_features) = sample_context_features(
        &args,
        tokenmap,
        &sample.relevant_lemmas,
        &sample.prev_tactics,
        &sample.obligation.hypotheses,
        &sample.obligation.goal,
    );
    let (goal_keywords, goal_subwords) = normalize_sentence_length(
        tokenizer.tokenize(&sample.obligation.goal),
        args.max_length,
        args.max_subwords,
        0,
    )
    .into_iter()
    .unzip();
    let goal_mask = get_goal_mask(&sample.obligation.goal, args.max_length);
    let selected_prems: Vec<&String> = if all_premises.len() > args.max_premises {
        match tac_arg {
            TacticArgument::HypVar(hyp_idx) => {
                let mut other_prems = all_premises.clone();
                let arg_hyp = other_prems.remove(hyp_idx);
                let mut selected: Vec<&String> = other_prems
                    .choose_multiple(&mut thread_rng(), args.max_premises - 1)
                    .copied()
                    .collect();
                selected.insert(thread_rng().gen_range(0, args.max_premises), arg_hyp);
                selected
            }
            _ => all_premises
                .choose_multiple(&mut thread_rng(), args.max_premises)
                .copied()
                .collect(),
        }
    } else if all_premises.len() == 0 {
        lazy_static! {
            static ref COLONSTRING: String = ":".to_string();
        }
        vec![&COLONSTRING]
    } else {
        all_premises.clone()
    };
    let (premise_keywords, premise_subwords): (Vec<Vec<_>>, Vec<Vec<_>>) = selected_prems
        .iter()
        .map(|prem| {
            normalize_sentence_length(
                tokenizer.tokenize(get_hyp_type(prem)),
                args.max_length,
                args.max_subwords,
                0,
            )
            .into_iter()
            .unzip()
        })
        .unzip();

    let premise_features: Vec<Vec<f64>> = score_hyps(
        &selected_prems.iter().cloned().cloned().collect(),
        &sample.obligation.goal,
    )
    .into_iter()
    .zip(selected_prems)
    .map(|(score, hyp)| vec![score, equality_hyp_feature(hyp, &sample.obligation.goal)])
    .collect();
    FPAInputTensorSample {
        premise_keywords,
        premise_subwords,
        premise_features,
        num_premises: all_premises.len() as i64,
        goal_keywords,
        goal_subwords,
        goal_mask,
        word_features,
        vec_features,
    }
}

// Note: This is probably wrong since the premises get trimmed, invalidating indices.
fn parse_fpa_argument(
    dargs: &DataloaderArgs,
    hyps: Vec<String>,
    goal: &str,
    arg: &str,
) -> TacticArgument {
    let targ = arg.trim();
    let argstr_tokens: Vec<&str> = targ[..targ.len() - 1].split_whitespace().collect();
    if argstr_tokens.len() == 0 {
        TacticArgument::NoArg
    } else if argstr_tokens.len() > 1 {
        panic!(
            "A multi argument tactic made it past the context filter! arg is {}",
            arg
        )
    } else {
        let goal_symbols = get_words(goal);
        let arg_token = argstr_tokens[0];
        match goal_symbols
            .into_iter()
            .take(dargs.max_length)
            .enumerate()
            .find(|(_idx, symbol)| symbol_matches(*symbol, arg_token))
        {
            Some((idx, _symbol)) => {
                return TacticArgument::GoalToken(idx);
            }
            None => (),
        }
        match indexed_premises(hyps.iter().map(|s| s.as_ref()))
            .into_iter()
            .find(|(_idx, hname)| *hname == arg_token)
        {
            Some((idx, _hname)) => {
                return TacticArgument::HypVar(idx);
            }
            None => panic!(
                "An unknown tactic made it past the context filter with args: {}\n\
                            Hyps are {:?}\n\
                            Goal is {}",
                arg, hyps, goal
            ),
        }
    }
}
fn parse_tactic(
    tactic: &str,
    context: TacticContext,
    metadata: &FPAMetadata,
    dargs: &DataloaderArgs,
) -> FPAOutput {
    let (tac, arg) = split_tactic(tactic).expect(&format!("Couldn't split tactic {}", tactic));
    FPAOutput {
        tactic: metadata.0.lookup(tac.to_string()) as usize,
        argument: parse_fpa_argument(
            dargs,
            context.obligation.hypotheses,
            &context.obligation.goal,
            &arg,
        ),
    }
}

fn fpa_tensors(
    sample: ScrapedTactic,
    metadata: &FPAMetadata,
    args: &DataloaderArgs,
) -> FPATensorSample {
    let sample_output: FPAOutput =
        parse_tactic(&sample.tactic, sample.clone().into(), metadata, args);
    FPATensorSample {
        input: fpa_input_tensors(
            sample.into(),
            sample_output.argument.clone(),
            metadata,
            args,
        ),
        output: sample_output.as_tensors(args),
    }
}

pub fn features_polyarg_tensors_rs(
    args: DataloaderArgs,
    filename: String,
    metadata: Option<PickleableFPAMetadata>,
) -> PyResult<(PickleableFPAMetadata, FPATensorDataset, (Vec<i64>, i64))> {
    let filter = parse_filter(&args.context_filter);
    let my_bar_style = ProgressStyle::with_template("{msg}: {wide_bar} [{elapsed}/{eta}]").unwrap();
    let spinner = ProgressBar::new(341028)
        .with_message("Loading data from file")
        .with_style(my_bar_style.clone());

    let mut raw_data_iter = scraped_from_file(
        File::open(filename)
            .map_err(|_err| exceptions::PyValueError::new_err("Failed to open file"))?,
    )
    .flat_map(|datum| match datum {
        ScrapedData::Vernac(_) => None,
        ScrapedData::Tactic(t) => {
            spinner.inc(1);
            Some(t)
        }
    })
    .flat_map(preprocess_datum)
    .filter(|datum| apply_filter(args.max_length, &filter, datum));

    let features_data_sample: Vec<ScrapedTactic> = raw_data_iter.by_ref().take(4096).collect();
    // Put the data used for constructing the features metadata back
    // at the beginning of the raw data iter.
    let raw_data_iter = features_data_sample.iter().cloned().chain(raw_data_iter);

    let raw_data: Vec<ScrapedTactic> = match args.max_tuples {
        Some(max) => raw_data_iter.take(max).collect(),
        None => raw_data_iter.collect(),
    };
    let length: u64 = raw_data.len().try_into().unwrap();
    let (indexer, rest_meta) = match metadata {
        Some((indexer, tokenizer, tmap)) => (
            OpenIndexer::from_pickleable(indexer),
            Some((tokenizer, tmap)),
        ),
        None => match &args.load_embedding {
            Some(path) => {
                let embedding = OpenIndexer::<String>::load_from_text(path);
                // embedding.freeze();
                (embedding, None)
            }
            None => {
                let mut indexer = OpenIndexer::new();
                for sample in features_data_sample.iter() {
                    match get_stem(&sample.tactic) {
                        Some(stem) => indexer.add(stem),
                        None => (),
                    }
                }
                (indexer, None)
            }
        },
    };
    let (tokenizer, features_token_map) = match rest_meta {
        Some((ptok, ptmap)) => (ptok.inner.unwrap(), FeaturesTokenMap::from_dicts(ptmap)),
        None => {
            let use_unknowns = false;
            let num_reserved_tokens = 2;
            let keywords_file = args
                .keywords_file
                .clone()
                .expect("No keywords file passed!");
            let subwords_file = args
                .subwords_file
                .clone()
                .expect("No subwords file passed!");
            let tokenizer = IdentChunkTokenizer::new_from_files(
                use_unknowns,
                num_reserved_tokens,
                &keywords_file,
                &subwords_file,
            );
            let tmap = match &args.load_features_state {
                Some(path) => FeaturesTokenMap::load_from_text(path),
                None => FeaturesTokenMap::initialize(&features_data_sample, args.num_keywords),
            };
            (tokenizer, tmap)
        }
    };
    match &args.save_features_state {
        Some(path) => features_token_map.save_to_text(path),
        None => (),
    };

    let metadata = (indexer, tokenizer, features_token_map.clone());
    let samples: Vec<FPATensorSample> = raw_data
        .into_par_iter()
        .map(|sample| fpa_tensors(sample, &metadata, &args))
        .progress_with(
            ProgressBar::new(length)
                .with_message("Processing samples")
                .with_style(my_bar_style)
                .with_finish(ProgressFinish::AndLeave),
        )
        .collect();
    Ok((
        fpa_metadata_to_pickleable(metadata),
        samples.into(),
        (features_token_map.word_features_sizes(), VEC_FEATURES_SIZE),
    ))
}

/// This function is for debugging purposes
#[allow(dead_code)]
pub fn lookup_hyp(premises: Vec<String>, hyp_name: &str) -> String {
    let hyp_idx = indexed_premises(premises.iter().map(|s| s.as_ref()))
        .into_iter()
        .find(|(_idx, hname)| *hname == hyp_name)
        .expect(&format!("Couldn't find a hyp with name {}", hyp_name))
        .0;
    premises[hyp_idx].clone()
}

fn get_goal_mask(goal: &str, max_length: usize) -> Vec<bool> {
    lazy_static! {
        static ref STARTS_WITH_LETTER: Regex =
            Regex::new(r"^\w.*").expect("Couldn't compile regex");
    }

    let words = get_words(goal);
    let mut mask_vec: Vec<_> = words
        .into_iter()
        .take(max_length)
        .map(|goal_word| STARTS_WITH_LETTER.is_match(goal_word))
        .collect();
    if mask_vec.len() < max_length {
        mask_vec.extend([false].repeat(max_length - mask_vec.len()));
    }
    mask_vec.insert(0, true);
    mask_vec
}

pub fn tokenize_fpa(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    term: String,
) -> Vec<IdentChunk> {
    let (_indexer, tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
    normalize_sentence_length(
        tokenizer.tokenize(&term),
        args.max_length,
        args.max_subwords,
        0,
    )
}

pub fn get_premise_features_rs(
    _args: DataloaderArgs,
    _metadata: PickleableFPAMetadata,
    goal: String,
    premise: String,
) -> FloatTensor1D {
    let score = gestalt_ratio(&goal, get_hyp_type(&premise));
    let eq_feat = equality_hyp_feature(&premise, &goal);
    vec![score, eq_feat]
}
pub fn get_premise_features_size_rs(
    _args: DataloaderArgs,
    _metadata: PickleableFPAMetadata,
) -> i64 {
    2
}

pub fn sample_fpa_batch_rs(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    context_batch: Vec<TacticContext>,
) -> FPAInputTensorDataset {
    let (_indexer, tokenizer, ftmap) = fpa_metadata_from_pickleable(metadata);
    // context_batch.par_iter().map(|sample| fpa_input_tensors(sample,
    let (word_features_batch, vec_features_batch) = context_batch
        .iter()
        .map(|ctxt| {
            sample_context_features_rs(
                &args,
                &ftmap,
                &ctxt.relevant_lemmas,
                &ctxt.prev_tactics,
                &ctxt.obligation.hypotheses,
                &ctxt.obligation.goal,
            )
        })
        .unzip();

    let premises_batch: Vec<Vec<String>> = context_batch
        .par_iter()
        .map(|ctxt| {
            ctxt.obligation
                .hypotheses
                .iter()
                .chain(ctxt.relevant_lemmas.iter())
                .map(|p| p.clone())
                .collect()
        })
        .collect();

    let premise_scores_batch: Vec<Vec<f64>> = premises_batch
        .par_iter()
        .zip(context_batch.par_iter())
        .map(|(premises, context)| score_hyps(premises, &context.obligation.goal))
        .collect();

    let premise_features_batch = premises_batch
        .par_iter()
        .zip(premise_scores_batch.par_iter())
        .zip(context_batch.par_iter())
        .map(|((premises, scores), ctxt)| {
            premises
                .iter()
                .zip(scores.iter())
                .map(|(premise, score)| {
                    vec![*score, equality_hyp_feature(premise, &ctxt.obligation.goal)]
                })
                .collect()
        })
        .collect();

    let tgoals_batch = context_batch
        .par_iter()
        .map(|ctxt| {
            normalize_sentence_length(
                tokenizer.tokenize(&ctxt.obligation.goal),
                args.max_length,
                args.max_subwords,
                0,
            )
            .into_iter()
            .unzip()
        })
        .unzip();
    let goal_symbols_mask: BoolTensor2D = context_batch
        .par_iter()
        .map(|ctxt| get_goal_mask(&ctxt.obligation.goal, args.max_length))
        .collect();
    let tprems_batch: (Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<Vec<i64>>>>) = premises_batch
        .into_iter()
        .map(|premises| {
            premises
                .into_iter()
                .map(|premise| {
                    normalize_sentence_length(
                        tokenizer.tokenize(get_hyp_type(&premise)),
                        args.max_length,
                        args.max_subwords,
                        0,
                    )
                    .into_iter()
                    .unzip()
                })
                .unzip()
        })
        .unzip();

    let num_hyps_batch = tprems_batch
        .0
        .iter()
        .map(|tprems| tprems.len() as i64)
        .collect();

    (
        tprems_batch,
        premise_features_batch,
        num_hyps_batch,
        tgoals_batch,
        goal_symbols_mask,
        word_features_batch,
        vec_features_batch,
    )
}

pub fn sample_fpa_rs(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    relevant_lemmas: Vec<String>,
    prev_tactics: Vec<String>,
    hypotheses: Vec<String>,
    goal: String,
) -> (
    (Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<Vec<i64>>>>),
    FloatUnpaddedTensor3D,
    LongTensor1D,
    (Vec<Vec<i64>>, Vec<Vec<Vec<i64>>>),
    BoolTensor2D,
    LongTensor2D,
    FloatTensor2D,
) {
    let (_indexer, tokenizer, ftmap) = fpa_metadata_from_pickleable(metadata);
    let (word_features, vec_features) = sample_context_features_rs(
        &args,
        &ftmap,
        &relevant_lemmas,
        &prev_tactics,
        &hypotheses,
        &goal,
    );
    let all_premises: Vec<String> = hypotheses
        .into_iter()
        .chain(relevant_lemmas.into_iter())
        .collect();
    let premise_scores = score_hyps(&all_premises, &goal);
    let premise_features = all_premises
        .iter()
        .zip(premise_scores.iter())
        .map(|(premise, score)| vec![*score, equality_hyp_feature(premise, &goal)])
        .collect();
    let tokenized_goal = normalize_sentence_length(
        tokenizer.tokenize(&goal),
        args.max_length,
        args.max_subwords,
        0,
    )
    .into_iter()
    .unzip();

    let goal_symbols_mask = get_goal_mask(&goal, args.max_length);

    let tokenized_premises: (Vec<Vec<i64>>, Vec<Vec<Vec<i64>>>) = all_premises
        .into_iter()
        .map(|premise| {
            normalize_sentence_length(
                tokenizer.tokenize(get_hyp_type(&premise)),
                args.max_length,
                args.max_subwords,
                0,
            )
            .into_iter()
            .unzip()
        })
        .unzip();
    let num_hyps = tokenized_premises.0.len();
    (
        (vec![tokenized_premises.0], vec![tokenized_premises.1]),
        vec![premise_features],
        vec![num_hyps as i64],
        (vec![tokenized_goal.0], vec![tokenized_goal.1]),
        vec![goal_symbols_mask],
        vec![word_features],
        vec![vec_features],
    )
}

pub fn decode_fpa_result_rs(
    args: DataloaderArgs,
    metadata: PickleableFPAMetadata,
    premises: Vec<String>,
    goal: &str,
    tac_idx: i64,
    arg_idx: i64,
) -> String {
    let stem = decode_fpa_stem_rs(&args, metadata, tac_idx);
    let arg = decode_fpa_arg_rs(&args, premises, goal, arg_idx);
    if arg == "" {
        format!("{}.", stem)
    } else {
        format!("{} {}.", stem, arg)
    }
}

pub fn decode_fpa_stem_rs(
    _args: &DataloaderArgs,
    metadata: PickleableFPAMetadata,
    tac_idx: i64,
) -> String {
    let (indexer, _tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
    indexer.reverse_lookup(tac_idx)
}

pub fn encode_fpa_stem_rs(
    _args: &DataloaderArgs,
    metadata: PickleableFPAMetadata,
    tac_stem: String,
) -> i64 {
    let (indexer, _tokenizer, _ftmap) = fpa_metadata_from_pickleable(metadata);
    indexer.lookup(tac_stem)
}

pub fn decode_fpa_arg_rs(
    args: &DataloaderArgs,
    premises: Vec<String>,
    goal: &str,
    arg_idx: i64,
) -> String {
    let argtype = if arg_idx == 0 {
        TacticArgument::NoArg
    } else if (arg_idx as usize) <= args.max_length {
        TacticArgument::GoalToken(arg_idx as usize - 1)
    } else {
        TacticArgument::HypVar((arg_idx as usize) - args.max_length - 1)
    };
    match argtype {
        TacticArgument::NoArg => "".to_string(),
        TacticArgument::Unrecognized => "".to_string(),
        TacticArgument::GoalToken(tidx) => {
            // assert!(tidx < get_words(goal).len(), format!("{}, {:?}, {}", goal, get_words(goal), tidx));
            if tidx >= get_words(goal).len() {
                "<INVALID>".to_string()
            } else {
                get_words(goal)[tidx].to_string()
            }
        }
        TacticArgument::HypVar(hidx) => {
            assert!(hidx < premises.len());
            let all_vars = premises[hidx]
                .split(":")
                .next()
                .expect("No colon in hyp")
                .trim();
            if all_vars.contains(",") {
                all_vars.split(",").next().unwrap().to_string()
            } else {
                all_vars.to_string()
            }
        }
    }
}

fn equality_hyp_feature(hyp: &str, goal: &str) -> f64 {
    lazy_static! {
        static ref EQ: Regex = Regex::new(r"^\s*eq\s+(.*)").expect("Can't build eq regex");
    }
    let normalized_hyp_type = get_hyp_type(hyp).replace("\n", " ");
    let normalized_goal = goal.replace("\n", " ");
    let equals_match = EQ.captures(&normalized_hyp_type);
    if let Some(captures) = equals_match {
        let normalized_string = captures
            .get(1)
            .expect("Can't get capture group")
            .as_str()
            .replace("\n", " ");
        let (left_side, right_side) = split_to_next_matching_paren_or_space(&normalized_string);
        if normalized_goal.contains(left_side.trim()) && right_side != "" {
            -1.0
        } else if normalized_goal.contains(right_side.trim()) && right_side != "" {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    }
}

pub fn fpa_get_num_possible_args_rs(args: &DataloaderArgs) -> i64 {
    (args.max_length + args.max_premises + 1) as i64
}

pub fn encode_fpa_arg_unbounded(
    args: &DataloaderArgs,
    hyps: Vec<String>,
    goal: &str,
    arg: &str,
) -> Result<i64, String> {
    let targ = arg.trim();
    let argstr_tokens: Vec<&str> = targ[..targ.len() - 1].split_whitespace().collect();
    if argstr_tokens.len() == 0 {
        Ok(arg_to_index(args, TacticArgument::NoArg))
    } else if argstr_tokens.len() > 1 {
        Err(format!(
            "A multi argument tactic made it past the context filter! arg is {}",
            arg
        ))
    } else {
        let goal_symbols = get_words(goal);
        let arg_token = argstr_tokens[0];
        match goal_symbols
            .into_iter()
            .take(args.max_length)
            .enumerate()
            .find(|(_idx, symbol)| symbol_matches(*symbol, arg_token))
        {
            Some((idx, _symbol)) => {
                return Ok(arg_to_index(args, TacticArgument::GoalToken(idx)));
            }
            None => (),
        }
        match indexed_premises(hyps.iter().map(|s| s.as_ref()))
            .into_iter()
            .find(|(_idx, hname)| *hname == arg_token)
        {
            Some((idx, _hname)) => {
                return Ok(arg_to_index(args, TacticArgument::HypVar(idx)));
            }
            None => Err(format!(
                "An unknown tactic made it past the context filter with args: {}\n\
                            Hyps are {:?}\n\
                            Goal is {}",
                arg, hyps, goal
            )),
        }
    }
}

fn get_argument<'a>(
    args: &DataloaderArgs,
    scraped: &'a ScrapedTactic,
) -> (TacticArgument, Vec<&'a String>) {
    let all_hyps: Vec<&String> = scraped
        .context
        .focused_hyps()
        .iter()
        .chain(scraped.relevant_lemmas.iter())
        .collect();
    macro_rules! rand_bounded_hyps {
        () => {
            if all_hyps.len() > args.max_premises {
                // all_hyps.iter().take(args.max_premises).cloned().collect()
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
        panic!(
            "A multi argument tactic made it past the context filter! {}",
            scraped.tactic
        )
    } else {
        let goal_symbols = get_words(scraped.context.focused_goal());
        let arg_token = argstr_tokens[0];
        match goal_symbols
            .into_iter()
            .take(args.max_length)
            .enumerate()
            .find(|(_idx, symbol)| symbol_matches(*symbol, arg_token))
        {
            Some((idx, _symbol)) => {
                return (TacticArgument::GoalToken(idx), rand_bounded_hyps!());
            }
            None => (),
        };
        match indexed_premises(all_hyps.iter().map(|s| s.as_ref()))
            .into_iter()
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
                    // let mut selected_hyps: Vec<&String> =
                    //     other_hyps.into_iter().take(args.max_premises - 1).collect();
                    let new_hyp_idx = thread_rng().gen_range(0, args.max_premises);
                    // let new_hyp_idx = args.max_premises - 1;
                    selected_hyps.insert(new_hyp_idx, all_hyps[idx]);
                    return (TacticArgument::HypVar(new_hyp_idx), selected_hyps);
                } else {
                    return (TacticArgument::HypVar(idx), all_hyps);
                }
            }
            None => (),
        };
        assert!(
            false,
            "An unknown tactic made it past the context filter: {}, arg {}, arg_token {}\n\
             Goal is {}",
            scraped.tactic,
            argstr_tokens[0],
            arg_token,
            scraped.context.focused_goal()
        );
        (TacticArgument::Unrecognized, rand_bounded_hyps!())
    }
}
fn arg_to_index(dargs: &DataloaderArgs, arg: TacticArgument) -> i64 {
    match arg {
        // For compatibility with the python version, we'll treat these as the same for now.
        TacticArgument::Unrecognized => 0,
        TacticArgument::NoArg => 0,
        TacticArgument::GoalToken(tidx) => (tidx + 1) as i64,
        TacticArgument::HypVar(hidx) => (hidx + dargs.max_length + 1) as i64,
    }
}
