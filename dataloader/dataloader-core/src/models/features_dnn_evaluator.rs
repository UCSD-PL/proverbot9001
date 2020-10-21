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

use pyo3::exceptions;
use pyo3::prelude::*;

use std::fs::File;

use crate::models::evaluator_common::*;
use crate::features::{context_features, TokenMap, VEC_FEATURES_SIZE};
use crate::scraped_data::*;

pub fn features_to_total_distances_tensors(
    args: DataloaderArgs,
    filename: String,
    map: Option<TokenMap>,
) -> PyResult<(
    TokenMap,
    LongTensor2D,
    FloatTensor2D,
    FloatTensor2D,
    Vec<i64>,
    i64,
)> {
    match File::open(filename) {
        Result::Ok(file) => {
            let scraped = scraped_from_file(file).collect();
            let distanced = tactic_distances(scraped);
            let (tactics, distances): (Vec<ScrapedTactic>, Vec<usize>) =
                distanced.into_iter().unzip();
            let outputs = normalize_distances(args.max_distance, distances)
                .into_iter()
                .map(|distance| vec![distance])
                .collect();
            let tmap = match map {
                Some(m) => m,
                None => TokenMap::initialize(&tactics, args.num_keywords),
            };
            let (word_features, float_features) = context_features(&args, &tmap, &tactics);
            let word_features_sizes = tmap.word_features_sizes();

            Ok((
                tmap,
                word_features,
                float_features,
                outputs,
                word_features_sizes,
                VEC_FEATURES_SIZE,
            ))
        }
        Result::Err(_err) => Err(PyErr::new::<exceptions::TypeError, _>(
            "Failed to open file",
        )),
    }
}
