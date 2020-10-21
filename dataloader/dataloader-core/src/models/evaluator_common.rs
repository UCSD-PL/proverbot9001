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
use std::cmp::min;

pub fn normalize_distances(max_distance: usize, distances: Vec<usize>) -> Vec<f64> {
    distances
        .into_iter()
        .map(|x| (min(x, max_distance) as f64) / (max_distance as f64))
        .collect()
}

pub fn tactic_distances(scraped_data: Vec<ScrapedData>) -> Vec<(ScrapedTactic, usize)> {
    let mut in_proof = false;
    let mut interaction_buffer = Vec::new();
    let mut blocks = Vec::new();

    for interaction in scraped_data {
        match interaction {
            ScrapedData::Tactic(tac) => {
                if !in_proof {
                    interaction_buffer.clear();
                    in_proof = true;
                }
                interaction_buffer.push(tac)
            }
            ScrapedData::Vernac(_cmd) => {
                if in_proof {
                    blocks.push(interaction_buffer.clone());
                    in_proof = false;
                }
            }
        }
    }

    let mut result = Vec::new();

    for block in blocks {
        let mut distanced_block: Vec<(ScrapedTactic, usize)> = label_block_distances(block);
        result.append(&mut distanced_block);
    }
    return result;
}

fn label_block_distances(block: Vec<ScrapedTactic>) -> Vec<(ScrapedTactic, usize)> {
    let mut path_segments: Vec<Vec<ScrapedTactic>> = vec![Vec::new()];
    let mut closed_distances: Vec<usize> = vec![0, 0];
    let mut finished_segments: Vec<Vec<(ScrapedTactic, usize)>> = vec![Vec::new(), Vec::new()];

    let close_goal =
        |path_segments: &mut Vec<Vec<ScrapedTactic>>,
         closed_distances: &mut Vec<usize>,
         finished_segments: &mut Vec<Vec<(ScrapedTactic, usize)>>| {
            let last_segment = path_segments.pop().expect("Not enough path segments");
            let last_segment_len = last_segment.len();
            let last_distance = closed_distances.pop().expect("Not enough closed distances");
            let mut closed_tacs: Vec<(ScrapedTactic, usize)> = last_segment
                .into_iter()
                .rev()
                .zip((1 + last_distance)..)
                .collect::<Vec<(ScrapedTactic, usize)>>()
                .into_iter()
                .rev()
                .collect();

            let mut already_closed_tacs = finished_segments
                .pop()
                .expect("Not enough finished segments");
            let last_finished_segment = finished_segments
                .last_mut()
                .expect("Not enough finished segments");
            last_finished_segment.append(&mut closed_tacs);
            last_finished_segment.append(&mut already_closed_tacs);
            let next_last_distance = closed_distances
                .last_mut()
                .expect("Not enough closed distances");
            *next_last_distance += last_distance + last_segment_len
        };

    for interaction in block.into_iter() {
        let trimmed_tac = interaction.tactic.trim();
        if trimmed_tac == "{" {
            path_segments.push(Vec::new());
            closed_distances.push(0);
            finished_segments.push(Vec::new());
        } else if trimmed_tac == "}" {
            close_goal(
                &mut path_segments,
                &mut closed_distances,
                &mut finished_segments,
            );
        } else if trimmed_tac == "Qed." {
            close_goal(
                &mut path_segments,
                &mut closed_distances,
                &mut finished_segments,
            );
            {
                let last_finished_segment = finished_segments
                    .last_mut()
                    .expect("Not enougn finished segments");
                last_finished_segment.push((interaction, 0));
            }
            return finished_segments.pop().unwrap();
        } else {
            let last_path_segment = path_segments
                .last_mut()
                .expect("Not enougn finished segments");
            last_path_segment.push(interaction);
        }
    }
    assert_eq!(path_segments.len(), 1);
    close_goal(
        &mut path_segments,
        &mut closed_distances,
        &mut finished_segments,
    );
    assert_eq!(finished_segments.len(), 1);
    finished_segments
        .pop()
        .expect("Not enough finished segments")
}
